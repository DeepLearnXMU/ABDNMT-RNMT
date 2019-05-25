import argparse
import copy
import json
import os
import shutil
import sys
import traceback
from collections import OrderedDict, defaultdict
from datetime import timedelta
from time import strftime, localtime

import torch
import torch.nn as nn
import torch.optim.lr_scheduler
from logbook import StreamHandler, Logger, StderrHandler, WARNING
from lunas.iterator import Iterator
from lunas.readers import Zip, TextLine, Shuffle
from nltk.translate.bleu_score import corpus_bleu
from tensorboardX import SummaryWriter

import options
import thseq.models as models
import thseq.optim as optim
import thseq.optim.lr_scheduler
from state import State, Loader
from thseq.data.plain import restore_bpe
from thseq.data.plain import text_to_indices
from thseq.data.vocabulary import Vocabulary
from thseq.utils.meters import AverageMeter, StopwatchMeter, SpeedMeter
from thseq.utils.misc import aggregate_value_by_key
from thseq.utils.misc import set_seed
from thseq.utils.tensor import cuda
from thseq.utils.tensor import pack_tensors

logger = Logger()


def override(config_to, config_from, only_override_null=True):
    config_to = copy.deepcopy(config_to)
    config_from = copy.deepcopy(config_from)

    if not config_from:
        return config_to
    if not config_to:
        return config_from

    assert isinstance(config_from, argparse.Namespace)

    assert isinstance(config_to, argparse.Namespace)
    for k, v in config_from.__dict__.items():
        if v is None:
            continue
        if only_override_null:
            if getattr(config_to, k) is None:
                setattr(config_to, k, v)
                # elif k in immutable_keys:
                #     sys.stderr.write('Warning: trying to override immutable key: {}\n'.format(k))
        else:
            setattr(config_to, k, v)
    return config_to


class Trainer(object):
    def __init__(self, args, model, optimizer, criterion, lr_scheduler):
        super().__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler

        self._buffered_stats = defaultdict(lambda: [])

        # initialize meters
        self.meters = OrderedDict()
        self.meters['wps'] = SpeedMeter()  # words per second
        self.meters['gnorm'] = AverageMeter()  # gradient norm
        self.meters['clip'] = AverageMeter()  # % of steps clipped
        self.meters['oom'] = AverageMeter()  # out of memory

        self.count_oom = 0

    def clear_buffered_stats(self):
        self._buffered_stats.clear()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        return self.lr_scheduler.step(epoch, val_loss)

    def lr_step_update(self, num_steps):
        """Update the learning rate after each step."""
        return self.lr_scheduler.step_update(num_steps)

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the model replica."""
        return self.model

    def get_meter(self, name):
        """Get a specific meter by name."""
        if name not in self.meters:
            return None
        return self.meters[name]

    def evaluate(self, iterator: Iterator, beam_size=4, bpe=True, r2l=False):
        self.model.eval()

        all_hyps = []
        all_refs = []
        for batch in iterator.iter_epoch():
            outputs = self.model.translate(batch.data['src'], beam_size)
            # pre-processing of the target side strictly following this order:
            # 1. bpe
            # 2. reverse
            if r2l:
                outputs = list(map(lambda x: x[::-1], outputs))
            if bpe:
                for i, output in enumerate(outputs):
                    outputs[i] = restore_bpe(' '.join(output)).split()
            outputs = batch.revert(outputs)
            refs = batch.revert(batch.data['refs'])
            all_hyps.extend(outputs)
            all_refs.extend(refs)
        bleu = corpus_bleu(all_refs, all_hyps)
        score = bleu * 100

        return score

    def scale_clip_grad_(self, normalization):
        parameters = list(filter(lambda p: p.grad is not None, self.model.parameters()))
        # normalize gradients by normalization
        for p in parameters:
            p.grad.data.div_(normalization)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.args.clip_norm)
        return grad_norm

    def train_step(self, samples):
        """Do forward, backward and parameter update."""
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints

        # forward and backward pass
        self.meters['wps'].start()
        logs = []
        sample_sizes_r2l = []
        sample_sizes_l2r = []
        oom = 0
        losses = []
        for sample in samples:
            try:
                output = get_log(self.args, self.model, sample)
                loss_r2l, loss_l2r, sample_size_r2l, sample_size_l2r, nll_loss, log = output
                loss = loss_l2r / sample_size_l2r + loss_r2l / sample_size_r2l
                loss.backward()

                sample_sizes_r2l.append(sample_size_r2l)
                sample_sizes_l2r.append(sample_size_l2r)
                logs.append(log)
                losses.append(loss.item())

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    oom += 1
                    self.model.zero_grad()
                else:
                    raise e
        if oom == len(samples):
            logger.warn('Ran out of memory, skipping batch')
            return None
        # gather logging outputs from all replicas
        log = agg_logs(logs)
        log['loss'] = sum(losses)/len(losses)

        try:
            # all-reduce and rescale gradients, then take an optimization step
            grad_norm = self.scale_clip_grad_(len(samples) - oom)
            # take an optimization step
            self.optimizer.step()
            self.optimizer.zero_grad()
            ntok_l2r = log.get('ntok_l2r', 0)
            nsentences = log.get('nsentences', 0)
            # update meters
            self.meters['wps'].stop(ntok_l2r)
            if grad_norm is not None:
                self.meters['gnorm'].update(grad_norm)
                self.meters['clip'].update(1. if grad_norm > self.args.clip_norm else 0.)
            self.meters['oom'].update(oom)

        except OverflowError as e:
            self.optimizer.zero_grad()
            logger.warn('Overflow detected during parameters update, {str(e)}')
        except RuntimeError as e:
            if 'out of memory' in str(e):
                self.model.zero_grad()
                return None
            else:
                raise e

        return log

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def reset_meters(self):
        for meter in self.meters.values():
            meter.reset()


def init_parameters(module):
    if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        return
    else:
        for name, para in module._parameters.items():
            if para is not None and para.requires_grad:
                if para.dim() >= 2:
                    nn.init.uniform_(para, -0.08, 0.08)
                else:
                    para.data.zero_()


def check_cuda_availability():
    if not torch.cuda.is_available():
        logger.warn('Training on CPU')

    for i in range(torch.cuda.device_count()):
        logger.info(f'Using device: {torch.cuda.get_device_name(i)}')


def stat_parameters(model):
    num_params = 0
    para_stats = []
    num_params_requires_grad = 0
    for n, p in model.named_parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_requires_grad += p.numel()
        para_stats.append(f'{n}: {p.shape}, '
                          f'requires_grad={p.requires_grad}, '
                          f'mean/min/max={p.mean():.3f}/{p.min():.3f}/{p.max():.3f}')
    # logger.info(f'Parameters\n{os.linesep.join(para_stats)}')

    check_cuda_availability()

    if num_params == num_params_requires_grad:
        logger.info(f'Num. of parameters: {num_params}')
    else:
        logger.info(f'Num. of optimizable parameters: {num_params_requires_grad}')
        logger.info(f'Num. of parameters: {num_params}')


def get_train_iterator(args, source_vocab: Vocabulary, target_vocab: Vocabulary):
    threads = args.num_workers

    src = TextLine(args.train[0], bufsize=args.buffer_size, num_threads=threads)
    r2l = TextLine(args.train[1], bufsize=args.buffer_size, num_threads=threads)
    l2r = TextLine(args.train[2], bufsize=args.buffer_size, num_threads=threads)
    ds = Zip([src, r2l, l2r], bufsize=args.buffer_size, num_threads=threads)

    def fn(src, r2l, l2r):
        src = torch.as_tensor(text_to_indices(src, source_vocab))
        r2l = torch.as_tensor(text_to_indices(r2l, target_vocab))
        l2r = torch.as_tensor(text_to_indices(l2r, target_vocab))
        n_src_tok = src.size(0)
        n_trg_tok_r2l = r2l.size(0)
        n_trg_tok_l2r = l2r.size(0)

        return {
            'src': src,
            'r2l': r2l,
            'l2r': l2r,
            'n_src_tok': n_src_tok,
            'ntok_r2l': n_trg_tok_r2l,
            'ntok_l2r': n_trg_tok_l2r,
        }

    ds = ds.select(fn)
    shuffle = args.shuffle
    if shuffle != 0:
        ds = Shuffle(ds, shufsize=shuffle, bufsize=args.buffer_size, num_threads=threads)

    # limit = args.length_limit
    # if limit is not None and len(limit) == 1:
    #     limit *= len(args.train)
    # if limit is not None:
    #     ds = ds.where(
    #         lambda x: x['n_src_tok'] - 1 <= limit[0] and x['n_trg_tok'] - 1 <= limit[1]
    #     )

    def collate_fn(xs):
        return {
            'src': cuda(pack_tensors(aggregate_value_by_key(xs, 'src'), source_vocab.pad_id)),
            'r2l': cuda(pack_tensors(aggregate_value_by_key(xs, 'r2l'), target_vocab.pad_id)),
            'l2r': cuda(pack_tensors(aggregate_value_by_key(xs, 'l2r'), target_vocab.pad_id)),
            'ntok_src': aggregate_value_by_key(xs, 'n_src_tok', sum),
            'ntok_r2l': aggregate_value_by_key(xs, 'ntok_r2l', sum),
            'ntok_l2r': aggregate_value_by_key(xs, 'ntok_l2r', sum),
        }

    sample_size_fn = None
    if not args.batch_by_sentence:
        sample_size_fn = lambda x: x['ntok_l2r']

    batch_size = args.batch_size[0]
    padded_size = None
    padded_size_fn = lambda xs: 0 if not xs else \
        max(xs, key=lambda x: x['ntok_l2r'])['ntok_l2r'] * len(xs)

    if torch.cuda.is_available():
        batch_size *= torch.cuda.device_count()
        if len(args.batch_size) > 1:
            padded_size = args.batch_size[1]
            padded_size *= torch.cuda.device_count()
    # padded_size = None

    iterator = Iterator(
        ds, batch_size,
        padded_size=padded_size,
        cache_size=max(args.sort_buffer_factor, 1) * batch_size,
        sample_size_fn=sample_size_fn,
        padded_size_fn=padded_size_fn,
        collate_fn=collate_fn,
        sort_cache_by=lambda sample: sample['ntok_l2r'],
        sort_batch_by=lambda sample: -sample['n_src_tok'],
        # sort_cache_by=lambda sample: -sample['n_src_tok'],
        strip_batch=True
    )

    return iterator


def get_dev_iterator(args, vocs):
    svoc, tvoc = vocs

    n_input = 2 if args.arch in ['abdrnn2', ] else 1
    dev = args.dev[:n_input]
    refs = args.dev[n_input:]

    if n_input == 1:
        vocabs = [svoc]
    elif n_input == 2:
        vocabs = [svoc, tvoc]
    else:
        raise NotImplementedError('a maximum of 2 sources is allowed')
    bufsize = args.buffer_size
    batch_size = args.eval_batch_size
    threads = args.num_workers

    def fn(*xs):
        tensors = [torch.as_tensor(text_to_indices(x, voc)) for x, voc in zip(xs, vocabs)]
        return {
            'src': tensors,
            'n_tok': tensors[0].size(0)
        }

    src = [TextLine(f, bufsize=bufsize, num_threads=6) for f in dev]
    src = Zip(src, bufsize=bufsize, num_threads=6).select(fn)

    refs = [TextLine(ref, bufsize=args.buffer_size, num_threads=threads)
            for ref in refs]
    refs = Zip(refs, bufsize=args.buffer_size, num_threads=threads)

    ds = Zip([src, refs], bufsize=args.buffer_size, num_threads=threads).select(
        lambda x, ys: {
            'src': x['src'],
            'n_tok': x['n_tok'],
            'refs': [y.split() for y in ys]  # tokenize references
        }
    )

    # for x in ds:
    #     print(x)
    #     exit(0)

    def collate_fn(xs):
        inputs = aggregate_value_by_key(xs, 'src')
        inputs = list(zip(*inputs))
        inputs = [cuda(pack_tensors(input, voc.pad_id)) for input, voc in zip(inputs, vocabs)]
        return {
            'src': inputs[0] if len(inputs) == 1 else inputs,
            'n_tok': aggregate_value_by_key(xs, 'n_tok', sum),
            'n_snt': inputs[0].size(0),
            'refs': aggregate_value_by_key(xs, 'refs')

        }

    iterator = Iterator(
        ds, batch_size,
        cache_size=batch_size,
        collate_fn=collate_fn,
        sort_cache_by=lambda sample: -sample['n_tok'],
    )

    return iterator


def get_log(args, model, sample):
    # sample: {'net_input':[x, y_l2r,y_r2l],'target_l2r':l2r,'target_r2l':r2l,'ntok_l2r':n1,'ntok_r2l':n2}
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        loss_r2l, loss_l2r, nll_loss = nn.parallel.data_parallel(model, sample['net_input'])
        loss_l2r = loss_l2r.sum()
        loss_r2l = loss_r2l.sum()
        nll_loss = nll_loss.sum()
    else:
        loss_r2l, loss_l2r, nll_loss = model(*sample['net_input'])

    sample_size_r2l = sample['target_r2l'].size(0) if args.batch_by_sentence else sample['ntok_r2l']
    sample_size_l2r = sample['target_l2r'].size(0) if args.batch_by_sentence else sample['ntok_l2r']
    log = {
        'loss_l2r': loss_l2r.data.item(),
        'loss_r2l': loss_r2l.data.item(),
        'nll_loss': nll_loss.data.item(),
        'ntok_src': sample['ntok_src'],
        'ntok_l2r': sample['ntok_l2r'],
        'ntok_r2l': sample['ntok_r2l'],
        'nsentences': sample['target_l2r'].size(0),
        'sample_size_l2r': sample_size_l2r,
        'sample_size_r2l': sample_size_r2l,
    }
    return loss_r2l, loss_l2r, sample_size_r2l, sample_size_l2r, nll_loss, log


def agg_logs(logs):
    loss_l2r = sum(log.get('loss_l2r') for log in logs)
    loss_r2l = sum(log.get('loss_r2l') for log in logs)
    nll_loss = sum(log.get('nll_loss') for log in logs)
    ntok_src = sum(log.get('ntok_src') for log in logs)
    ntok_l2r = sum(log.get('ntok_l2r') for log in logs)
    ntok_r2l = sum(log.get('ntok_r2l') for log in logs)
    nsent = sum(log.get('nsentences') for log in logs)
    sample_size_l2r = sum(log.get('sample_size_l2r') for log in logs)
    sample_size_r2l = sum(log.get('sample_size_r2l') for log in logs)

    return {
        'loss_l2r': loss_l2r,
        'loss_r2l': loss_r2l,
        'nll_loss': nll_loss,
        'per_word_loss': nll_loss / ntok_l2r,
        'ntok_src': ntok_src,
        'ntok_l2r': ntok_l2r,
        'ntok_r2l': ntok_r2l,
        'nsentences': nsent,
        'sample_size_l2r': sample_size_l2r,
        'sample_size_r2l': sample_size_r2l,
    }


def main(args):
    set_seed(args.seed)

    # load vocabularies
    vocabularies = state_dict.get('vocabularies')

    if not vocabularies:
        if not args.vocab_size:
            args.vocab_size = [None]
        if len(args.vocab_size) == 1:
            args.vocab_size *= len(args.vocab)
        assert len(args.vocab_size) == len(args.vocab)
        vocabularies = [Vocabulary(filename, size) for filename, size in
                        zip(args.vocab, args.vocab_size)]

    source_vocab: Vocabulary = vocabularies[0]
    target_vocab: Vocabulary = vocabularies[1]

    # build model and criterion
    stop_watcher = StopwatchMeter(state_less=True)

    # 1. Build model
    model = models.build_model(args, vocabularies)
    if args.pretrain:
        logger.info('Loading pretraining parameters ...')
        pretrain = Loader.load_state(args.pretrain, 'cpu')
        pretrain = pretrain['model']
        from thseq.utils.misc import load_pretrain
        loaded, not_loaded = load_pretrain(model.encoder, pretrain, 'encoder')
        logger.info(f'Encoder loaded: {" ".join(loaded)}\n', )
        logger.info(f'Encoder not loaded: {" ".join(not_loaded)}\n', )
        loaded, not_loaded = load_pretrain(model.decoder.r2l, pretrain, 'decoder')
        logger.info(f'Decoder loaded: {" ".join(loaded)}\n', )
        logger.info(f'Decoder not loaded: {" ".join(not_loaded)}\n', )

    # dummy_input = (torch.zeros(100, 10).long(), torch.zeros(80, 10).long())
    # with SummaryWriter(log_dir=log_dir) as writer:
    #     writer.add_graph(model,dummy_input)
    #     del dummy_input
    # import sys
    # sys.exit(0)

    # Initialize parameters
    if not resume:
        logger.info(f'Model: \n{model}')
        model.apply(init_parameters)

        stat_parameters(model)
        logger.info(f'Batch size = {args.batch_size[0] * torch.cuda.device_count()} '
                    f'({args.batch_size[0]} x {torch.cuda.device_count()})')

    model = cuda(model)

    optimizer = optim.build_optimizer(args, model.parameters())
    lr_scheduler = thseq.optim.lr_scheduler.build_lr_scheduler(args, optimizer)

    # build trainer
    trainer = Trainer(args, model, optimizer, None, lr_scheduler)

    # build data iterator
    iterator = get_train_iterator(args, source_vocab, target_vocab)

    # Group stateful instances as a checkpoint
    state = State(args.save_checkpoint_secs, args.save_checkpoint_steps,
                  args.keep_checkpoint_max, args.keep_best_checkpoint_max,
                  args=args, trainer=trainer, model=model, criterion=None, optimizer=optimizer,
                  lr_scheduler=lr_scheduler, iterator=iterator,
                  vocabularies=vocabularies)

    # Restore state
    state.load_state_dict(state_dict)

    # Train until the learning rate gets too small
    import math
    max_epoch = args.max_epoch or math.inf
    max_step = args.max_step or math.inf

    eval_iter = get_dev_iterator(args, [source_vocab, target_vocab])

    reseed = lambda: set_seed(args.seed + state.step)

    kwargs = {}
    if resume:
        kwargs = {'purge_step': state.step}
    reseed()

    def before_epoch_callback():
        # 0-based
        logger.info(f'Start epoch {state.epoch + 1}')

    def after_epoch_callback():
        step0, step1 = state.step_in_epoch, iterator.step_in_epoch
        total0, total1 = state.step, iterator.step
        logger.info(f'Finished epoch {state.epoch + 1}. '
                    f'Failed steps: {step1 - step0} out of {step1} in last epoch and '
                    f'{total1 - total0} out of {total1} in total. ')

        state.increase_epoch()
        if state.eval_scores:
            eval_score = -state.eval_scores[-1]
            trainer.lr_step(state.epoch, -eval_score)

    trainer.reset_meters()

    with SummaryWriter(log_dir=os.path.join(args.model, 'tensorboard'), **kwargs) as writer:
        batches = []
        for batch in iterator.while_true(
                predicate=(lambda: (args.min_lr is None or trainer.get_lr() > args.min_lr)
                                   and state.epoch < max_epoch
                                   and state.step < max_step),
                before_epoch=before_epoch_callback,
                after_epoch=after_epoch_callback
        ):
            model.train()
            reseed()

            batches.append(batch)
            if len(batches) % args.accumulate == 0:
                samples = []
                for batch in batches:
                    input = batch.data['src']
                    r2l = batch.data['r2l']
                    l2r = batch.data['l2r']

                    ntok_r2l = batch.data['ntok_r2l']
                    ntok_l2r = batch.data['ntok_l2r']
                    ntok_src = batch.data['ntok_src']

                    sample = {
                        'net_input': (input, r2l, l2r),
                        'target_r2l': r2l,
                        'target_l2r': l2r,
                        'ntok_src': ntok_src,
                        'ntok_r2l': ntok_r2l,
                        'ntok_l2r': ntok_l2r,
                    }
                    samples.append(sample)
                batches.clear()
                log = trainer.train_step(samples)
                if not log:
                    continue
            else:
                continue

            state.increase_num_steps()
            trainer.lr_step_update(state.step)
            pwc = log["per_word_loss"]  # natural logarithm
            total_steps = state.step

            wps = trainer.meters["wps"].avg
            gnorm = trainer.meters['gnorm'].val
            cur_lr = trainer.get_lr()
            info = f'{total_steps} ' \
                f'|loss={log["loss"]:.4f} ' \
                f'|pwc={pwc:.4f} ' \
                f'|lr={cur_lr:.6e} ' \
                f'|norm={gnorm:.2f} ' \
                f'|wps={wps:.2f} ' \
                f'|input={(log.get("ntok_src", 0), log.get("ntok_l2r", 0), log.get("ntok_r2l", 0))} '
            logger.info(info)
            # torch.cuda.empty_cache()

            writer.add_scalar('loss', log['loss_r2l'] + log['loss_l2r'], total_steps)
            writer.add_scalar('lr', cur_lr, total_steps)

            if total_steps % args.eval_steps == 0:
                stop_watcher.start()
                with torch.no_grad():
                    val_score = trainer.evaluate(eval_iter, r2l=args.r2l)
                stop_watcher.stop()
                state.add_valid_score(val_score)
                writer.add_scalar(f'dev/bleu', val_score, total_steps)
                logger.info(
                    f'Validation bleu at {total_steps}: {val_score:.2f}, '
                    f'took {timedelta(seconds=stop_watcher.sum // 1)}')

            state.try_save()

        # Evaluate at the end of training.
        stop_watcher.start()
        with torch.no_grad():
            val_score = trainer.evaluate(eval_iter, r2l=args.r2l)
        stop_watcher.stop()
        state.add_valid_score(val_score)
        writer.add_scalar(f'dev/bleu', val_score, state.step)
        logger.info(f'Validation bleu at {state.step}: {val_score:.2f}, '
                    f'took {timedelta(seconds=stop_watcher.sum // 1)}')
    logger.info(f'Training finished at {strftime("%b %d, %Y, %H:%M:%S", localtime())}, '
                f'took {timedelta(seconds=state.elapsed_time // 1)}')
    logger.info(f'Best validation bleu: {max(state.eval_scores)}, at {state.get_best_time()}')


def prepare():
    input_args = '--train toys/reverse/train.src toys/reverse/train.trg ' \
                 '--dev toys/reverse/dev.src toys/reverse/dev.trg ' \
                 '--vocab toys/reverse/vocab.src toys/reverse/vocab.trg ' \
                 '--model runs/test ' \
                 '--hidden-size 32 ' \
                 '--max-epoch 2 ' \
                 '--eval-steps 1 ' \
                 '--shuffle -1 ' \
                 '--eval-batch-size 1 ' \
                 '--save-checkpoint-steps 1 ' \
                 '--arch multi-head-rnn ' \
                 '--lr 0.001 '.split()
    input_args = None
    parser = options.get_training_parser()
    # 1. Parse static command-line args and get default args
    cli_args, default_args, unknown_args = options.parse_static_args(parser, input_args=input_args)

    # 2. Load config args
    config_args = None
    if cli_args.config:
        args_list = []
        for config_file in cli_args.config:
            with open(config_file) as r:
                args_list.append(json.loads(r.read()))
        args_list = [argparse.Namespace(**item) for item in args_list]
        config_args = args_list[0]

        for args_ in args_list[1:]:
            config_args = override(config_args, args_, False)
    # 3. Load model args
    if cli_args.scratch and os.path.exists(cli_args.model):
        shutil.rmtree(cli_args.model)

    try:
        ckp_path = cli_args.model
        if os.path.isdir(cli_args.model):
            ckp_path = Loader.get_latest(cli_args.model)[1]
        state_dict = Loader.load_state(ckp_path)

    except FileNotFoundError:
        state_dict = {}
    resume = len(state_dict) > 0

    model_args = state_dict.get('args')
    # 4. Override by priorities.
    # cli_args > config_args > model_args > default_args
    args = override(config_args, cli_args, False)
    args = override(model_args, args, False)
    args = override(default_args, args, False)
    # 5. Parse a second time to get complete cli args
    cli_args, default_args = options.parse_dynamic_args(parser, input_args=input_args, parsed_args=args)
    # 6. Retain valid keys of args
    valid_keys = set(default_args.__dict__.keys())
    # 7. Override again
    args = override(args, cli_args, False)
    args = override(default_args, args, False)
    # 8. Remove invalid keys
    stripped_args = argparse.Namespace()
    for k in valid_keys:
        setattr(stripped_args, k, getattr(args, k))

    config_name = os.path.join(args.model, 'config.json')

    if not os.path.exists(args.model):
        os.makedirs(args.model)

    if model_args != args or not os.path.exists(config_name):
        with open(config_name, 'w') as w:
            w.write(json.dumps(args.__dict__, indent=4, sort_keys=True))

    if len(args.train) == 1:
        assert args.langs and len(args.langs) == 2, args.langs

        prefix = args.train[0]
        args.train = [f'{prefix}.{args.langs[0]}', f'{prefix}.{args.langs[1]}']

    if len(args.dev) == 1:
        assert args.langs and len(args.langs) == 2, args.langs

        prefix = args.dev[0]
        args.dev = [f'{prefix}.{args.langs[0]}', f'{prefix}.{args.langs[1]}']

    if len(args.vocab) == 1:
        assert args.langs and len(args.langs) == 2, args.langs

        prefix = args.vocab[0]
        args.vocab = [f'{prefix}.{args.langs[0]}', f'{prefix}.{args.langs[1]}']

    return args, state_dict, resume


if __name__ == '__main__':
    args, state_dict, resume = prepare()

    # redirect stdout and stderr to log file
    # redirection = open(log_name, 'a', buffering=1)
    # sys.stdout = redirection
    # sys.stderr = redirection

    stdout_handler = StreamHandler(sys.stdout, bubble=True)
    stderr_handler = StderrHandler(level=WARNING)
    # write logs to log.MODEL file
    # file_handler = FileHandler(log_name, bubble=True)
    # file_handler.format_string = '{record.message},{record.extra[cwd]}'
    # file_handler.format_string = '[{record.time:%Y-%m-%d %H:%M:%S.%f%z}] {record.level_name}: {record.message}'
    # with file_handler.applicationbound():
    stdout_handler.format_string = '[{record.time:%Y-%m-%d %H:%M:%S.%f%z}] ' \
                                   '{record.level_name}: {record.message}'
    with stdout_handler.applicationbound():
        if resume:
            logger.info(f'Resume training from checkpoint: {Loader.get_latest(args.model)[1]}')

        try:
            main(args)
        except Exception as e:
            logger.error(f'\n{traceback.format_exc()}')
