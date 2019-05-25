import os
import sys
from datetime import timedelta
from typing import List

import torch
from lunas import Stdin, Iterator, TextLine, Zip

import options
from state import Loader
from thseq.data.plain import text_to_indices, restore_bpe
from thseq.data.vocabulary import Vocabulary
from thseq.models import ensemble, build_model
from thseq.utils.meters import TimeMeter
from thseq.utils.misc import aggregate_value_by_key
from thseq.utils.tensor import pack_tensors, cuda


def loads(models, select='best', n=1):
    paths = []
    for model in models:
        if os.path.isdir(model):
            if select == 'best':
                ckp2paths = Loader.get_bests(model, n)
            elif select == 'latest':
                ckp2paths = Loader.get_latests(model, n)
            elif select == 'best-window':
                ckp2paths = Loader.get_best_window(model, n)
            elif select == 'latest-window':
                ckp2paths = Loader.get_latest_window_around_best(model, n)
            else:
                raise NotImplementedError(f'select method ({select}) is not implemented.')
            paths.extend([path for _, path in ckp2paths])
        else:
            paths.append(model)
    states = Loader.load_states(paths, 'cpu')

    def build(state):
        vocabularies = state['vocabularies']
        model = build_model(state['args'], vocabularies)
        model.load_state_dict(state['model'])
        return model

    models = [build(state) for state in states]
    return states, models, paths


def translate_batch(model, batch, bpe=True, r2l=False):
    outputs = model.translate(batch.data['src'], args.k, args.max_length,
                              args.length_normalization_factor,
                              args.length_normalization_const,
                              args.retain_attn)

    if r2l:
        outputs = list(map(lambda x: x[::-1], outputs))
    if bpe:
        outputs = list(map(restore_bpe, outputs))

    outputs = batch.revert(outputs)

    for tokens in outputs:
        sys.stdout.write('{}\n'.format(' '.join(tokens)))
        sys.stdout.flush()


def get_eval_iterator(args, vocabs: List[Vocabulary]):
    batch_size = args.batch_size
    bufsize = 10000
    threads=1
    if args.interactive:
        batch_size = 1
        bufsize = 1

    def fn(*xs):
        tensors = [torch.as_tensor(text_to_indices(x, voc)) for x, voc in zip(xs, vocabs)]
        return {
            'src': tensors,
            'n_tok': tensors[0].size(0)
        }

    if not args.input:
        src = [Stdin(bufsize=bufsize, num_threads=threads, sentinel='')]
    else:
        src = [TextLine(f, bufsize=bufsize, num_threads=threads) for f in args.input]
    src = Zip(src, bufsize=bufsize, num_threads=threads).select(fn)

    def collate_fn(xs):
        inputs = aggregate_value_by_key(xs, 'src')
        inputs = list(zip(*inputs))
        inputs = [cuda(pack_tensors(input, voc.pad_id)) for input, voc in zip(inputs, vocabs)]
        return {
            'src': inputs[0] if len(inputs) == 1 else inputs,
            'n_tok': aggregate_value_by_key(xs, 'n_tok', sum),
            'n_snt': inputs[0].size(0)
        }

    iterator = Iterator(
        src, batch_size,
        cache_size=batch_size,
        collate_fn=collate_fn,
        sort_cache_by=lambda sample: -sample['n_tok'],
    )

    return iterator


def main(args):
    states, models, paths = loads(args.models, args.select, args.n)
    vocabularies = states[0].get('vocabularies')
    source_vocab, target_vocab = vocabularies

    if args.verbose:
        sys.stderr.write(f'Decoding using checkpoints: {paths} \n')

    if len(models) == 1:
        model = models[0]
    else:
        model = ensemble.AvgPrediction(models)

    cuda(model)

    meter = TimeMeter()
    vocabs=[source_vocab]
    if len(args.input)==2:
        vocabs.append(target_vocab)

    it = get_eval_iterator(args, vocabs)

    n_tok = 0
    n_snt = 0
    for batch in it.iter_epoch():
        translate_batch(model, batch, True, getattr(states[0]['args'], 'r2l', None))
        meter.update(batch.data['n_tok'])
        n_snt += batch.data['n_snt']
        n_tok += batch.data['n_tok']

    sys.stderr.write(f'Snt={n_snt}, Tok={n_tok}, '
                     f'Time={timedelta(seconds=meter.elapsed_time)}, '
                     f'Avg={meter.avg:.2f}tok/s\n')


if __name__ == '__main__':
    parser = options.get_generation_parser()
    args = parser.parse_args()

    main(args)
