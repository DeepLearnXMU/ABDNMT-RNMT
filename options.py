import argparse
from copy import deepcopy

import torch
from logbook import Logger

from thseq.criterions import CRITERION_REGISTRY
from thseq.models import MODEL_REGISTRY
from thseq.optim import OPTIMIZER_REGISTRY
from thseq.optim.lr_scheduler import LR_SCHEDULER_REGISTRY
from thseq.utils.misc import get_value_or_default

DEFAULT_ARGS_REGISTRY = {}
logger = Logger()


def get_training_parser():
    parser = get_parser('Trainer')
    add_distributed_args(parser)
    add_model_args(parser)
    add_dataset_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
    add_evaluation_args(parser)

    return parser


def get_generation_parser():
    parser = get_parser('Generation')
    add_generation_parser(parser)
    return parser


def get_parser(desc):
    parser = argparse.ArgumentParser(desc)
    parser.add_argument('--log-interval', type=int, metavar='N',
                        help='log progress every N batches (when progress bar is disabled)')
    # seed 9527 is BAAAAD!
    parser.add_argument('--seed', type=int, default=9527, metavar='N',
                        help='pseudo random number generator seed. (default is 9527)')
    parser.add_argument('--fp16', action='store_true', default=None, help='use FP16')
    parser.add_argument('--gmail', type=str,
                        help='send message via gmail, formatted as from@gmail.com,password,to@gmail.com, '
                             'this option will be encrypted using ~/.ssh/id_rsa before saving to config '
                             'file and decrypted using ~/.ssh/id_rsa.pub')

    return parser


def add_dataset_args(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument('--train', nargs='+', help='full paths or prefix of the training set.')
    group.add_argument('--dev', nargs='+', help='full paths or prefix of the development set. '
                                                'When there are more than 2 arguments given, '
                                                'all the rest except the first one are references.')
    group.add_argument('--vocab', nargs='+', help='full paths or prefix of the vocabulary files.')
    group.add_argument('--langs', nargs=2, help='language suffixes')
    group.add_argument('--vocab-size', nargs='+', type=int,
                       help='limit the vocabulary size. If only one value is provided, '
                            'it will be reused for all vocabularies. (default is [30000])', default=[30000])
    group.add_argument('--buffer-size', type=int, help='pre-process a bulk of samples. '
                                                       '(default is 100000)', default=100000)
    group.add_argument('--sort-buffer-factor', metavar='FACTOR', type=int,
                       help='sort samples for training efficiency. The effective size of the sorting buffer '
                            'will be (BATCH_SIZE * FACTOR). A 0 value disables buffer sorting. (default is 20).',
                       default=32)
    group.add_argument('--num-workers', type=int, help='process the buffer in parallel. (default is 6)', default=6)
    group.add_argument('--shuffle', type=int, metavar='N',
                       help='shuffle the data. If N == 0, then shuffling is disabled.'
                            'If N > 0, shuffles a buffer of size N.'
                            'If N == -1, shuffles all the data. '
                            'Note that this will significantly increase the memory footprint '
                            'by loading all the data at a time. (default is 100000)', default=100000)
    group.add_argument('--length-limit', type=int, nargs='+',
                       help='limit the lengths of training samples. (default is [1000])',
                       default=[1000])
    group.add_argument('--batch-size', type=int, metavar='BATCH_SIZE',
                       help='a tuple specifying batch size and optionally padded batch size. (default is (80,))',
                       nargs='+',
                       default=(80,))
    group.add_argument('--batch-by-sentence', type=int, metavar='N',
                       help='whether batch the samples by number of sentences, '
                            'otherwise batch by number of tokens. '
                            '(default is 1, which means batching by sentences)',
                       default=1)


def add_evaluation_args(parser):
    group = parser.add_argument_group('Validation')
    group.add_argument('--eval-steps', type=int, help='(default is 5000)', default=5000)
    group.add_argument('--eval-batch-size', type=int, help='(default is 1)', default=32)
    group.add_argument('--r2l', type=int)
    group.add_argument('--patience', type=int)


def add_model_args(parser):
    group = parser.add_argument_group('Model configuration')

    # Model definitions can be found under thseq/models/
    group.add_argument(
        '--arch', '-a', metavar='ARCH',
        choices=list(MODEL_REGISTRY.keys()),
        help='model architecture: {}'.format(
            ', '.join(MODEL_REGISTRY.keys())),
        default='rnn',
    )

    # Criterion definitions can be found under thseq/criterions/
    group.add_argument(
        '--criterion', metavar='CRIT',
        choices=list(CRITERION_REGISTRY.keys()),
        help='training criterion: {}. (default is ce)'.format(
            ', '.join(CRITERION_REGISTRY.keys())),
        default='ce'
    )

    group.add_argument('--config', type=str, nargs='+',
                       help='support multiple config files, the relative priority '
                            'is defined by their orders from low to high.')

    return group


def add_optimization_args(parser):
    group = parser.add_argument_group('Optimization')
    group.add_argument('--pretrain', type=str)
    group.add_argument('--max-epoch', '--me', type=int, metavar='N',
                       help='force stop training at specified epoch')
    group.add_argument('--max-step', '--mu', type=int, metavar='N',
                       help='force stop training at specified update. (default is 1000000)', default=2000000)
    group.add_argument('--clip-norm', type=float, metavar='NORM',
                       help='clip threshold of gradients. (default is 1.0).', default=1.0)
    group.add_argument('--sentence-avg', type=int,
                       help='normalize gradients by the number of sentences in a batch'
                            ' (default is to normalize by number of sentences)', default=1)
    group.add_argument('--accumulate', type=int, metavar='N',
                       help='accumulate N batches for one parameters update. (default is 1)', default=1)

    # Optimizer definitions can be found under fairseq/optim/
    group.add_argument('--optimizer', metavar='OPT',
                       choices=list(OPTIMIZER_REGISTRY.keys()),
                       help='optimizer: {}. (default is adam)'.format(', '.join(OPTIMIZER_REGISTRY.keys())),
                       default='adam')
    group.add_argument('--lr', '--learning-rate', type=float, nargs='+', metavar='LR_1 LR_2 ... LR_N',
                       help='learning rate for the first N epochs; all epochs >N using LR_N'
                            ' (note: this may be interpreted differently depending on --lr-scheduler)', )
    group.add_argument('--momentum', type=float, metavar='M',
                       help='momentum factor')
    group.add_argument('--weight-decay', '--wd', type=float, metavar='WD',
                       help='weight decay. (default is 0)',
                       default=0)

    # Learning rate schedulers can be found under fairseq/optim/lr_scheduler/
    group.add_argument('--lr-scheduler',
                       help='learning rate scheduler: {}. (default is fixed)'.format(
                           ', '.join(LR_SCHEDULER_REGISTRY.keys())),
                       default='fixed')
    group.add_argument('--lr-shrink', type=float, metavar='LS',
                       help='learning rate shrink factor for annealing, lr_new = (lr * lr_shrink). (default is 0.5)',
                       default=0.5)
    group.add_argument('--min-lr', type=float, metavar='LR',
                       help='minimum learning rate. (default is None)')
    group.add_argument('--min-lr-bound', type=int, help='set lr as min_lr when scheduler attempts '
                                                        'to set a value lower than min_lr.')
    group.add_argument('--min-loss-scale', type=float, metavar='D',
                       help='minimum loss scale (for FP16 training)')

    return group


def add_checkpoint_args(parser):
    group = parser.add_argument_group('Checkpoint')
    group.add_argument('--model', required=True,
                       help='save model to this directory. when the directory exists, '
                            'training will resume from the latest checkpoint.')
    group.add_argument('--initialize', metavar='INIT', type=str, help='path to checkpoint file. '
                                                                      'model parameters will be initialized from INIT.')
    group.add_argument('--save-checkpoint-secs', metavar='N', type=int,
                       help='save checkpoint every N seconds')
    group.add_argument('--save-checkpoint-steps', metavar='N', type=int,
                       help='save checkpoint every N steps. when both '
                            '--save-checkpoint-sec and --save-checkpoint-step are given, '
                            'then only this option will be effective. (default is 2000)', default=2000)
    group.add_argument('--keep-checkpoint-max', metavar='N', type=int,
                       help='the maximum number of checkpoint files to save. (default saves recent 10)', default=1)

    group.add_argument('--keep-best-checkpoint-max', metavar='N', type=int,
                       help='the maximum number of best checkpoints to save, which are evaluated on the dev set. '
                            '(default saves best 10)', default=1)
    group.add_argument('--scratch', action='store_true',
                       help='training from scratch regardless of whether the model directory'
                            'exists.')


def add_generation_parser(parser):
    group = parser.add_argument_group('Generation')
    group.add_argument('--models', required=True,
                       nargs='+',
                       help='either the model directories or specific checkpoint files.')
    group.add_argument('--select', metavar='SELECT', default='best',
                       choices=['best', 'latest', 'best-window', 'latest-window'],
                       help='select checkpoints')
    group.add_argument('-n', metavar='N', type=int, default=1,
                       help='select N checkpoints. '
                            'This is used as window size when SELECT is best-window. '
                            '(default is 1)')
    group.add_argument('-k', type=int, help='beam width. (default is 10)', default=10)
    group.add_argument('--interactive', '-i', action='store_true', help='enable interactive translation')
    group.add_argument('--batch-size', '-b', type=int, default=1)
    group.add_argument('--max-length', type=int)
    group.add_argument('--length-normalization-factor', type=float, default=1)
    group.add_argument('--length-normalization-const', type=float, default=0)
    group.add_argument('--retain-attn', type=int)
    group.add_argument('--bpe', type=int, help='bpe post-processing. (default is 1)', default=1)
    group.add_argument('--r2l', type=int, help='r2l reverse. (default is 0)', default=0)
    group.add_argument('--input', type=str,nargs='+', help='input file or files.')
    group.add_argument('--verbose', '-v',action='store_true')



def add_distributed_args(parser):
    group = parser.add_argument_group('distributed')

    group.add_argument("--dist-world-size", type=int, default=1)
    group.add_argument("--dist-local-devices", type=int, nargs='+', default=list(range(torch.cuda.device_count())))
    group.add_argument("--dist-start-rank", type=int, default=0)
    group.add_argument("--dist-backend", type=str, default='nccl')
    group.add_argument("--dist-init-method", type=str, default='tcp://localhost:9527')


def parse_static_args(parser, input_args=None):
    """ Parse the args a first time.
    """
    no_default_parser = get_no_default_parser(parser)
    args, unknown_args = no_default_parser.parse_known_args(input_args)
    defaults = get_defaults(parser)
    return args, defaults, unknown_args


def parse_dynamic_args(parser, input_args=None, parsed_args=None):
    """ Parse dynamic args.
    """
    if parsed_args:
        args = parsed_args
    else:
        args, _ = parser.parse_known_args(input_args)

    # Add model-specific args to parser.
    if get_value_or_default(args, 'arch') is not None:
        model_specific_group = parser.add_argument_group(
            'Model-specific configuration',
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values.
            argument_default=argparse.SUPPRESS,
        )
        MODEL_REGISTRY[args.arch].add_args(model_specific_group)

    # Add dynamic args to parser.
    if get_value_or_default(args, 'criterion') is not None:
        CRITERION_REGISTRY[args.criterion].add_args(parser)
    if get_value_or_default(args, 'optimizer') is not None:
        OPTIMIZER_REGISTRY[args.optimizer].add_args(parser)
    if get_value_or_default(args, 'lr_scheduler') is not None:
        LR_SCHEDULER_REGISTRY[args.lr_scheduler].add_args(parser)

    # Parse a second time.
    no_default_parser = get_no_default_parser(parser)

    args = no_default_parser.parse_args(input_args)

    default_args = get_defaults(parser)

    return args, default_args


def get_no_default_parser(parser):
    no_default_parser = deepcopy(parser)
    clear_defaults(no_default_parser)
    return no_default_parser


def get_defaults(parser: argparse.ArgumentParser, exclude=None) -> argparse.Namespace:
    if exclude is None:
        exclude = []
    exclude += ['help']
    exclude = set(exclude)

    args = argparse.Namespace()
    for action in parser._actions:
        if action.dest not in exclude:
            args.__dict__[action.dest] = action.default
    return args


def clear_defaults(parser: argparse.ArgumentParser, exclude=None) -> None:
    if exclude is None:
        exclude = []
    exclude += ['help']
    exclude = set(exclude)

    for action in parser._actions:
        if action.dest not in exclude:
            action.default = None
