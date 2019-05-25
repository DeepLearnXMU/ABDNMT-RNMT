import argparse
import functools

from state import *


def average_parameters(states):
    models = [s['model'] for s in states]
    for k, v in models[0].items():
        vs = [model['k'] for model in models[1:]] + [v]
        v = functools.reduce(torch.add, vs)
        models[0][k] = v
    return states[0]


def main(args):
    select, n = args.select, args.n
    paths = []
    for model in args.models:
        if os.path.isdir(model):
            if select == 'best':
                ckps = get_bests(model, n)
            elif select == 'latest':
                ckps = get_latests(model, n)
            elif select == 'best_window':
                ckps = get_best_window(model, n)
            else:
                raise NotImplementedError(f'select method ({select}) is not implemented.')
            paths.extend(get_paths(model, ckps))
        else:
            paths.append(model)
    if len(paths) < 2:
        raise ValueError(f'Not enough checkpoints. Got: {paths}')
    states = load_states(paths, 'cpu')
    state = average_parameters(states)
    torch.save(state, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+',
                        help='model directory or specific checkpoint files.')
    parser.add_argument('--output', type=str)
    parser.add_argument('--select', metavar='SELECT', default='best',
                        choices=['best', 'latest', 'best-window'],
                        help='select checkpoints')
    parser.add_argument('-n', metavar='N', type=int, default=1)
    args = parser.parse_args()
    main(args)
