import argparse
import os


def main(args):
    vocab = args.vocab
    corpus = args.corpus

    with open(vocab) as r:
        vocab = {}
        for l in r:
            w, c = l.split()
            vocab[w] = c

    for f in corpus:
        with open(f) as r:
            m, n = 0, 0
            for l in r:
                ws = l.split()
                n += len(ws)
                m += len([w for w in ws if w in vocab])
        print(f'{os.path.basename(f)}: {m}, {n}, {m / n}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vocab', type=str)
    parser.add_argument('corpus', type=str, nargs='+')
    args = parser.parse_args()
    main(args)
