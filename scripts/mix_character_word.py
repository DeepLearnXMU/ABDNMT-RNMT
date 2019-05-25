import argparse
import sys
from collections import Counter, defaultdict


def main(args):
    corpus = args.corpus
    limit = args.limit
    occur = args.occur

    if limit is None:
        limit = sys.maxsize
    if occur is None:
        occur = 0

    vocab = Counter()
    cvocab = Counter()
    with open(corpus) as r:
        for l in r:
            vocab.update(l.split())
            cvocab.update(''.join(l.split()))

    mix_vocab = defaultdict(int)
    for i, (c, n) in enumerate(cvocab.most_common()):
        if n < occur or i >= limit:
            break
        mix_vocab[c] = n


    for w, n in vocab.most_common():
        if n < occur or len(mix_vocab) >= limit:
            break
        if w not in mix_vocab:
            mix_vocab[w] = n
    
    with open(corpus) as r:
        for l in r:
            words = [w if w in mix_vocab else ' '.join(w) for w in l.split()]
            sys.stdout.write(f'{" ".join(words)}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, help='text format', required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--occur", type=int)
    args = parser.parse_args()
    main(args)
