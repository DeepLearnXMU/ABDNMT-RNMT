import argparse
import functools
import itertools
import sys
from collections import Counter
from multiprocessing import Pool


def count(lines):
    voc = Counter()
    for l in lines:
        voc.update(l.split())
    return voc


def group(lines, group):
    groups = []
    chunk, remain = divmod(len(lines), group)
    chunk2, remain2 = divmod(remain, group)
    chunk += chunk2
    idxs = [i * chunk for i in range(group)]
    idxs = idxs + [len(lines)]
    for i, j in itertools.zip_longest(idxs[:-1], idxs[1:]):
        groups.append(lines[i:j])
    return groups


def parallel(r, n, buffer_size):
    pool = Pool(n)

    voc = Counter()
    while True:
        buffer = list(itertools.islice(r, buffer_size))
        if buffer:
            groups = group(buffer, n)
            vocs = pool.map(count, groups)
            voc = functools.reduce(lambda voc1, voc2: voc2.update(voc1) or voc2, [voc] + vocs)
        if len(buffer) < buffer_size:
            break

    return voc


def main(args):
    corpus = args.corpus
    limit = args.limit
    occur = args.occur

    sys.stderr.write("corpus: %s, limit: %r, occur: %r\n" % (corpus or 'stdin', limit, occur))

    if limit is None:
        limit = sys.maxsize
    if occur is None:
        occur = 0

    vocab = Counter()
    if corpus:
        r = open(corpus)
    else:
        r = sys.stdin
    if args.parallel and args.parallel > 1:
        vocab = parallel(r, args.parallel, args.buffer)
    else:
        for l in r:
            vocab.update(l.split())

    if corpus:
        r.close()

    num_tok = sum(vocab.values())
    num_in_vocab = 0

    word2cnt = vocab.most_common()
    for i, (word, cnt) in enumerate(word2cnt):
        if cnt < occur or i >= limit:
            break
        sys.stdout.write("%s %d\n" % (word, cnt))
        num_in_vocab += cnt
    coverage = num_in_vocab / num_tok
    sys.stderr.write(f"coverage: {coverage}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, help='text format')
    parser.add_argument("--limit", type=int)
    parser.add_argument("--occur", type=int)
    parser.add_argument("--parallel", type=int)
    parser.add_argument("--buffer", type=int, default=200000)
    args = parser.parse_args()
    main(args)
