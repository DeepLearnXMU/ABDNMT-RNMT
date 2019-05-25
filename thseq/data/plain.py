import re

import numpy

from thseq.data.vocabulary import Vocabulary
from thseq.utils.tensor import cuda

__all__ = ['data_length', 'convert_data', 'restore_bpe']

BPE_PATTERN = re.compile('(@@ )|(@@ ?$)')


def restore_bpe(line):
    tokenized = isinstance(line, list)
    if tokenized:
        line = ' '.join(line)
    line = BPE_PATTERN.sub('', line)
    if tokenized:
        line = line.split()
    return line


def data_length(line):
    return len(line.split())


def tokenize(data):
    return data.split()


def convert_to_array(data, padding_value):
    batch = len(data)
    data_len = list(map(len, data))
    max_len = max(data_len)

    seq = numpy.ones((batch, max_len)) * padding_value
    seq = numpy.array(seq, numpy.int64)

    for idx, item in enumerate(data):
        seq[:data_len[idx], idx] = item

    return seq


def text_to_indices(line, vocab: Vocabulary):
    words = line.split()
    return vocab.to_indices(words)


def pack_arrays(arrays, padding_value, dtype=None):
    dtype = dtype or arrays[0].dtype
    lens = [a.shape[0] for a in arrays]
    max_len = max(lens)
    a = numpy.zeros((len(arrays), max_len)) + padding_value
    a = numpy.array(a, dtype=dtype)

    mask = numpy.arange(max_len) < numpy.array(lens)[:, None]
    a[mask] = numpy.concatenate(arrays)
    return a


def convert_data(data, voc, tokenizer=True, try_cuda=True):
    if tokenizer:
        data = [voc.to_indices(tokenize(seq)) for seq in data]
    else:
        data = [voc.to_indices(seq) for seq in data]
    seq = convert_to_array(data, voc.pad_id)
    if try_cuda:
        seq = cuda(seq)
    return seq
