import random

import numpy
import torch


class Vocabulary(object):
    def __init__(self, file_or_object, size: int = None, eos: str = '</s>', unk: str = '<unk>',
                 pad: str = "<blank>") -> None:
        super().__init__()

        self._size = size
        self._pad = pad
        self._eos = eos
        self._unk = unk

        self._special_symbols = [self._pad, self._eos, self._unk, ]
        self._special_symbol_ids = []

        word2index, index2word = self.load(file_or_object)

        self._eos_id: int = word2index[eos]
        self._unk_id: int = word2index[unk]
        self._pad_id: int = word2index[pad]

        self._word2index, self._index2word = word2index, index2word

        for sym in self._special_symbols:
            self._special_symbol_ids.append(self._word2index[sym])

    @property
    def eos(self):
        return self._eos

    @property
    def unk(self):
        return self._unk

    @property
    def pad(self):
        return self._pad

    @property
    def eos_id(self):
        return self._eos_id

    @property
    def unk_id(self):
        return self._unk_id

    @property
    def pad_id(self):
        return self._pad_id

    def load(self, file_or_object):
        limit = self._size
        special_symbols = [symbol for symbol in self._special_symbols]

        if isinstance(file_or_object, str):
            symbols = self._from_file(file_or_object)
        else:
            symbols = self._from_object(file_or_object)

        symbols = [symbol for symbol in symbols if symbol not in special_symbols]

        if limit is None:
            limit = len(symbols)

        symbols = symbols[:limit]

        word2index = dict()
        index2word = dict()
        for i, word in enumerate(special_symbols + symbols):
            word2index[word] = i
            index2word[i] = word

        return word2index, index2word

    def _from_object(self, obj):
        symbols = [symbol for symbol in obj]
        return symbols

    def _from_file(self, voc_file):
        symbols = []
        if voc_file is not None:
            with open(voc_file) as r:
                for l in r:
                    symbols.append(l.split()[0])
        return symbols

    def to_indices(self, seq, dtype='int64'):
        prepends = []
        appends = [self.eos_id]

        w2i = self._word2index

        ids = [w2i.get(word, self.unk_id) for word in seq]
        return numpy.array(prepends + ids + appends, dtype)

    def to_words(self, ids, discards=None):
        if discards is None:
            discards = [self.eos_id]
        if isinstance(discards, int):
            discards = [discards]

        if not isinstance(discards[0], int):
            raise ValueError('discards should be a list of token ids, got {}'.format(type(discards[0])))

        if isinstance(ids, torch.Tensor):
            ids = list(ids.numpy())

        seq = [self._index2word[id] for id in ids if id not in discards]
        return seq

    def sample(self, length, k):
        symbols = list(self._word2index.keys())
        symbols = symbols[max(self._special_symbol_ids) + 1:]
        return [random.choices(symbols, k=length) for _ in range(k)]

    @classmethod
    def random(cls, size):
        symbols = set()
        chars = [chr(ord('0') + i) for i in range(10)]
        chars += [chr(ord('a') + i) for i in range(26)]
        chars += [chr(ord('A') + i) for i in range(26)]
        max_len = 8
        for i in range(size + 10):
            n_ch = random.randint(1, max_len)
            symbols.add(''.join(random.choices(chars, k=n_ch)))

        while len(symbols) < size:
            n_ch = random.randint(1, max_len)
            symbols.add(''.join(random.choices(chars, k=n_ch)))

        return cls(symbols, size)

    def __len__(self):
        return len(self._word2index)
