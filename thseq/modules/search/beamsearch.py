import itertools
from typing import List, Callable

import numpy
import torch

from thseq.modules.search.path import Path
from thseq.modules.search.stop_criteria import StopCriterion, get_stop_criteria
from thseq.utils.nested import select
from thseq.utils.tensor import cuda


class Beam(object):
    def __init__(self, width: int, candidates: List[List[Path]]):
        B, K = len(candidates), len(candidates[0])
        assert B > 0
        self.width: int = width
        self.alive: List[List[Path]] = candidates  # B x K, B might dynamically decreased
        # index corresponding to the source position in the batch
        self._alive_source: List[int] = list(range(B))
        self.ended: List[List[Path]] = [[] for _ in range(B)]  # B x ?, won't change
        self._B = B

    def forward(self, log_prob: torch.Tensor, is_path_ended: Callable[[Path], bool],
                stop_criteria: Callable[[List[Path], List[Path], int], bool]):
        # B: batch size
        # K: beam size
        # V: vocab size
        B, K_, V = log_prob.size()

        assert K_ <= V
        assert self.effective_batch_size() == B
        assert K_ == 1 or self.width == K_

        K = self.width

        prev_scores, mask = self._collect_scores()
        prev_scores = log_prob.new_tensor(prev_scores)  # B x K
        mask = log_prob.new_tensor(mask).byte()

        new_scores = prev_scores.unsqueeze(-1) + log_prob  # B x K x V
        new_scores.masked_fill_(mask.unsqueeze(-1), -float('inf'))  # mask ended paths

        new_scores = new_scores.reshape(B, -1)  # B x KV
        # B x 2K
        top_score, top_idx = new_scores.topk(2 * K, -1)
        top_beam = top_idx // V  # values range from 0 -> K
        top_tok = top_idx % V  # values range from 0 -> V

        node_scores = log_prob.view(B, -1).gather(1, top_idx)  # B x 2K
        for i in range(B):
            # update candidates
            paths = []
            for k, v, node_score, score in itertools.zip_longest(top_beam[i],
                                                                 top_tok[i],
                                                                 node_scores[i],
                                                                 top_score[i]):
                p = self.alive[i][k].extend(v.item(), node_score.item(), score.item(), k.item())
                paths.append(p)

            self.alive[i] = paths[:K]  # keep a constant alive size of K
            self.ended[self.source_index(i)] += [p for p in paths
                                                 if is_path_ended(p)]

        # when alive is stripped, the returned value will record
        # the index of each item before they were stripped.
        prev_indices = self._update_alive(stop_criteria)

        if not self.empty():
            # B' x K
            # use j * K instead of i * K, since alive buffer might be changed,
            # we should use it's source(original) position to index decoding state.
            beam_idxs = [[path.get_latest_source(default=0) + prev_i * K
                          for path in alive]
                         for alive, prev_i in zip(self.alive, prev_indices)]
            return beam_idxs
        else:
            return None

    def _collect_scores(self):
        scores = [[path.score for path in paths] for paths in self.alive]
        mask = [[path.ended for path in paths] for paths in self.alive]
        return scores, mask

    def _update_alive(self, stop_criteria):
        stops = [i for i, j in enumerate(self.source_indices())
                 if stop_criteria(self.alive[i], self.ended[j], j)]
        prev_index_in_batch = list(range(self.effective_batch_size()))
        for i in stops[::-1]:
            self.alive.pop(i)
            self._alive_source.pop(i)
            prev_index_in_batch.pop(i)
        self._error_check()

        return prev_index_in_batch

    def _error_check(self):
        alive_idxs = list(self.source_indices())
        for i in range(self._B):
            if i not in alive_idxs:  # ended
                ended = self.ended[i]
                if len(ended) == 0:
                    raise RuntimeError('alive is cleared, but ended is still empty. '
                                       'Please check correctness of stop_criteria.')

    def empty(self):
        return self.effective_batch_size() == 0

    def effective_batch_size(self):
        return len(self.alive)

    def source_index(self, index_in_alive: int):
        return self._alive_source[index_in_alive]

    def source_indices(self):
        for i in range(self.effective_batch_size()):
            yield self._alive_source[i]


def get_bound_length(batch_size, lens, min_len, max_len, min_len_factor, max_len_factor):
    min_lens = None
    max_lens = None
    lens = lens.float()
    # 1. coarse-grained
    if min_len:
        min_lens = torch.as_tensor([min_len] * batch_size).long()
    if max_len:
        max_lens = torch.as_tensor([max_len] * batch_size).long()
    # 2. fine-grained
    if lens is not None:
        if min_len_factor:
            f_min_lens = lens * min_len_factor
            min_lens = f_min_lens if min_lens is None else torch.max(f_min_lens, min_lens)
        if max_len_factor:
            f_max_lens = lens * max_len_factor
            max_lens = f_max_lens if max_lens is None else torch.min(f_max_lens, max_lens)

    # plus 1, for bos token
    if min_lens is not None:
        min_lens += 1
    # plus 1, for eos token
    if max_lens is not None:
        max_lens += 1

    if min_lens is not None and max_lens is not None:
        assert (min_lens < max_lens).all()

    return cuda(min_lens), cuda(max_lens)


def beamsearch_kbest(fn, state, lens, batch_size, beam_width, eos: int, bos: int = None,
                     length_penalty: float = 1.0,
                     min_len_factor: float = 0.5, max_len_factor: float = 3.0,
                     min_len: int = None, max_len: int = None,
                     topk: int = 1,
                     stop_criteria: str = 'find_K_ended',
                     expand_args: bool = False) -> List[List[Path]]:
    """
    Args:
        fn: A callable function that takes `state` as input and outputs a tuple of (log_prob, new_state).
        state: A tuple, list or a dictionary. This is the initial state of the decoding process.
        lens: A list of ints representing source sequence lengths.
            Setting it to `None` will disable fine-grained length constraint.
        batch_size:
        beam_width:
        eos:
        bos: (Optional.) if not provided, reuse eos as bos.
        length_penalty:
        min_len_factor: Fine-grained constraint over each output sequence's length.
        max_len_factor:
        min_len: Coarse-grained constraint over all output sequences' length.
        max_len:
        topk:
        device:
        stop_criteria: Available options: ['find_K_ended', 'top_path_ended'],
            also support rules combination. For example, 'find_K_ended || top_path_ended' for or logic,
             or 'A && B' for and logic.
        expand_args: feed expanded `state` as args to `fn`.
    Returns:

    """
    B = batch_size

    if lens is not None:
        assert len(lens) == B, (len(lens), B)

    bos = eos if bos is None else bos

    # set up constraint as the intersection of the intervals.
    min_lens, max_lens = get_bound_length(batch_size, lens, min_len, max_len, min_len_factor, max_len_factor)

    # initialize beam
    beam = Beam(beam_width, [[Path([bos])] for _ in range(B)])
    is_path_ended = lambda path: path.nodes[-1] == eos

    # set up stopping criteria
    criterion = StopCriterion(beam_width, min_lens, max_lens, is_path_ended)
    stop_criteria = get_stop_criteria(criterion, stop_criteria)

    length = 1  # including bos token
    while not beam.empty():
        B = beam.effective_batch_size()
        # batch size might be changed when a source meets stopping criteria
        # input = [path.nodes[-1] for paths in beam.alive for path in paths]
        # input = to_cuda(torch.as_tensor(input).long()).unsqueeze(1)  # BK x 1
        input = [[path.nodes for path in paths] for paths in beam.alive]
        input = cuda(torch.as_tensor(input).long())  # B x K x T
        input = input.view(-1, input.size(-1))  # BK x T

        # log_prob is of shape BK x 1 x V
        if expand_args:
            if isinstance(state, (tuple, list)):
                log_prob, state = fn(input, *state)
            elif isinstance(state, dict):
                log_prob, state = fn(input, **state)
            else:
                log_prob, state = fn(input, state)
        else:
            log_prob, state = fn(input, state)

        log_prob = log_prob.view(B, -1, log_prob.size(-1))  # B x K x V
        K_ = log_prob.size(1)  # 1 at first step and K at succeeding steps.

        # tweak probs
        idxs = torch.as_tensor(list(beam.source_indices())).long()

        if min_lens is not None:
            mask = length < min_lens[idxs]
            if mask.any():
                mask = cuda(mask)
                log_prob[:, :, eos].masked_fill_(mask.view(-1, 1), -numpy.inf)
        if max_lens is not None:
            mask = length == max_lens[idxs] - 1
            if mask.any():
                mask = cuda(mask)
                log_prob[:, :, :eos].masked_fill_(mask.view(-1, 1, 1), -numpy.inf)
                log_prob[:, :, eos + 1:].masked_fill_(mask.view(-1, 1, 1), -numpy.inf)

        if K_ == 1:
            repeat_idxs = torch.arange(B).view(-1, 1).expand(-1, beam_width).flatten()
            repeat_idxs = cuda(torch.as_tensor(repeat_idxs).long())
            state = select(state, repeat_idxs)

        beam_idxs = beam.forward(log_prob, is_path_ended, stop_criteria)

        if beam_idxs is not None:
            # B' x K
            beam_idxs = cuda(torch.as_tensor(beam_idxs).long())
            beam_idxs = beam_idxs.view(-1)
            state = select(state, beam_idxs)

        length += 1

    def rescore(path: Path):
        """
        Re-score the path.
        """
        score = path.score / (len(path) - 1)  # exclude bos token
        path.penalized_score = score

    def top(paths: List[Path], k=None):
        """
        Select from paths
        """
        paths.sort(
            key=lambda path: path.penalized_score
            if path.penalized_score is not None else path.score,
            reverse=True)
        return paths[:(k or 1)]

    ended = beam.ended

    for paths in ended:
        for path in paths:
            rescore(path)

    paths = [top(paths, topk) for paths in ended]

    return paths


def beamsearch(fn, state, lens, batch_size, beam_width, eos: int, bos: int = None,
               length_penalty: float = 1.0,
               min_len_factor: float = 0.5, max_len_factor: float = 3.0,
               min_len: int = None, max_len: int = None,
               stop_criteria: str = 'find_K_ended',
               expand_args: bool = False) -> List[Path]:
    """
    See `beamsearch_kbest`
    """
    paths = beamsearch_kbest(fn, state, lens, batch_size, beam_width, eos, bos=bos,
                             length_penalty=length_penalty,
                             min_len_factor=min_len_factor, max_len_factor=max_len_factor,
                             min_len=min_len, max_len=max_len,
                             topk=1,
                             stop_criteria=stop_criteria,
                             expand_args=expand_args)
    return [x[0] for x in paths]
