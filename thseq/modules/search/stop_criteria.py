from typing import List, Callable

from thseq.modules.search.path import Path


class StopCriterion(object):
    def __init__(self, beam_width: int, min_lens: List[int], max_lens: List[int],
                 is_path_ended: Callable[[Path], bool]):
        super().__init__()
        self.beam_width = beam_width
        self.min_lens = min_lens
        self.max_lens = max_lens
        self.is_path_ended = is_path_ended

    def reach_max_len(self, alive: List[Path], ended: List[Path], idx: int):
        return self.max_lens is not None and len(alive[0]) >= self.max_lens[idx]

    def find_K_ended(self, alive: List[Path], ended: List[Path], idx: int):
        # stop when the number of ended paths reaches K
        return len(ended) >= self.beam_width

    def top_path_ended(self, alive: List[Path], ended: List[Path], idx: int):
        # stop when a ended path have the highest score than all alive path.
        if ended:
            p0 = max(alive, key=lambda p: p.score)
            p1 = max(ended, key=lambda p: p.score)
            return p1.score > p0.score
        return False

    def composite_or(self, *criterion):
        def f(alive: List[Path], ended: List[Path], idx: int):
            return any(criteria(alive, ended, idx) for criteria in criterion)

        return f

    def composite_and(self, *criterion):
        def f(alive: List[Path], ended: List[Path], idx: int):
            return all(criteria(alive, ended, idx) for criteria in criterion)

        return f


def get_stop_criteria(criterion: StopCriterion, str_criteria: str, bound_max_len=True):
    and_logic = [x.strip() for x in str_criteria.split('&&')]
    or_logic = [x.strip() for x in str_criteria.split('||')]

    if len(and_logic) > 1:
        criteria = criterion.composite_and(*[getattr(criterion, x) for x in and_logic])
    elif len(or_logic) > 1:
        criteria = criterion.composite_or(*[getattr(criterion, x) for x in or_logic])
    else:
        criteria = getattr(criterion, str_criteria)

    if bound_max_len:
        criteria = criterion.composite_or(criteria, criterion.reach_max_len)

    return criteria
