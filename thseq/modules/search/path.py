from typing import List


class Path(object):
    def __init__(self, nodes: List = None, node_scores: List[float] = None, score: float = 0.):
        nodes = nodes or []
        score = score or 0.
        node_scores = node_scores or [0.] * len(nodes)
        self.nodes: List = nodes
        self.node_scores: List[float] = node_scores
        self.score: float = score
        self._ended = False
        self._sources: List = []
        self.penalized_score = None

    def get_latest_source(self, default=0):
        """
        Get the source where the last node came from
        """
        if len(self._sources) > 0:
            return self._sources[-1]
        else:
            return default

    def extend(self, node, node_score: float = 0., score: float = 0., source=None):
        if self.ended:
            raise ValueError('Cannot add node to an ended path.')
        path = Path(self.nodes + [node], self.node_scores + [node_score], score or self.score + node_score)
        if source is not None:
            path._sources = self._sources + [source]
        return path

    def end(self):
        self._ended = True

    @property
    def ended(self):
        return self._ended

    def __len__(self):
        return len(self.nodes)
