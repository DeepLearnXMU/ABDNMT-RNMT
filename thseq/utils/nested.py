from collections import OrderedDict

import torch


def map(func, x, default=None):
    """
    Apply `func` to every tensor of x along its structure.
    Args:
        func: callable function to apply to a single tensor.
        x: nested structure
        default: callable or value for mapping non-nested non-tensor element.

    Returns:

    """

    def _map(x):
        if isinstance(x, (tuple, list)):
            rv = x.__class__(_map(item) for item in x)
        elif isinstance(x, (dict, OrderedDict)):
            rv = x.__class__((k, _map(v)) for k, v in x.items())
        elif isinstance(x, torch.Tensor):
            rv = func(x)
        else:
            rv = default(x) if callable(default) else default
        return rv

    return _map(x)


def select(x, idxs, axis=0):
    return map(lambda x: x.index_select(axis, idxs), x, lambda x: x)


def shape(x):
    return map(lambda x: list(x.size()), x)


def clone(x):
    return map(lambda x: x, x, lambda x: x)
