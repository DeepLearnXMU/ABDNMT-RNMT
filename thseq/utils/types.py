import argparse
import functools
from collections import OrderedDict, defaultdict, deque

import numpy


def to_bool(x):
    return True if x else False

def rset_attr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rget_attr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rget_attr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


