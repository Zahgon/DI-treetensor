from functools import wraps

from treevalue import TreeValue, flatten_values

__all__ = [
    'ireduce',
    'return_self',
]


def ireduce(rfunc, piter=None):
    piter = piter or (lambda x: x)

    def _decorator(func):
        pass

    return _decorator


def return_self(func):
    pass
