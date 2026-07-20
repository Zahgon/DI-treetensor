import warnings
from functools import wraps
from typing import Optional

import torch

from ...common import ireduce

__all__ = ['rmreduce', 'post_reduce', 'auto_reduce']


def _reduce_func(rfunc):
    rfunc = rfunc or (lambda x: x)

    def _new_func(ts):
        pass

    return _new_func


def rmreduce(rfunc=None):
    return ireduce(_reduce_func(rfunc))


def post_reduce(rfunc=None, prefunc=None):
    rfunc = rfunc or (lambda x, *args, **kwargs: x)

    def _decorator(func):
        pass

    return _decorator


def _default_auto_determine(*args, out=None, **kwargs):
    pass


def _default_auto_condition(*args, out=None, **kwargs):
    pass


def auto_reduce(rfunc, nrfunc, determine=None, condition=None):
    determine = determine or _default_auto_determine
    condition = condition or _default_auto_condition

    def _decorator(func):
        pass

    return _decorator
