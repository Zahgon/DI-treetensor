import builtins

import torch

from .base import doc_from_base, func_treelize
from ..stream import stream_call
from ...common import ireduce

__all__ = [
    'equal',
    'isfinite', 'isinf', 'isnan', 'isclose',
    'eq', 'ne', 'lt', 'le', 'gt', 'ge',
]


@doc_from_base()
@ireduce(builtins.all)
@func_treelize()
def equal(input, other):
    pass


@doc_from_base()
@func_treelize()
def isfinite(input):
    pass


@doc_from_base()
@func_treelize()
def isinf(input):
    pass


@doc_from_base()
@func_treelize()
def isnan(input):
    pass


@doc_from_base()
@func_treelize()
def isclose(input, other, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize()
def eq(input, other, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize()
def ne(input, other, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize()
def lt(input, other, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize()
def le(input, other, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize()
def gt(input, other, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize()
def ge(input, other, *args, **kwargs):
    pass
