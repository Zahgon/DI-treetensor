import torch

from .base import doc_from_base, func_treelize
from ..stream import stream_call

__all__ = [
    'dot', 'matmul', 'mm',
]


@doc_from_base()
@func_treelize()
def dot(input, other, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize()
def matmul(input, other, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize()
def mm(input, mat2, *args, **kwargs):
    pass
