import torch

from .base import doc_from_base, func_treelize
from ...common import return_self

__all__ = [
    'detach', 'detach_'
]


@doc_from_base()
@func_treelize()
def detach(input):
    pass


@doc_from_base()
@return_self
@func_treelize()
def detach_(input):
    pass
