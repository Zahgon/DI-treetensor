import torch

from .base import doc_from_base, func_treelize
from ..stream import stream_call
from ...common import return_self

__all__ = [
    'abs', 'abs_', 'clamp', 'clamp_', 'sign', 'sigmoid', 'sigmoid_',
    'round', 'round_', 'floor', 'floor_', 'ceil', 'ceil_',
    'add', 'sub', 'mul', 'div', 'pow', 'neg', 'neg_',
    'exp', 'exp_', 'exp2', 'exp2_', 'sqrt', 'sqrt_',
    'log', 'log_', 'log2', 'log2_', 'log10', 'log10_',
    'dist', 'norm',
]


@doc_from_base()
@func_treelize()
def abs(input, *args, **kwargs):
    pass


@doc_from_base()
@return_self
@func_treelize()
def abs_(input):
    pass


@doc_from_base()
@func_treelize()
def clamp(input, *args, **kwargs):
    pass


@doc_from_base()
@return_self
@func_treelize()
def clamp_(input, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize()
def sign(input, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize()
def round(input, *args, **kwargs):
    pass


@doc_from_base()
@return_self
@func_treelize()
def round_(input):
    pass


@doc_from_base()
@func_treelize()
def floor(input, *args, **kwargs):
    pass


@doc_from_base()
@return_self
@func_treelize()
def floor_(input):
    pass


@doc_from_base()
@func_treelize()
def ceil(input, *args, **kwargs):
    pass


@doc_from_base()
@return_self
@func_treelize()
def ceil_(input):
    pass


@doc_from_base()
@func_treelize()
def sigmoid(input, *args, **kwargs):
    pass


@doc_from_base()
@return_self
@func_treelize()
def sigmoid_(input):
    pass


@doc_from_base()
@func_treelize()
def add(input, other, *args, **kwargs):
    """
    Adds the scalar ``other`` to each element of the ``input`` input and
    returns a new resulting tree tensor.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.add(
        ...     ttorch.tensor([1, 2, 3]),
        ...     ttorch.tensor([3, 5, 11]),
        ... )
        tensor([ 4,  7, 14])

        >>> ttorch.add(
        ...     ttorch.tensor({
        ...         'a': [1, 2, 3],
        ...         'b': {'x': [[3, 5], [9, 12]]},
        ...     }),
        ...     ttorch.tensor({
        ...         'a': [3, 5, 11],
        ...         'b': {'x': [[31, -15], [13, 23]]},
        ...     })
        ... )
        <Tensor 0x7f11b139c710>
        ├── a --> tensor([ 4,  7, 14])
        └── b --> <Tensor 0x7f11b139c630>
            └── x --> tensor([[ 34, -10],
                              [ 22,  35]])
    """
    return stream_call(torch.add, input, other, *args, **kwargs)


@doc_from_base()
@func_treelize()
def sub(input, other, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize()
def mul(input, other, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize()
def div(input, other, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize()
def pow(input, exponent, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize()
def neg(input, *args, **kwargs):
    pass


@doc_from_base()
@return_self
@func_treelize()
def neg_(input):
    pass


@doc_from_base()
@func_treelize()
def exp(input, *args, **kwargs):
    pass


@doc_from_base()
@return_self
@func_treelize()
def exp_(input):
    pass


@doc_from_base()
@func_treelize()
def exp2(input, *args, **kwargs):
    pass


@doc_from_base()
@return_self
@func_treelize()
def exp2_(input):
    pass


@doc_from_base()
@func_treelize()
def sqrt(input, *args, **kwargs):
    pass


@doc_from_base()
@return_self
@func_treelize()
def sqrt_(input):
    pass


@doc_from_base()
@func_treelize()
def log(input, *args, **kwargs):
    pass


@doc_from_base()
@return_self
@func_treelize()
def log_(input):
    pass


@doc_from_base()
@func_treelize()
def log2(input, *args, **kwargs):
    pass


@doc_from_base()
@return_self
@func_treelize()
def log2_(input):
    pass


@doc_from_base()
@func_treelize()
def log10(input, *args, **kwargs):
    pass


@doc_from_base()
@return_self
@func_treelize()
def log10_(input):
    pass


@doc_from_base()
@func_treelize()
def dist(input, other, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize()
def norm(input, *args, **kwargs):
    pass
