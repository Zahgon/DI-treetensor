import torch
from treevalue import TreeValue
from treevalue.tree.common import TreeStorage

from .base import doc_from_base, func_treelize
from ..stream import stream_call
from ...utils import args_mapping

__all__ = [
    'tensor', 'as_tensor', 'clone',
    'zeros', 'zeros_like',
    'randn', 'randn_like',
    'rand', 'rand_like',
    'randint', 'randint_like',
    'ones', 'ones_like',
    'full', 'full_like',
    'empty', 'empty_like',
]

args_treelize = args_mapping(lambda i, x: TreeValue(x) if isinstance(x, (dict, TreeStorage, TreeValue)) else x)


@doc_from_base()
@args_treelize
@func_treelize()
def tensor(data, *args, **kwargs):
    """
    In ``treetensor``, you can create a tree tensor with simple data structure.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.tensor(True)  # the same as torch.tensor(True)
        tensor(True)

        >>> ttorch.tensor([1, 2, 3])  # the same as torch.tensor([1, 2, 3])
        tensor([1, 2, 3])

        >>> ttorch.tensor({'a': 1, 'b': [1, 2, 3], 'c': [[True, False], [False, True]]})
        <Tensor 0x7ff363bbcc50>
        ├── a --> tensor(1)
        ├── b --> tensor([1, 2, 3])
        └── c --> tensor([[ True, False],
                          [False,  True]])
    """
    return stream_call(torch.tensor, data, *args, **kwargs)


@doc_from_base()
@args_treelize
@func_treelize()
def as_tensor(data, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize()
def clone(input, *args, **kwargs):
    pass


@doc_from_base()
@args_treelize
@func_treelize()
def zeros(*args, **kwargs):
    pass


@doc_from_base()
@args_treelize
@func_treelize()
def zeros_like(input, *args, **kwargs):
    pass


@doc_from_base()
@args_treelize
@func_treelize()
def randn(*args, **kwargs):
    pass


@doc_from_base()
@args_treelize
@func_treelize()
def randn_like(input, *args, **kwargs):
    pass


@doc_from_base()
@args_treelize
@func_treelize()
def rand(*args, **kwargs):
    pass


@doc_from_base()
@args_treelize
@func_treelize()
def rand_like(input, *args, **kwargs):
    pass


@doc_from_base()
@args_treelize
@func_treelize()
def randint(*args, **kwargs):
    pass


@doc_from_base()
@args_treelize
@func_treelize()
def randint_like(input, *args, **kwargs):
    pass


@doc_from_base()
@args_treelize
@func_treelize()
def ones(*args, **kwargs):
    pass


@doc_from_base()
@args_treelize
@func_treelize()
def ones_like(input, *args, **kwargs):
    pass


@doc_from_base()
@args_treelize
@func_treelize()
def full(*args, **kwargs):
    pass


@doc_from_base()
@args_treelize
@func_treelize()
def full_like(input, *args, **kwargs):
    pass


@doc_from_base()
@args_treelize
@func_treelize()
def empty(*args, **kwargs):
    pass


@doc_from_base()
@args_treelize
@func_treelize()
def empty_like(input, *args, **kwargs):
    pass
