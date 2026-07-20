import torch
from hbutils.reflection import post_process
from treevalue import TreeValue

from .base import doc_from_base, func_treelize, auto_tensor
from ..stream import stream_call

__all__ = [
    'cat', 'split', 'chunk', 'stack',
    'reshape', 'where', 'squeeze', 'unsqueeze',
    'index_select',
]


@doc_from_base()
@func_treelize(subside=True)
def cat(tensors, *args, **kwargs):
    """
    Concatenates the given sequence of ``seq`` tensors in the given dimension.
    All tensors must either have the same shape (except in the concatenating dimension) or be empty.

    Examples:

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t1 = torch.randint(10, 30, (2, 3))
        >>> t1
        tensor([[21, 29, 17],
                [16, 11, 16]])
        >>> t2 = torch.randint(30, 50, (2, 3))
        tensor([[46, 46, 46],
                [30, 47, 36]])
        >>> t2
        >>> t3 = torch.randint(50, 70, (2, 3))
        tensor([[51, 65, 65],
                [54, 67, 57]])
        >>> t3
        >>> ttorch.cat((t1, t2, t3))
        tensor([[21, 29, 17],
                [16, 11, 16],
                [46, 46, 46],
                [30, 47, 36],
                [51, 65, 65],
                [54, 67, 57]])

        >>> tt1 = ttorch.Tensor({
        ...    'a': t1,
        ...    'b': {'x': t2, 'y': t3},
        ... })
        >>> tt1
        <Tensor 0x7fed579acf60>
        ├── a --> tensor([[21, 29, 17],
        │                 [16, 11, 16]])
        └── b --> <Tensor 0x7fed579acf28>
            ├── x --> tensor([[46, 46, 46],
            │                 [30, 47, 36]])
            └── y --> tensor([[51, 65, 65],
                              [54, 67, 57]])
        >>> tt2 = ttorch.Tensor({
        ...    'a': t2,
        ...    'b': {'x': t3, 'y': t1},
        ... })
        >>> tt2
        <Tensor 0x7fed579d62e8>
        ├── a --> tensor([[46, 46, 46],
        │                 [30, 47, 36]])
        └── b --> <Tensor 0x7fed579d62b0>
            ├── x --> tensor([[51, 65, 65],
            │                 [54, 67, 57]])
            └── y --> tensor([[21, 29, 17],
                              [16, 11, 16]])
        >>> tt3 = ttorch.Tensor({
        ...    'a': t3,
        ...    'b': {'x': t1, 'y': t2},
        ... })
        >>> tt3
        <Tensor 0x7fed579d66a0>
        ├── a --> tensor([[51, 65, 65],
        │                 [54, 67, 57]])
        └── b --> <Tensor 0x7fed579d65f8>
            ├── x --> tensor([[21, 29, 17],
            │                 [16, 11, 16]])
            └── y --> tensor([[46, 46, 46],
                              [30, 47, 36]]
        >>> ttorch.cat((tt1, tt2, tt3))
        <Tensor 0x7fed579d6ac8>
        ├── a --> tensor([[21, 29, 17],
        │                 [16, 11, 16],
        │                 [46, 46, 46],
        │                 [30, 47, 36],
        │                 [51, 65, 65],
        │                 [54, 67, 57]])
        └── b --> <Tensor 0x7fed579d6a90>
            ├── x --> tensor([[46, 46, 46],
            │                 [30, 47, 36],
            │                 [51, 65, 65],
            │                 [54, 67, 57],
            │                 [21, 29, 17],
            │                 [16, 11, 16]])
            └── y --> tensor([[51, 65, 65],
                              [54, 67, 57],
                              [21, 29, 17],
                              [16, 11, 16],
                              [46, 46, 46],
                              [30, 47, 36]])
        >>> ttorch.cat((tt1, tt2, tt3), dim=1)
        <Tensor 0x7fed579644a8>
        ├── a --> tensor([[21, 29, 17, 46, 46, 46, 51, 65, 65],
        │                 [16, 11, 16, 30, 47, 36, 54, 67, 57]])
        └── b --> <Tensor 0x7fed57964438>
            ├── x --> tensor([[46, 46, 46, 51, 65, 65, 21, 29, 17],
            │                 [30, 47, 36, 54, 67, 57, 16, 11, 16]])
            └── y --> tensor([[51, 65, 65, 21, 29, 17, 46, 46, 46],
                              [54, 67, 57, 16, 11, 16, 30, 47, 36]])
    """
    return stream_call(torch.cat, tensors, *args, **kwargs)


@doc_from_base()
@post_process(auto_tensor)
@func_treelize(return_type=TreeValue, rise=True)
def split(tensor, split_size_or_sections, *args, **kwargs):
    pass


@doc_from_base()
@post_process(auto_tensor)
@func_treelize(return_type=TreeValue, rise=True)
def chunk(input, chunks, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize(subside=True)
def stack(tensors, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize()
def reshape(input, shape):
    pass


@doc_from_base()
@func_treelize()
def squeeze(input, *args, **kwargs):
    pass


@doc_from_base()
@func_treelize()
def unsqueeze(input, dim):
    pass


@doc_from_base()
@func_treelize()
def where(condition, x, y):
    pass


@doc_from_base()
@func_treelize()
def index_select(input, dim, index, *args, **kwargs):
    pass
