from functools import wraps

import torch
from hbutils.reflection import post_process
from treevalue import TreeValue
from treevalue import func_treelize as original_func_treelize
from treevalue.tree.common import TreeStorage

from .base import Torch
from ..common import Object, clsmeta, ireduce
from ..utils import doc_from_base as original_doc_from_base
from ..utils import replaceable_partial, current_names, args_mapping

func_treelize = post_process(post_process(args_mapping(
    lambda i, x: TreeValue(x) if isinstance(x, (dict, TreeStorage, TreeValue)) else x)))(
    replaceable_partial(original_func_treelize)
)
doc_from_base = replaceable_partial(original_doc_from_base, base=torch.Size)

__all__ = [
    'Size'
]


def _post_index(func):
    pass


@current_names()
class Size(Torch, metaclass=clsmeta(torch.Size, allow_dict=True)):
    def __init__(self, data):
        """
        In :class:`treetensor.torch.Size`, it's similar with the original :class:`torch.Size`.

        Examples::

            >>> import torch
            >>> import treetensor.numpy as ttorch
            >>> ttorch.Size([1, 2, 3])
            torch.Size([1, 2, 3])

            >>> ttorch.Size({
            ...     'a': [1, 2, 3],
            ...     'b': {'x': [3, 4, ]},
            ...     'c': [5],
            ... })
            <Size 0x7fe00b115970>
            ├── a --> torch.Size([1, 2, 3])
            ├── b --> <Size 0x7fe00b115250>
            │   └── x --> torch.Size([3, 4])
            └── c --> torch.Size([5])
        """
        super(Torch, self).__init__(data)

    @doc_from_base()
    @ireduce(sum)
    @func_treelize(return_type=Object)
    def numel(self: torch.Size) -> Object:
        pass

    @doc_from_base()
    @_post_index
    @func_treelize(return_type=Object)
    def index(self: torch.Size, value, *args, **kwargs) -> Object:
        pass

    @doc_from_base()
    @ireduce(sum)
    @func_treelize(return_type=Object)
    def count(self: torch.Size, *args, **kwargs) -> Object:
        """
        Get the occurrence count of the sizes in this tree.

        Example::

            >>> import torch
            >>> import treetensor.numpy as ttorch
            >>> ttorch.Size({
            ...     'a': [1, 2],
            ...     'b': {'x': [3, 2, 4]},
            ... }).count(2)
            2
        """
        return self.count(*args, **kwargs)
