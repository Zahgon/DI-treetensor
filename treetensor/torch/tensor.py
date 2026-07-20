import numpy as np
import torch as pytorch
from hbutils.reflection import post_process
from treevalue import method_treelize, TreeValue, typetrans

from .base import Torch, rmreduce, post_reduce, auto_reduce
from .size import Size
from .stream import stream_call
from ..common import Object, ireduce, clsmeta, return_self, auto_tree, get_tree_proxy
from ..numpy import ndarray
from ..utils import current_names, class_autoremove, replaceable_partial
from ..utils import doc_from_base as original_doc_from_base

__all__ = [
    'Tensor'
]

doc_from_base = replaceable_partial(original_doc_from_base, base=pytorch.Tensor)


def _auto_tensor(t):
    pass


_TorchProxy, _InstanceTorchProxy = get_tree_proxy(pytorch.Tensor, _auto_tensor)


def _to_tensor(data, *args, **kwargs):
    pass


class _BaseTensorMeta(clsmeta(_to_tensor, allow_dict=True)):
    pass


class _TensorMeta(_BaseTensorMeta):
    def __init__(cls, *args, **kwargs):
        _BaseTensorMeta.__init__(cls, *args, **kwargs)
        cls.__proxy = None

    @property
    def torch(cls):
        pass

    def __getattr__(cls, name):
        try:
            return cls.torch.__getattr__(name)
        except AttributeError:
            raise AttributeError(f"type object {repr(cls.__name__)} has no attribute {repr(name)}")


@current_names()
@class_autoremove
class Tensor(Torch, metaclass=_TensorMeta):
    __auto_tensor = lambda x: replaceable_partial(
        auto_tree,
        cls=[(pytorch.is_tensor, Tensor)]
    )(x)

    def __init__(self, data, *args, **kwargs):
        """
        In :class:`treetensor.torch.Tensor`, it's similar but a little bit different with the
        original :class:`torch.Tensor`.

        Examples::

            >>> import torch
            >>> import treetensor.torch as ttorch
            >>> pytorch.Tensor([1, 2, 3])  # in torch.Tensor, default type is float32
            tensor([1., 2., 3.])

            >>> ttorch.Tensor([1, 2, 3])  # a native Tensor object, its type is auto detected with torch.tensor
            tensor([1, 2, 3])

            >>> ttorch.Tensor([1, 2, 3], dtype=pytorch.float32)  # with float32 type
            tensor([1., 2., 3.])

            >>> ttorch.Tensor({
            ...     'a': [1, 2, 3],
            ...     'b': {'x': [4.0, 5, 6]},
            ...     'c': [[True, ], [False, ]],
            ... })  # a tree-based Tensor object
            <Tensor 0x7f537bb9a880>
            ├── a --> tensor([1, 2, 3])
            ├── b --> <Tensor 0x7f537bb9a0d0>
            │   └── x --> tensor([4., 5., 6.])
            └── c --> tensor([[ True],
                              [False]])
        """
        super(Torch, self).__init__(data)

    @method_treelize(return_type=Object)
    def __get_attr(self, key):
        pass

    def _attr_extern(self, name):
        pass

    @property
    def torch(self):
        pass

    @doc_from_base()
    @method_treelize(return_type=ndarray)
    def numpy(self: pytorch.Tensor) -> np.ndarray:
        pass

    @doc_from_base()
    @method_treelize(return_type=Object)
    def tolist(self: pytorch.Tensor):
        pass

    @doc_from_base()
    @method_treelize()
    def cpu(self: pytorch.Tensor, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def cuda(self: pytorch.Tensor, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def to(self: pytorch.Tensor, *args, **kwargs):
        """
        Turn the original tree tensor to another format.

        Example::

            >>> import torch
            >>> import treetensor.torch as ttorch
            >>> ttorch.tensor({
            ...     'a': [[1, 11], [2, 22], [3, 33]],
            ...     'b': {'x': [[4, 5], [6, 7]]},
            ... }).to(pytorch.float64)
            <Tensor 0x7ff363bb6518>
            ├── a --> tensor([[ 1., 11.],
            │                 [ 2., 22.],
            │                 [ 3., 33.]], dtype=torch.float64)
            └── b --> <Tensor 0x7ff363bb6ef0>
                └── x --> tensor([[4., 5.],
                                  [6., 7.]], dtype=torch.float64)
        """
        return stream_call(self.to, *args, **kwargs)

    @doc_from_base()
    @ireduce(sum)
    @method_treelize(return_type=Object)
    def numel(self: pytorch.Tensor):
        pass

    @property
    @doc_from_base()
    @method_treelize(return_type=Size)
    def shape(self: pytorch.Tensor):
        pass

    @property
    @method_treelize()
    def grad(self):
        pass

    @property
    @method_treelize(return_type=Object)
    def requires_grad(self):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def requires_grad_(self, requires_grad=True):
        pass

    @doc_from_base()
    @method_treelize()
    def detach(self):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def detach_(self):
        pass

    @post_reduce(pytorch.all)
    @method_treelize(return_type=Object)
    def __all_r(self, *args, **kwargs):
        pass

    @method_treelize()
    def __all_nr(self, *args, **kwargs):
        pass

    @doc_from_base()
    @auto_reduce(__all_r, __all_nr)
    def all(self: pytorch.Tensor, *args, reduce=None, **kwargs) -> bool:
        """
        See :func:`treetensor.torch.all`
        """
        pass  # pragma: no cover

    @post_reduce(pytorch.any)
    @method_treelize(return_type=Object)
    def __any_r(self, *args, **kwargs):
        pass

    @method_treelize()
    def __any_nr(self, *args, **kwargs):
        pass

    @doc_from_base()
    @auto_reduce(__any_r, __any_nr)
    def any(self: pytorch.Tensor, *args, reduce=None, **kwargs) -> bool:
        """
        See :func:`treetensor.torch.any`
        """
        pass  # pragma: no cover

    @post_reduce(pytorch.max)
    @method_treelize(return_type=Object)
    def __max_r(self, *args, **kwargs):
        pass

    @post_process(__auto_tensor)
    @method_treelize(return_type=TreeValue, rise=True)
    def __max_nr(self, *args, **kwargs):
        pass

    @doc_from_base()
    @auto_reduce(__max_r, __max_nr)
    def max(self: pytorch.Tensor, *args, reduce=None, **kwargs):
        """
        See :func:`treetensor.torch.max`
        """
        pass  # pragma: no cover

    @post_reduce(pytorch.min)
    @method_treelize(return_type=Object)
    def __min_r(self, *args, **kwargs):
        pass

    @post_process(__auto_tensor)
    @method_treelize(return_type=TreeValue, rise=True)
    def __min_nr(self, *args, **kwargs):
        pass

    @doc_from_base()
    @auto_reduce(__min_r, __min_nr)
    def min(self: pytorch.Tensor, *args, reduce=None, **kwargs):
        """
        See :func:`treetensor.torch.min`
        """
        pass  # pragma: no cover

    @post_reduce(pytorch.sum)
    @method_treelize(return_type=Object)
    def __sum_r(self, *args, **kwargs):
        pass

    @post_process(__auto_tensor)
    @method_treelize(return_type=TreeValue, rise=True)
    def __sum_nr(self, *args, **kwargs):
        pass

    @doc_from_base()
    @auto_reduce(__sum_r, __sum_nr)
    def sum(self: pytorch.Tensor, *args, reduce=None, **kwargs):
        """
        See :func:`treetensor.torch.sum`
        """
        pass  # pragma: no cover

    @method_treelize()
    def __eq__(self, other):
        """
        See :func:`treetensor.torch.eq`.
        """
        return self == other

    @method_treelize()
    def __ne__(self, other):
        """
        See :func:`treetensor.torch.ne`.
        """
        return self != other

    @method_treelize()
    def __lt__(self, other):
        """
        See :func:`treetensor.torch.lt`.
        """
        return self < other

    @method_treelize()
    def __gt__(self, other):
        """
        See :func:`treetensor.torch.gt`.
        """
        return self > other

    @method_treelize()
    def __le__(self, other):
        """
        See :func:`treetensor.torch.le`.
        """
        return self <= other

    @method_treelize()
    def __ge__(self, other):
        """
        See :func:`treetensor.torch.ge`.
        """
        return self >= other

    @doc_from_base()
    @method_treelize()
    def clone(self, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def dot(self, other, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def mm(self, mat2, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def matmul(self, tensor2, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def isfinite(self):
        pass

    @doc_from_base()
    @method_treelize()
    def isinf(self):
        pass

    @doc_from_base()
    @method_treelize()
    def isnan(self):
        pass

    @doc_from_base()
    @method_treelize()
    def isclose(self, other, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def abs(self, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def abs_(self, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def clamp(self, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def clamp_(self, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def sign(self, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def sign_(self, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def sigmoid(self, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def sigmoid_(self, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def floor(self, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def floor_(self, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def ceil(self, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def ceil_(self, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def round(self, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def round_(self, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def add(self, other, *args, **kwargs):
        """
        See :func:`treetensor.torch.add`.
        """
        return stream_call(self.add, other, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def add_(self, other, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def sub(self, other, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def sub_(self, other, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def mul(self, other, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def mul_(self, other, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def div(self, other, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def div_(self, other, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def pow(self, exponent, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def pow_(self, exponent, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def neg(self, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def neg_(self, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def exp(self, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def exp_(self, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def exp2(self, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def exp2_(self, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def sqrt(self, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def sqrt_(self, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def log(self, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def log_(self, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def log2(self, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def log2_(self, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def log10(self, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def log10_(self, *args, **kwargs):
        pass

    @doc_from_base()
    @post_process(__auto_tensor)
    @method_treelize(return_type=TreeValue, rise=True)
    def split(self, split_size, *args, **kwargs):
        pass

    @doc_from_base()
    @post_process(__auto_tensor)
    @method_treelize(return_type=TreeValue, rise=True)
    def chunk(self, chunks, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def reshape(self, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def squeeze(self, *args, **kwargs):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def squeeze_(self, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def unsqueeze(self, dim):
        pass

    @doc_from_base()
    @return_self
    @method_treelize()
    def unsqueeze_(self, dim):
        pass

    @doc_from_base()
    @method_treelize()
    def where(self, condition, y, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def index_select(self, dim, index):
        pass

    @rmreduce()
    @method_treelize(return_type=Object)
    def __masked_select_r(self, mask, *args, **kwargs):
        pass

    @method_treelize()
    def __masked_select_nr(self, mask, *args, **kwargs):
        pass

    def __ms_determine(mask, *args, out=None, **kwargs):
        pass

    def __ms_condition(mask, *args, out=None, **kwargs):
        pass

    @doc_from_base()
    @auto_reduce(__masked_select_r, __masked_select_nr,
                 __ms_determine, __ms_condition)
    def masked_select(self, mask):
        """
        See :func:`treetensor.torch.masked_select`.
        """
        pass  # pragma: no cover

    @post_reduce(pytorch.std)
    @method_treelize(return_type=Object)
    def __std_r(self, *args, **kwargs):
        pass

    @method_treelize()
    def __std_nr(self, *args, **kwargs):
        pass

    @doc_from_base()
    @auto_reduce(__std_r, __std_nr)
    @method_treelize()
    def std(self, *args, reduce=None, **kwargs):
        """
        See :func:`treetensor.torch.std`.
        """
        pass  # pragma: no cover

    @post_reduce(pytorch.mean)
    @method_treelize(return_type=Object)
    def __mean_r(self, *args, **kwargs):
        pass

    @method_treelize()
    def __mean_nr(self, *args, **kwargs):
        pass

    @doc_from_base()
    @auto_reduce(__mean_r, __mean_nr)
    @method_treelize()
    def mean(self, *args, reduce=None, **kwargs):
        """
        See :func:`treetensor.torch.mean`.
        """
        pass  # pragma: no cover

    @doc_from_base()
    @method_treelize()
    def dist(self, other, *args, **kwargs):
        pass

    @doc_from_base()
    @method_treelize()
    def norm(self, *args, **kwargs):
        pass
