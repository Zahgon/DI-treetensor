from functools import lru_cache

import numpy
import torch
from treevalue import method_treelize

from .base import TreeNumpy
from ..common import Object, ireduce, clsmeta, get_tree_proxy
from ..utils import current_names

__all__ = [
    'ndarray'
]

_ArrayProxy, _InstanceArrayProxy = get_tree_proxy(numpy.ndarray)


@lru_cache()
def _get_tensor_class(args0):
    pass


class _BaseArrayMeta(clsmeta(numpy.asarray, allow_dict=True)):
    pass


class _ArrayMeta(_BaseArrayMeta):
    def __init__(cls, *args, **kwargs):
        _BaseArrayMeta.__init__(cls, *args, **kwargs)
        cls.__proxy = None

    @property
    def np(cls):
        pass

    def __getattr__(cls, name):
        try:
            return cls.np.__getattr__(name)
        except AttributeError:
            raise AttributeError(f"type object {repr(cls.__name__)} has no attribute {repr(name)}")


@current_names()
class ndarray(TreeNumpy, metaclass=_ArrayMeta):

    @method_treelize(return_type=Object)
    def __get_attr(self, key):
        pass

    def _attr_extern(self, name):
        pass

    @property
    def np(self):
        pass

    @method_treelize(return_type=Object)
    def tolist(self: numpy.ndarray):
        pass

    @property
    @ireduce(sum)
    @method_treelize(return_type=Object)
    def size(self: numpy.ndarray) -> int:
        pass

    @property
    @ireduce(sum)
    @method_treelize(return_type=Object)
    def nbytes(self: numpy.ndarray) -> int:
        pass

    @ireduce(sum)
    @method_treelize(return_type=Object)
    def sum(self: numpy.ndarray, *args, **kwargs):
        pass

    @ireduce(all)
    @method_treelize(return_type=Object)
    def all(self: numpy.ndarray, *args, **kwargs):
        pass

    @ireduce(any)
    @method_treelize(return_type=Object)
    def any(self: numpy.ndarray, *args, **kwargs):
        pass

    @method_treelize(return_type=_get_tensor_class)
    def tensor(self: numpy.ndarray, *args, **kwargs):
        tensor_: torch.Tensor = torch.from_numpy(self)
        if args or kwargs:
            tensor_ = tensor_.to(*args, **kwargs)
        return tensor_

    @method_treelize()
    def __eq__(self, other):
        """
        See :func:`treetensor.numpy.eq`.
        """
        return self == other

    @method_treelize()
    def __ne__(self, other):
        """
        See :func:`treetensor.numpy.ne`.
        """
        return self != other

    @method_treelize()
    def __lt__(self, other):
        """
        See :func:`treetensor.numpy.lt`.
        """
        return self < other

    @method_treelize()
    def __gt__(self, other):
        """
        See :func:`treetensor.numpy.gt`.
        """
        return self > other

    @method_treelize()
    def __le__(self, other):
        """
        See :func:`treetensor.numpy.le`.
        """
        return self <= other

    @method_treelize()
    def __ge__(self, other):
        """
        See :func:`treetensor.numpy.ge`.
        """
        return self >= other
