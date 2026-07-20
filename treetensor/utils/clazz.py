import types
from functools import reduce
from operator import __or__
from typing import Iterable, TypeVar

__all__ = [
    'tag_names', 'inherit_names', 'current_names',
]

_DOC_TAG = '__doc_names__'
_CLS_TYPE = TypeVar('_CLS_TYPE', bound=type)


def _get_names(clazz: type):
    return set(getattr(clazz, _DOC_TAG, set()))


def _set_names(clazz: type, names: Iterable[str]):
    setattr(clazz, _DOC_TAG, set(names))


def tag_names(names: Iterable[str], keep: bool = True):
    pass


def inherit_names(*clazzes: type, keep: bool = True):
    pass


class _TempClazz:
    @property
    def prop(self):
        pass


PropertyType = type(_TempClazz.prop)


def _is_property(clazz, name):
    prop = getattr(clazz, name)
    return isinstance(prop, PropertyType) and (
            not hasattr(clazz.__base__, name) or getattr(clazz.__base__, name) is not prop
    )


def _is_func(clazz, name):
    func = getattr(clazz, name)
    return isinstance(func, types.FunctionType) and (
            not hasattr(clazz.__base__, name) or getattr(clazz.__base__, name) is not func
    )


def _is_classmethod(clazz, name):
    method = getattr(clazz, name)
    return isinstance(method, types.MethodType) and (
            not hasattr(clazz.__base__, name) or getattr(clazz.__base__, name).__func__ is not method.__func__
    )


def current_names(keep: bool = True):
    def _decorator(cls: _CLS_TYPE) -> _CLS_TYPE:
        pass

    return _decorator
