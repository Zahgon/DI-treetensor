from functools import wraps
from typing import Callable, Union, Any

__all__ = [
    'replaceable_partial',
    'args_mapping',
]


def replaceable_partial(func, **kws):
    @wraps(func)
    def _new_func(*args, **kwargs):
        pass

    return _new_func


def args_mapping(mapper: Callable[[Union[int, str], Any], Any]):
    def _decorator(func):
        pass

    return _decorator
