from .reflection import removed

__all__ = [
    'doc_from', 'doc_from_base',
]

_DOC_FROM_TAG = '__doc_from__'


def doc_from(src):
    def _decorator(obj):
        pass

    return _decorator


def doc_from_base(base, name: str = None):
    def _decorator(func):
        pass

    return _decorator
