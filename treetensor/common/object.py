import builtins

from treevalue import method_treelize

from .trees import BaseTreeStruct, clsmeta
from .wrappers import ireduce

__all__ = [
    "Object",
]


def _object(obj):
    pass


class Object(BaseTreeStruct, metaclass=clsmeta(_object, allow_dict=True)):

    def __init__(self, data):
        """
        In :class:`treetensor.common.Object`, object or object tree can be initialized.

        Examples::

            >>> from treetensor.common import Object
            >>> Object(1)
            1

            >>> Object({'a': 1, 'b': 2, 'x': {'c': 233}})
            <Object 0x7fe00b1153a0>
            ├── a --> 1
            ├── b --> 2
            └── x --> <Object 0x7fe00b115ee0>
                └── c --> 233
        """
        BaseTreeStruct.__init__(self, data)

    @ireduce(builtins.all, piter=list)
    @method_treelize()
    def all(self):
        pass

    @ireduce(builtins.any, piter=list)
    @method_treelize()
    def any(self):
        pass
