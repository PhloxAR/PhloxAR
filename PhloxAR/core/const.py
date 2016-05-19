# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys


class _const:
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, key, value):
        if key in self.__dict__:
            raise self.ConstError('Const value {} already '
                                  'defined!'.format(key))
        if not key.isupper():
            raise self.ConstCaseError('Const name {} is not all '
                                      'uppercase!'.format(key))
        self.__dict__[key] = value

sys.modules[__name__] = _const()
