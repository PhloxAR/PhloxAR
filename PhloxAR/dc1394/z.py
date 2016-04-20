from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

try:
    # Python 2
    from UserDict import UserDict
    from UserDict import DictMixin as MutableMapping
except ImportError:
    # Python 3
    from collections import UserDict
    from collections import MutableMapping

# using urllib & urllib2
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen
