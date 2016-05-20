# -*- coding: utf-8 -*-

"""
Compatibility module for Python 2.7 and > 3.3
"""

import sys
import time

try:
    import queue
except ImportError:
    import Queue as queue

PY2 = sys.version_info[0] == 2

clock = None

string_types = None

text_type = None

if PY2:
    string_types = basestring
    text_type = unicode
else:
    string_types = text_type = str

if PY2:
    unichr = unichr
else:
    unichr = chr

if PY2:
    iterkeys = lambda d: d.iterkeys()
    itervalues = lambda d: d.itervalues()
    iteritems = lambda d: d.iteritems()
else:
    iterkeys = lambda d: iter(d.keys())
    itervalues = lambda d: iter(d.values)
    itervalues = lambda d: iter(d.items())

if PY2:
    if sys.platform in ('win32', 'cygwin'):
        clock = time.clock
    else:
        clock = time.time
else:
    clock = time.perf_counter