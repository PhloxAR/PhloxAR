# -*- coding: utf-8 -*-

"""
Compatibility module for Python 2.7 and > 3.3
"""

from __future__ import unicode_literals

import sys
import time

try:
    import queue
except ImportError:
    import Queue as queue

PY2 = sys.version < '3'

clock = None

if PY2:
    unichr = unichr
    long = long
    fileopen = file
else:
    unichr = chr
    long = int
    fileopen = open

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

if PY2:
    from urllib2 import urlopen, build_opener
    from urllib2 import HTTPBasicAuthHandler, HTTPPasswordMgrWithDefaultRealm
else:
    from urllib import urlopen
    from urllib.request import build_opener, HTTPBasicAuthHandler
    from urllib.request import HTTPPasswordMgrWithDefaultRealm

if PY2:
    from UserDict import UserDict, MutableMapping
    from cStringIO import StringIO
    import SocketServer as socketserver
    import SimpleHTTPServer
else:
    from collections import UserDict, MutableMapping
    import http.server as SimpleHTTPServer
    import io.StringIO as StringIO
    import socketserver
