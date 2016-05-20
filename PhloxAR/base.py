#!/usr/bin/env python
# -*- coding: utf-8 -*-

##
#
# Copyright (c) 2012, Sight Machine
# All rights reserved.
#
# Copyright 2016(c) Matthias Y. Chen
# <matthiasychen@gmail.com/matthias_cy@outlook.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

from __future__ import division, print_function
from __future__ import unicode_literals, absolute_import

import os
import sys

if sys.version > '3':
    PY3 = True
else:
    PY3 = False


if PY3:
    from collections import UserDict, MutableMapping
    from urllib.request import  urlopen
    import socketserver as SocketServer
    import http.server as SimpleHTTPServer
    import io.StringIO as StringIO

else:
    from UserDict import UserDict, MutableMapping
    from urllib2 import urlopen
    import SocketServer
    import SimpleHTTPServer
    from cStringIO import StringIO

try:
    import cv2
except ImportError:
    raise ImportError("Cannot load OpenCV library which is required.")
else:
    if cv2.__version__ < '3':
        raise ImportError("Your OpenCV library version is lower than 3.")

try:
    from PIL import Image as PILImage
    from PIL import ImageFont as PILImageFont
    from PIL import GifImagePlugin as PILGifImagePlugin
    getheader = PILGifImagePlugin.getheader
    getdata = PILGifImagePlugin.getdata
except ImportError:
    raise ImportError("Cannot load PIL.")

# optional libraries

# binary code
ZXING_ENABLED = True
try:
    import zxing
except ImportError:
    ZXING_ENABLED = False

# recognition
OCR_ENABLED = True
try:
    import tesseract
except ImportError:
    OCR_ENABLED = False

PYSCREENSHOT_ENABLED = True
try:
    import pyscreenshot
except ImportError:
    PYSCREENSHOT_ENABLED = False

ORANGE_ENABLED = True
try:
    try:
        import orange
    except ImportError:
        import Orange; import orange

    import orngTest  # for cross validation
    import orngStat
    import orngEnsemble  # for baggin / boosting

except ImportError:
    ORANGE_ENABLED = False

VIMBA_ENABLED = True
try:
    import pymba
except ImportError:
    # TODO: log an error the pymba is not installed
    VIMBA_ENABLED = False
except Exception:
    # TODO: log an error that AVT Vimba DLL is not installed properly
    VIMBA_ENABLED = False

    consoleHandler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    consoleHandler.setFormatter(formatter)
    logger = logging.getLogger('Main Logger')
    logger.addHandler(consoleHandler)


class InitOptionsHandler(object):
    """
    This handler is supposed to store global variables. For now, its only
    value defines if phlox is being run on an ipython notebook.
    """

    def __init__(self):
        self.on_notebook = False
        self.headless = False

    def enable_notebook(self):
        self.on_notebook = True

    def set_headless(self):
        # set SDL to use the dummy NULL video driver
        # so it doesn't need ad windoing system
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        self.headless = True

init_options_handler = InitOptionsHandler()

try:
    import pygame as sdl2
except ImportError:
    init_options_handler.set_headless()


# couple quick typecheck helper functions
def isnum(n):
    """
    Determines if it is a number or not.
    Returns: Boolean
    """
    return type(n) in (IntType, LongType, FloatType)


def istuple(n):
    """
    Determines if it is a tuple or not.
    Returns: Boolean
    """
    return type(n) == tuple


def rev_tuple(n):
    """
    Reverses a tuple.
    Returns: Tuple
    """
    return tuple(reversed(n))


# TODO: to remove.
def find(k, seq):
    """
    Search for item in a list.
    Returns: Boolean
    """
    # for item in seq:
    #     if k == item:
    #         return True
    if k in seq:
        return True
    else:
        return False


def test():
    """
    Run builtin unittests.
    """
    print('')


def download(url):
    """
    This function takes in a URL for a zip file, extracts it and
    returns the temporary path it was extracted to.
    """
    if url is None:
        logger.warning("Please provide URL.")
        return None

    tmpdir = tempfile.mkdtemp()
    filename = os.path.basename(url)
    path = tmpdir + '/' + filename
    zdata = urlopen(url)

    print("Saving file to disk please wait...")
    with open(path, 'wb') as local:
        local.write(zdata.read())

    zfile = zipfile.ZipFile(path)
    print("Extracting zip file.")

    try:
        zfile.extractall(tmpdir)
    except:
        logger.warning("Couldn't extract zip file.")
        return None

    return tmpdir


def int2byte(i):
    """
    Integer to two bytes.
    """
    i1 = i % 256
    i2 = int(i/256)
    return chr(i1) + chr(i2)


# deprecated
def npArray2cvMat(mat, dtype=cv.CV_32FC1):
    """
    This function is a utility for converting numpy arrays to
    the cv.cvMat format.
    Returns: cvMatrix
    """
    if type(mat) == np.ndarray:
        size = len(mat.shape)
        tmp_mat = None
        if dtype in (cv.CV_32FC1, cv.cv32FC2, cv.CV_32FC3, cv.CV_32FC4):
            tmp_mat = np.array(mat, dtype='float32')
        elif dtype in (cv.CV_8UC1, cv.CV_8UC2, cv.CV_8UC3, cv.CV_8UC3):
            tmp_mat = np.array(mat, dtype='uint8')
        else:
            logger.warning("Input matrix type is not supported")
            return None

        if size == 1:  # this needs to be changed so we can do row/col vectors
            retVal = cv.CreateMat(mat.shape[0], 1, dtype)
            cv.SetData(retVal, tmp_mat.tostring(),
                       tmp_mat.dtype.itemsize * tmp_mat.shape[0])
        elif size == 2:
            retVal = cv.CreateMat(tmp_mat.shape[0], tmp_mat.shape[1], dtype)
            cv.SetData(retVal, tmp_mat.tostring(),
                       tmp_mat.dtype.itemsize * tmp_mat.shape[1])
        else:
            logger.warning("Input matrix type is not supported")
            return None

        return retVal
    else:
        logger.warning("Input matrix type is not supported")

try:
    import IPython
    ipy_ver = IPython.__version__
except ImportError:
    ipy_ver = None


# This is used with sys.excepthook to log all uncaught exceptions.
# By default, error messages ARE print to stderr.
def exception_handler(exc_type, exc_val, traceback):
    logger.error("", exc_info=(exc_type, exc_val, traceback))

    # exc_val has the most important info about the error.
    # It'd be possible to display only that and hide all the (unfriendly) rest.

sys.excepthook = exception_handler


def ipy_exc_handler(shell, exc_type, exc_val, traceback, tb_offset=0):
    logger.error("", exc_type, exc_val, traceback)


# The two following functions are used internally.
def init_logging(log_level):
    logger.setLevel(log_level)


def read_logging_level(log_level):
    levels_dict = {
        1: logging.DEBUG, 'debug': logging.DEBUG,
        2: logging.INFO, 'info': logging.INFO,
        3: logging.WARNING, 'warning': logging.WARING,
        4: logging.ERROR, 'error': logging.ERROR,
        5: logging.CRITICAL, 'critical': logging.CRITICAL
    }

    if isinstance(log_level, str):
        log_level = log_level.lower()

    if log_level in levels_dict:
        return levels_dict[log_level]
    else:
        print("The logging level given is not valid.")
        return None


def get_logging_level(log_level):
    """
    Prints the current logging level of the main logger.
    """
    levels_dict = {
        10: "DEBUG",
        20: "INFO",
        30: "WARNING",
        40: "ERROR",
        50: "CRITICAL"
    }

    print("The current logging level is: %s" %
          levels_dict[logger.getEffectiveLevel()])


def set_logging(log_level, file_name=None):
    """
    Sets the threshold for the logging system and, if desired,
    directs the messages to a log file.
    Level options:

    'DEBUG' or 1
    'INFO' or 2
    'WARNING' or 3
    'ERROR' or 4
    'CRITICAL' or 5

    If the user is on the interactive shell and wants to log to file,
    a custom excepthook is set. By default, if logging to file is not
    enabled, the way errors are displayed on the interactive shell is
    not changed.
    """
    if file_name and ipy_ver:
        try:
            if ipy_ver.startswith('0.10'):
                __IPYTHON__.set_custom_exc((Exception,), ipy_exc_handler)
            else:
                ip = get_ipython()
                ip.set_custom_exc((Exception,), ipy_exc_handler)
        except NameError:  # In case the interactive shell not used.
            sys.exc_clear()

    level = read_logging_level(log_level)

    if level and file_name:
        fileHandler = logging.FileHandler(filename=file_name)
        fileHandler.setLevel(level)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
        logger.removeHandler(consoleHandler)  # console logging is disabled.
        print("Now logging to %s with level %s." % (file_name, log_level))
    elif level:
        print("Now logging with level %s." % log_level)

    logger.setLevel(level)


def system():
    """
    Outputs various informations related to system and library.

    Main purpose:
    - While submitting a bug, report the output of this function
    - Check the current version and later upgrading the library
      based on the output.
    """
    try:
        import platform
        print("System: %s." % platform.system())
        print("OS version: %s." % platform.version())
        print("Python version: %s." % platform.python_version())

        try:
            from cv2 import __version__
            print("OpenCV version: %s." % __version__)
        except ImportError:
            print("OpenCV not installed.")

        if PIL_ENABLED:
            print("PIL version: %s." % pil.VERSION)
        else:
            print("PIL module not installed.")

        if ORANGE_ENABLED:
            print("Orange version: %s." % orange.verison)
        else:
            print("Orange module not installed.")

        try:
            import pygame as sdl2
            print("PyGame version: %s." % sdl2.__version__)
        except ImportError:
            print("PyGame module not installed")

        try:
            import pickle
            print("Pickle version: %s." % pickle.__version__)
        except ImportError:
            print("Pickle module not installed.")
    except ImportError:
        print("You need to install Platform to use this function.")


class lazy_property(object):
    """
    Used for lazy evaluation of an object attribute.
    """
    def __init__(self, func):
        self._func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __get__(self, obj, cls=None):
        if obj is None:
            return None
        res = obj.__dict__[self.__name__] = self._func(obj)
        return res

# supported image formats regular expression ignoring case
IMAGE_FORMATS = ('*.[bB][mM][Pp]', '*.[Gg][Ii][Ff]', '*.[Jj][Pp][Gg]',
                 '*.[jJ][pP][eE]', '*.[jJ][Pp][Ee][Gg]', '*.[pP][nN][gG]',
                 '*.[pP][bB][mM]', '*.[pP][gG][mM]', '*.[pP][pP][mM]',
                 '*.[tT][iI][fF]', '*.[tT][iI][fF][fF]', '*.[wW][eE][bB][pP]')

# maximum image size
MAX_DIMS = 2*6000  # about twice the size of a full 35mm images
LAUNCH_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
