# -*- coding: utf-8 -*-
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals
from PhloxAR.base import *


class HaarCascade(object):
    """
    This class wraps HaarCascade files for the findHaarFeatures file.
    To use the class provide it with the path to a Haar cascade XML file and
    optionally a name.
    """
    _cascade = None
    _name = None
    _cache = {}
    _fhandle = None

    def __init__(self, fname=None, name=None):
        if name is None:
            self._name = fname
        else:
            self._name = name

        # First checks the path given by the user, if not then checks default folder
        if fname is not None:
            if os.path.exists(fname):
                self._fhandle = os.path.abspath(fname)
            else:
                self._fhandle = os.path.join(LAUNCH_PATH, 'Features',
                                             'HaarCascades', fname)
                if not os.path.exists(self._fhandle):
                    logger.warning("Could not find Haar Cascade file " + fname)
                    logger.warning("Try running the function "
                                   "img.list_haar_features() to see what is "
                                   "available")
                    return

            self._cascade = cv.Load(self._fhandle)

            if HaarCascade._cache.has_key(self._fhandle):
                self._cascade = HaarCascade._cache[self._fhandle]
                return
            HaarCascade._cache[self._fhandle] = self._cascade

    def load(self, fname=None, name=None):
        if name is None:
            self._name = fname
        else:
            self._name = name

        if fname is not None:
            if os.path.exists(fname):
                self._fhandle = os.path.abspath(fname)
            else:
                self._fhandle = os.path.join(LAUNCH_PATH, 'Features',
                                             'HaarCascades', fname)
                if not os.path.exists(self._fhandle):
                    logger.warning("Could not find Haar Cascade file " + fname)
                    logger.warning("Try running the function "
                                   "img.list_haar_features() to see what is "
                                   "available")
                    return None

            self._cascade = cv.Load(self._fhandle)

            if self._fhandle in HaarCascade._cache:
                self._cascade = HaarCascade._cache[fname]
                return
            HaarCascade._cache[self._fhandle] = self._cascade
        else:
            logger.warning("No file path mentioned.")

    @property
    def cascade(self):
        return self._cascade

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def file_handle(self):
        return self._fhandle
