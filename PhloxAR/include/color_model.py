# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PhloxAR.base import *
from PhloxAR.image import *


class ColorModel(object):
    """
    The ColorModel is used to model the color of foreground and background
    objects by using a training set of images.

    You can crate the color model with any number of 'training' images, or
    add images to the model with add() and remove(). The for your data
    images, you can useThresholdImage() to return a segmented picture.
    """
    # TODO: discretize the color space into smaller intervals
    # TODO: work in HSV space
    _is_bkg = True
    _data = {}
    _bits = 1

    def __init__(self, data=None, is_bkg=True):
        self._is_bkg = is_bkg
        self._data = data
        self._bits = 1

        if data:
            try:
                [self.add(d) for d in data]
            except TypeError:
                self.add(data)

    def _make_canonical(self, data):
        pass

    def reset(self):
        pass

    def add(self, data):
        pass

    def threshold(self, image):
        pass

    def contains(self, c):
        pass

    def set_is_bkg(self):
        pass

    def load(self, filename):
        pass

    def save(self, filename):
        pass