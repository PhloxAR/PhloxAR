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
    is_background = True
    data = {}
    bits = 1

    def __init__(self, data=None):
        pass