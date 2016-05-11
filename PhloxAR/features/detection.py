# -*- coding: utf-8 -*-
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

'''
Image features detection.
All angles shalt be described in degrees with zero pointing east in the
plane of the image with all positive rotations going counter-clockwise.
Therefore a rotation from the x-axis to to the y-axis is positive and follows
the right hand rule.
'''

from PhloxAR.base import *
from PhloxAR.image import *
from PhloxAR.color import *
from PhloxAR.features.feature import Feature, FeatureSet


class Corner(Feature):
    """
    The Corner feature is a point returned by the find_corners function.
    Corners are used in machine vision as a very computationally efficient
    way to find unique features in an image. These corners can be used in
    conjunction with many other algorithms.
    """
    def __init__(self, i, x, y):
        points = [(x - 1, y - 1), (x - 1, y + 1), (x + 1, y + 1), (x + 1, y - 1)]
        super(Corner, self).__init__(i, x, y, points)

    def draw(self, color=Color.RED, width=1):
        pass
