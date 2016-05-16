# -*- coding: utf-8 -*-
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PhloxAR.base import *


__all__ = [
    'HaarLikeFeature'
]


class HaarLikeFeature(object):
    """
    Create a single Haar feature and optionally set the regions that define
    the Haar feature and its name. The formal of the feature is
    The format is [[[TL],[BR],SIGN],[[TL],[BR],SIGN].....]
    Where TR and BL are the unit coordinates for the top right and bottom
    left coordinates.
    For example
    [[[0,0],[0.5,0.5],1],[[0.5.0],[1.0,1.0],-1]]
    Takes the right side of the image and subtracts from the left hand side
    of the image.
    """
    _name = None
    _regions = None

    def __init__(self, name=None, regions=None):
        self._name = name
        self._regions = regions

    @property
    def regions(self):
        return self._regions

    @regions.setter
    def regions(self, regions):
        """
        Set the list of regions. The regions are square coordinates on a unit
        sized image followed by the sign of a region.
        The format is [[[TL],[BR],SIGN],[[TL],[BR],SIGN].....]
        Where TR and BL are the unit coordinates for the top right and bottom
        left coordinates.
        For example
        [[[0,0],[0.5,0.5],1],[[0.5.0],[1.0,1.0],-1]]
        Takes the right side of the image and subtracts from the left hand side
        of the image.
        """
        self._regions = regions

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        """
        Set the name of this feature, the name must be unique.
        """
        self._name = name

    def apply(self, int_img):
        """
        This method takes in an integral image and applies the haar-cascade
        to the image, and returns the result.
        """
        w = int_img.shape[0] - 1
        h = int_img.shape[1] - 1
        accumulator = 0
        for i in range(len(self._regions)):
            # using the integral image
            # a = Lower Right Hand Corner
            # b = upper right hand corner
            # c = lower left hand corner
            # d = upper left hand corner
            # sum = a - b - c  + d
            # regions are in
            # (p,q,r,s,t) format
            p = self._regions[i][0]  # p = left (all are unit length)
            q = self._regions[i][1]  # q = top
            r = self._regions[i][2]  # r = right
            s = self._regions[i][3]  # s = bottom
            sign = self._regions[i][4]  # t = sign
            xa = int(w * r)
            ya = int(h * s)
            xb = int(w * r)
            yb = int(h * q)
            xc = int(w * p)
            yc = int(h * s)
            xd = int(w * p)
            yd = int(h * q)
            accumulator += sign * (int_img[xa, ya] - int_img[xb, yb] -
                                   int_img[xc, yc] + int_img[xd, yd])
        return accumulator

    def write2file(self, f):
        """
        Write the Haar cascade to a human readable f. f is an open f pointer.
        """
        f.write(self._name)
        f.write(" " + str(len(self._regions)) + "\n")
        for i in range(len(self._regions)):
            tmp = self._regions[i]
            for j in range(len(tmp)):
                f.write(str(tmp[j]) + ' ')
            f.write('\n')
        f.write('\n')
