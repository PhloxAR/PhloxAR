# -*- coding:utf-8 -*-
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PhloxAR.base import math
from PhloxAR.base import sss
from PhloxAR.base import *
from PhloxAR.features.feature import Feature
from PhloxAR.color import Color
from PhloxAR.image import Image


class Blob(Feature):
    """
    Blob is a typical cluster of pixels that form a feature or unique
    shape that allows it to be distinguished from the reset of the image.
    Blobs typically are computed very quickly so they are used often to
    find various items in a picture based on properties. Typically these
    things like color, shape, size, etc. Since blobs are computed quickly
    they are typically used to narrow down search regions in an image,
    where you quickly find a blob and then that blobs region is used for
    more computational intensive type image processing.

    :Example:
    >>> img = Image('lena')
    >>> blobs = img.find_blobs()
    >>> blobs[-1].draw()
    >>> img.show()

    :Also:
    :py:meth: `find_blobs`
    :py:class: `BlobMaker`
    :py:meth: `find_blobs_from_mask`
    """
    seq = ''  # the cvseq object that defines this blob
    _contour = []  # the blob's outer perimeter as a set of (x, y) tuples
    _convex_hull = []  # the convex hull contour as a set of (x, y) tuples
    _min_rect = []  # the smallest box rotated to fit the blob
    _hu = []  # the seven Hu moments
    _perimeter = 0  # hte length of the perimeter in pixels
    _area = 0  # the area in pixels
    m00 = m01 = m02 = 0
    m10 = m11 = m12 = 0
    m20 = m21 = 0
    _contour_appx = None
    _label = ''  # a user label
    _label_color = []  # the color to draw the label
    _avg_color = []  # the average color of the blob's area.
    _hole_contour = []  # list of hole contours
    pickle_skip_properties = {'_img', '_hull_img', '_mask', '_hull_mask'}

    _img = ''  # Image()# the segmented image of the blob
    _hull_img = ''  # Image() the image from the hull.
    _mask = ''  # Image()# A mask of the blob area
    # Image()# A mask of the hull area ... we may want to use this for
    # the image mask
    _hull_mask = ''

    def __init__(self):
        self._sc
