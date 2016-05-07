# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals
from ..base import *
from ..color import *


class Feature(object):
    """
    Abstract class which real features descend from.
    Each feature object has:
    a draw() method,
    an image property, referencing the originating Image object,
    x and y coordinates
    default functions for determining angle, area, mean color, etc.
    these functions assume the feature is 1px
    """
    _x = 0.00
    _y = 0.00
    _max_x = None
    _max_y = None
    _min_x = None
    _min_y = None
    _width = None
    _height = None
    _src_img_width = None
    _src_img_height = None

    # bounding box, top left then width, height
    _bbox = None
    # [max_x, min_x, max_y, min_y]
    _extents = None
    # tuples in order[(top_left), (top_right), (bot_left), (bot_right)]
    _points = None

    # parent image
    _image = None

    def __init__(self, img, at_x, at_y, points):
        self._x = at_x
        self._y = at_y
        self._image = img
        self._points = points
        self._update_extents(new_feature=True)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def reassign(self, img):
        """
        Reassign the image of this feature and return an updated copy of the
        feature.
        :param img: the new image to which to assign the feature
        :return: a deep copied Feature object.
        """
        feature = copy.deepcopy(self)
        if self._image.width != img.width or self._image.height != img.height:
            warnings.warn("Don't reassign image of different size.")

        feature._image = img

        return feature

    def corners(self):
        self._update_extents()
        return self._points

    def coordinates(self):
        """
        Returns the x, y position of the feature. This is usually the center
        coordinate.
        :return: a (x, y) tuple of the position of the feature.
        """
        return npy.array([self._x, self._y])

    def draw(self, color=Color.GREEN):
        """
        Draw the feature on the source image.
        :param color: a RGB tuple to render the image.
        :return: None
        """
        self._image[self._x, self._y] = color

    def show(self, color=Color.GREEN):
        """
        Automatically draw the features on the image and show it.
        :param color: a RGB tuple
        :return: None
        """
        self.draw(color)
        self._image.show()

    def distance_from(self, point=(-1, -1)):
        pass

    def mean_color(self):
        pass

    def color_distance(self, color=(0, 0, 0)):
        pass

    def angle(self):
        pass

    def length(self):
        pass

    def distance_to_nearest_edge(self):
        pass

    def on_image_edge(self, tolerance=1):
        pass

    def aspect_ratio(self):
        pass

    def area(self):
        pass

    def width(self):
        pass

    def height(self):
        pass

    def crop(self):
        pass

    def __repr__(self):
        pass

    def bbox(self):
        pass

    def extents(self):
        pass

    def _update_extents(self, new_feature=False):
        pass

    def min_x(self):
        pass

    def min_y(self):
        pass

    def max_x(self):
        pass

    def max_y(self):
        pass

    def top_left_corner(self):
        pass

    def bottom_right_corner(self):
        pass

    def bottomLeftCorner(self):
        pass

    def topRightCorner(self):
        pass

    def above(self, object):
        pass

    def below(self, object):
        pass

    def right(self, object):
        pass

    def left(self, object):
        pass

    def contains(self, other):
        pass

    def overlaps(self, other):
        pass

    def is_contained_within(self, other):
        pass

    def _point_inside_polygon(self, point, polygon):
        pass

    def bcircle(self):
        pass
