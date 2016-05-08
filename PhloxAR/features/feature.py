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
    _aspect_ratio = None
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
        self._update_extents()
        return self._x

    @property
    def y(self):
        self._update_extents()
        return self._y

    @property
    def width(self):
        self._update_extents()
        return self._width

    @property
    def height(self):
        self._update_extents()
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
        """
        Given a point (default to center of the image), return the euclidean
        :param point: (x, y) tuple on the image to measure distance from
        :return: the distance as a floating point value in pixels.
        """
        if point[0] == -1 or point[1] == -1:
            point = npy.array(self._image.size()) / 2

        return spsd.euclidean(point, [self._x, self._y])

    def mean_color(self):
        """
        Return the average color within the feature as a tuple.
        :return: an RGB color tuple
        """
        return self._image[self._x, self._y]

    def color_distance(self, color=(0, 0, 0)):
        """
        Return the euclidean color distance of the color tuple at x, y from
        a given color (default black).
        :param color: a RGB tuple to calculate from which to calculate the
                       color distance.
        :return: a floating point color distance value.
        """
        return spsd.euclidean(npy.array(color), npy.array(self.mean_color()))

    def angle(self):
        """
        Return the angle (theta) in degrees of the feature. The default is 0
        (horizontal). Not valid for all features.
        :return:
        """
        return 0

    def length(self):
        """
        Returns the longest dimension of the feature (i.e., max(width, height))
        :return: a floating point length value.
        """
        return float(npy.max([self.width, self.height]))

    def distance_to_nearest_edge(self):
        """
        Returns the distance, in pixels, from the nearest image edge.
        :return: integer distance to the nearest edge.
        """
        w = self._image.width
        h = self._image.height

        return npy.min([self._min_x, self._min_y,
                        w - self._max_x, h - self._max_y])

    def on_image_edge(self, tolerance=1):
        """
        Returns True if the feature is less than 'tolerance' pixels away from
        the nearest edge.
        :param tolerance: the distance in pixels at which a feature qualifies
                           as being on the image edge.
        :return: True if the feature is on the edge, False otherwise.
        """
        return self.distance_to_nearest_edge() <= tolerance

    def aspect_ratio(self):
        """
        Return the aspect ratio of the feature, which for our purposes is
        max(width, height) / min(width, height)
        :return: a single floating point value of the aspect ratio.
        """
        self._update_extents()
        return self._aspect_ratio

    def area(self):
        """
        Returns the area (number of pixels) covered by the feature
        :return: an integer area of the feature
        """
        return self.width * self.height

    @property
    def width(self):
        """
        Returns the width of the feature.
        :return: integer value for the feature's width
        """
        self._update_extents()
        return self._width

    @property
    def height(self):
        """
        Returns the height of the feature.
        :return: integer value for the feature's height
        """
        self._update_extents()
        return self._height

    def crop(self):
        """
        Crops the source image to the location of the feature and
        returns a new Image.
        :return: an Image that is cropped to the feature position and size.
        """
        return self._image.crop(self._x, self._y,
                                self.width, self.height, centered=True)

    def __repr__(self):
        return "{}.{} at ({}, {})".format(self.__class__.__module__,
                                          self.__class__.__name__,
                                          self._x, self._y)

    def bbox(self):
        """
        Returns a rectangle which bounds the blob.
        :return: a list of [x, y, w, h] where (x, y) are the top left point
                  of the rectangle and w, h are its width and height.
        """
        self._update_extents()
        return self._bbox

    def extents(self):
        """
        Returns the maximum and minimum x and y values for the feature and
        returns them as a tuple
        :return: a tuple of the extents of the feature.
                  Order is(max_x, max_y, min_x, min_y)
        """
        self._update_extents()
        return self._extents

    def _update_extents(self, new_feature=False):
        max_x = self._max_x
        max_y = self._max_y
        min_x = self._min_y
        min_y = self._min_y
        width = self._width
        height = self._height
        extents = self._extents
        bbox = self._bbox

        if new_feature or None in [max_x, min_x, max_y, min_y,
                                   width, height, extents, bbox]:

            max_x = max_y = float("-infinity")
            min_x = min_y = float("infinity")

            for p in self._points:
                if p[0] > max_x:
                    max_x = p[0]
                if p[0] < min_x:
                    min_x = p[0]
                if p[1] > max_y:
                    max_y = p[1]
                if p[1] < min_y:
                    min_y = p[1]

            width = max_x - min_x
            height = max_y - min_y

            if width <= 0:
                width = 1

            if height <= 0:
                height = 1

            self._bbox = [min_x, min_y, width, height]
            self._extents = [max_x, min_x, max_y, min_y]

            if width > height:
                self._aspect_ratio = float(width / height)
            else:
                self._aspect_ratio = float(height / width)

            self._max_x = max_x
            self._min_x = min_x
            self._max_y = max_y
            self._min_y = min_y
            self._width = width
            self._height = height

    def min_x(self):
        """
        Returns the minimum x value of the bounding box of the feature.
        :return: an integer value of the minimum x value of the feature.
        """
        self._update_extents()
        return self._min_x

    def min_y(self):
        """
        Returns the minimum y value of the bounding box of the feature.
        :return: an integer value of the minimum y value of the feature.
        """
        self._update_extents()
        return self._min_y

    def max_x(self):
        """
        Returns the maximum x value of the bounding box of the feature.
        :return: an integer value of the maximum x value of the feature.
        """
        self._update_extents()
        return self._max_x

    def max_y(self):
        """
        Returns the maximum y value of the bounding box of the feature.
        :return: an integer value of the maximum y value of the feature.
        """
        self._update_extents()
        return self._max_y

    def top_left_corner(self):
        """
        Returns the top left corner of the bounding box of the blob as tuple
        :return: a tuple of the top left corner
        """
        self._update_extents()
        return self._min_x, self._min_y

    def bottom_right_corner(self):
        """
        Returns the bottom right corner of the bounding box of the blob as tuple
        :return: a tuple of the bottom right corner
        """
        self._update_extents()
        return self._max_x, self._max_y

    def bottom_left_corner(self):
        """
        Returns the bottom left corner of the bounding box of the blob as tuple
        :return: a tuple of the bottom left corner
        """
        self._update_extents()
        return self._min_x, self._max_y

    def top_right_corner(self):
        """
        Returns the top right corner of the bounding box of the blob as tuple
        :return: a tuple of the top right corner
        """
        self._update_extents()
        return self._max_x, self._min_y

    def above(self, obj):
        """
        Return True if the feature is above the object, where object can be
        a bounding box, bounding circle, a list of tuples in a closed polygon,
        or any other features.
        :param obj: bounding box - (x, y, w, h) where (x, y) is the top left
                    bounding circle - (x, y, r)
                    a list of (x, y) tuples defining a closed polygon
                    any two dimensional feature (e.g. blobs, circle ...
        :return: Bool, True if the feature is above the object, False otherwise.
        """
        if isinstance(obj, Feature):
            return self.max_y() < obj.min_y()
        elif isinstance(obj, tuple) or isinstance(obj, npy.ndarray):
            return self.max_y() < obj[1]
        elif isinstance(obj, float) or isinstance(obj, int):
            return self.max_y() < obj
        else:
            logger.warning("Did not recognize.")
            return None

    def below(self, obj):
        """
        Return True if the feature is below the object, where object can be
        a bounding box, bounding circle, a list of tuples in a closed polygon,
        or any other features.
        :param obj: bounding box - (x, y, w, h) where (x, y) is the top left
                    bounding circle - (x, y, r)
                    a list of (x, y) tuples defining a closed polygon
                    any two dimensional feature (e.g. blobs, circle ...
        :return: Bool, True if the feature is below the object, False otherwise.
        """

        if isinstance(obj, Feature):
            return self.min_y() > obj.max_y()
        elif isinstance(obj, tuple) or isinstance(obj, npy.ndarray):
            return self.min_y() > obj[1]
        elif isinstance(obj, float) or isinstance(obj, int):
            return self.min_y() > obj
        else:
            logger.warning("Did not recognize.")
            return None

    def right(self, obj):
        """
        Return True if the feature is to the right object, where object can be
        a bounding box, bounding circle, a list of tuples in a closed polygon,
        or any other features.
        :param obj: bounding box - (x, y, w, h) where (x, y) is the top left
                    bounding circle - (x, y, r)
                    a list of (x, y) tuples defining a closed polygon
                    any two dimensional feature (e.g. blobs, circle ...
        :return: Bool, True if the feature is to the right object, False otherwise.
        """

        if isinstance(obj, Feature):
            return self.min_x() > obj.max_x()
        elif isinstance(obj, tuple) or isinstance(obj, npy.ndarray):
            return self.min_x() > obj[0]
        elif isinstance(obj, float) or isinstance(obj, int):
            return self.min_x() > obj
        else:
            logger.warning("Did not recognize.")
            return None

    def left(self, obj):
        """
        Return True if the feature is to the left object, where object can be
        a bounding box, bounding circle, a list of tuples in a closed polygon,
        or any other features.
        :param obj: bounding box - (x, y, w, h) where (x, y) is the top left
                     bounding circle - (x, y, r)
                     a list of (x, y) tuples defining a closed polygon
                     any two dimensional feature (e.g. blobs, circle ...
        :return: Bool, True if the feature is to the left object, False otherwise.
        """

        if isinstance(obj, Feature):
            return self.max_x() < obj.min_x()
        elif isinstance(obj, tuple) or isinstance(obj, npy.ndarray):
            return self.max_x() < obj[0]
        elif isinstance(obj, float) or isinstance(obj, int):
            return self.max_x() < obj
        else:
            logger.warning("Did not recognize.")
            return None

    def contains(self, other):
        pass

    def overlaps(self, other):
        pass

    def is_contained_within(self, other):
        pass

    def _point_inside_polygon(self, point, polygon):
        pass

    def bcir(self):
        """
        Calculates the minimum bounding circle of the blob in the image as
        a (x, y, r) tuple.
        :return: a (x, y, r) tuple where (x, y) is the center of the circle
                  and r is the radius
        """
        try:
            import cv2
        except:
            logger.warning("Unable to import cv2")
            return None

        contour = self.contour()

        points = []

        for pair in contour:
            points.append([pair[0], pair[1]])

        points = npy.array(points)

        cen, rad = cv2.minEnclosingCircle(points)

        return cen[0], cen[1], rad

    def contour(self):
        return []
