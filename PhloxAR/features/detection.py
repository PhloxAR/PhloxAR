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
import numpy as npy
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
    def __init__(self, img, x, y):
        points = [(x - 1, y - 1), (x - 1, y + 1), (x + 1, y + 1), (x + 1, y - 1)]
        super(Corner, self).__init__(img, x, y, points)

    def draw(self, color=Color.RED, width=1):
        """
        Draw a small circle around the corner. Color tuple is single parameter,
        default is Color.RED.

        :param color: an RGB color triplet
        :param width: if width is less than zero the draw the feature filled in,
                       otherwise draw the contour using specified width.
        :return: None
        """
        self._image.draw_circle(self._x, self._y, 4, color, width)


class Line(Feature):
    """
    The line feature is returned by the find_lines function, also can be
    initialized with any two points.

    >>> line = Line(Image, (point1, point2))

    where point1 and point2 are (x, y) coordinate tuples

    >> line.points

    Returns a tuple of the two points.
    """
    # TODO: calculate the endpoints of the line

    def __init__(self, img, line):
        self._image = img
        self._vector = None
        self._y_intercept = None
        self._end_pts = copy(line)

        if self._end_pts[1][0] - self._end_pts[0][0] == 0:
            self._slope = float('inf')
        else:
            self._slope = (self._end_pts[1][1] - self._end_pts[0][1]) / (
                self._end_pts[1][0] - self._end_pts[0][0]
            )
        # coordinate of the line object is the midpoint
        at_x = (line[0][0] + line[1][0]) / 2
        at_y = (line[0][1] + line[1][1]) / 2
        xmin = int(npy.min([line[0][0], line[1][0]]))
        xmax = int(npy.max([line[0][0], line[1][0]]))
        ymin = int(npy.min([line[0][1], line[1][1]]))
        ymax = int(npy.max([line[0][1], line[1][1]]))
        points = [(xmin, ymin), (xmin, ymax), (xmax, ymax), [xmax, ymin]]
        super(Line, self).__init__(img, at_x, at_y, points)

    def draw(self, color=Color.BLUE, width=1):
        """
        Draw a the line, default color is Color.BLUE
        :param color: a RGB color triplet
        :param width: draw the line using specified width
        :return: None - modify the source image drawing layer
        """
        self._image.draw_line(self._end_pts[0], self._end_pts[1], color, width)

    @property
    def length(self):
        """
        Returns the length of the line.

        :return: a floating point length value

        :Example:
        >>> img = Image('lena.jpg')
        >>> lines = img.find_lines()
        >>> for l in lines:
        >>>     if l.length > 100:
        >>>         print("Oh my!, what a big line you have!")
        """
        return float(spsd.euclidean(self._end_pts[0], self._end_pts[1]))

    def crop(self):
        """
        Crops the source image to the location of the feature and returns
        a new Image.

        :return: an Image that is cropped to the feature position and size

        :Example:
        >>> img = Image('edge_test2.png')
        >>> l = img.find_lines()
        >>> line = l[0].crop()
        """
        tl = self.top_left_corner()
        return self._image.crop(tl[0], tl[1], self._width, self._height)

    def mean_color(self):
        """
        Returns the mean color of pixels under the line.
        Note that when the line falls "between" pixels, each pixels color
        contributes to the weighted average.

        :return: a RGB triplet corresponding to the mean color of the feature

        :Example:
        >>> img = Image('lena')
        >>> l = img.find_lines()
        >>> c = l[0].mean_color()
        """
        pt1, pt2 = self._end_pts
        # we are going to walk the line, and take the mean color from all the px
        # points -- there's probably a much more optimal way to do this
        xmax, xmin, ymax, ymin = self.extents()

        dx = xmax - xmin
        dy = ymax - ymin
        # orient the line so it is going in the positive direction
        # if it's a straight line, we can just get mean color on the slice
        if dx == 0.0:
            return self._image[pt1[0]:pt1[0] + 1, ymin:ymax].mean_color()
        if dy == 0.0:
            return self._image[xmin:xmax, pt1[1]:pt[1] + 1].mean_color()

        error = 0.0
        derr = dy / dx  # this is how much 'error' will increase in every step
        px = []
        weights = []
        if derr < 1:
            y = ymin
            # iterate over x
            for x in range(xmin, xmax):
                # this is the pixel we would draw on, check the color at that px
                # weight is reduced from 1.0 by the abs amount of error
                px.append(self._image[x, y])
                weights.append(1.0 - abs(error))

                # if we have error in either direction, we're going to use
                # the px above or below
                if error > 0:
                    px.append(self._image[x, y+1])
                    weights.append(error)

                if error < 0:
                    px.append(self._image[x, y-1])
                    weights.append(abs(error))

                error = error + derr

                if error >= 0.5:
                    y += 1
                    error = error - 1.0
        else:
            # this is a 'steep' line, so we iterate over x
            # copy and paste. ugh, sorry.
            x = xmin
            for y in range(ymin, ymax):
                # this is the pixel we would draw on, check the color at that px
                # weight is reduced from 1.0 by the abs amount of error
                px.append(self._image[x, y])
                weights.append(1.0 - abs(error))

                if error > 0:
                    px.append(self._image[x + 1, y])
                    weights.append(error)

                if error < 0:
                    px.append(self._image[x - 1, y])
                    weights.append(abs(error))

                error += 1.0 / derr  # we use the reciprocal of error
                if error >= 0.5:
                    x += 1
                    error -= 1.0

        # once we have iterated over every pixel in the line, we avg the weights
        clr_arr = npy.array(px)
        weight_arr = npy.array(weights)

        # multiply each color tuple by its weight
        weighted_clrs = npy.transpose(npy.transpose(clr_arr) * weight_arr)

        tmp = sum(weighted_clrs / sum(weight_arr))
        return float(tmp[0]), float(tmp[1]), float(tmp[2])

    def find_intersection(self, line):
        """"""
        pass


class Barcode(Feature):
    pass


class HaarFeature(Feature):
    pass


class Chessboard(Feature):
    pass


class TemplateMatch(Feature):
    pass


class Circle(Feature):
    pass


class KeyPoint(Feature):
    pass


class Motion(Feature):
    pass


class KeyPointMatch(Feature):
    pass


class ShapeContextDescriptor(Feature):
    pass


class ROI(Feature):
    pass
