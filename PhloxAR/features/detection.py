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
            return self._image[xmin:xmax, pt1[1]:pt1[1] + 1].mean_color()

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
                    error -= 1.0
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
        """
        Returns the intersection point of two lines.

        :param line: the other line to compute intersection
        :return: a point tuple

        :Example:
        >>> img = Image('lena')
        >>> l = img.find_lines()
        >>> c = l[0].find_intersection(l[1])
        """
        # TODO: NEEDS TO RETURN A TUPLE OF FLOATS
        if self._slope == float('inf'):
            x = self._end_pts[0][0]
            y = line.slope * (x - line._end_pts[1][0]) + line._end_pts[1][1]
            return x, y

        if line.slope == float("inf"):
            x = line._end_pts[0][0]
            y = self.slope * (x - self._end_pts[1][0]) + self._end_pts[1][1]
            return x, y

        m1 = self._slope
        x12, y12 = self._end_pts[1]
        m2 = line.slope
        x22, y22 = line._end_pts[1]

        x = (m1 * x12 - m2 * x22 + y22 - y12) / float(m1 - m2)
        y = (m1 * m2 * (x12 - x22) - m2 * y12 + m1 * y22) / float(m1 - m2)

        return x, y

    def is_parallel(self, line):
        """
        Checks whether two lines are parallel or not.

        :param line: the other line to be compared
        :return: Bool

         :Example:
         >>> img = Image('lena')
         >>> l = img.find_lines()
         >>> c = l[0].is_parallel(l[1])
        """
        if self._slope == line.slope:
            return True

        return False

    def is_perpendicular(self, line):
        """
        Checks whether two lines are perpendicular or not.

        :param line: the other line to be compared
        :return: Bool.

         :Example:
         >>> img = Image('lena')
         >>> l = img.find_lines()
         >>> c = l[0].is_perpendicular(l[1])
        """
        if self._slope == float('inf'):
            if line.slope == 0:
                return True
            return False

        if line.slope == float('inf'):
            if self.slope == 0:
                return True
            return False

        if self._slope * line.slope == -1:
            return True

        return False

    def image_intersections(self, img):
        """
        Returns a set of pixels where the line intersects with the binary image.

        :param img: binary image
        :return: a list of points

        :Example:
        >>> img = Image('lena')
        >>> l = img.find_lines()
        >>> c = l[0].image_intersections(img.binarize())
        """
        pixels = []
        if self._slope == float('inf'):
            for y in range(self._end_pts[0][1], self._end_pts[1][1] + 1):
                pixels.append((self._end_pts[0][0], y))
        else:
            for x in range(self._end_pts[0][0], self._end_pts[1][0] + 1):
                pixels.append((x, int(self._end_pts[1][1] +
                                      self._slope * (x - self._end_pts[1][0]))))
            for y in range(self._end_pts[0][1], self._end_pts[1][1] + 1):
                pixels.append((int(((y - self._end_pts[1][1]) / self._slope) +
                                   self._end_pts[1][0]), y))

        pixels = list(set(pixels))
        matched_pixels = []
        for pixel in pixels:
            if img[pixel[0], pixel[1]] == (255.0, 255.0, 255.0):
                matched_pixels.append(pixel)
        matched_pixels.sort()

        return matched_pixels

    def angle(self):
        """
        Angle of the line, from the left most point to right most point.
        Returns angel (theta), with 0 = horizontal,
        -pi/2 = vertical positive slope, pi/2 = vertical negative slope.

        :return: an angle value in degrees.

        :Example:
        >>> img = Image('lena')
        >>> ls = img.find_lines()
        >>> for l in ls:
        >>>     if l.angle() == 0:
        >>>         print("Horizontal!")
        """
        # first find left most point
        a = 0
        b = 1
        if self._end_pts[a][0] > self._end_pts[b][0]:
            b = 0
            a = 1

        dx = self._end_pts[b][0] - self._end_pts[a][0]
        dy = self._end_pts[b][1] - self._end_pts[a][1]

        # internal standard if degrees
        return float(360.0 * (atan2(dy, dx) / (2 * npy.pi)))

    def crop2image_edges(self):
        pass

    def get_vector(self):
        pass

    def dot(self, other):
        pass

    def cross(self, other):
        pass

    def get_y_intercept(self):
        pass

    def extend2image_edges(self):
        pass

    @property
    def slope(self):
        return self._slope


class Barcode(Feature):
    def __init__(self, img, zbsymbol):
        pass

    def __repr__(self):
        pass

    def draw(self, color=Color.GREEN, width=1):
        pass

    def length(self):
        pass

    def area(self):
        pass


class HaarFeature(Feature):
    def __init__(self, img, haar_obj, haar_classifier=None, cv2flag=True):
        pass

    def draw(self, color=Color.GREEN):
        pass

    def __getstate__(self):
        pass

    def mean_color(self):
        pass

    def area(self):
        pass


class Chessboard(Feature):
    def __init__(self, img, dim, subpixelscorners):
        pass

    def draw(self, no_need_color=None):
        pass

    def area(self):
        pass


class TemplateMatch(Feature):
    def __init__(self, img, template, location, quality):
        pass

    def _template_overlaps(self, other):
        pass

    def consume(self, other):
        pass

    def rescale(self, w, h):
        pass

    def crop(self):
        pass

    def draw(self, color=Color.GREEN, width=1):
        pass


class Circle(Feature):
    def __init__(self):
        pass

    def draw(self, color=Color.GREEN, width=1):
        pass

    def show(self, color=Color.GREEN):
        pass

    def distance_from(self, point=(-1, -1)):
        pass

    def mean_color(self):
        pass

    def area(self):
        pass

    def perimeter(self):
        pass

    def width(self):
        pass

    def height(self):
        pass

    def radius(self):
        pass

    def diameter(self):
        pass

    def crop(self):
        pass


class KeyPoint(Feature):
    def __init__(self, img, keypoint, descriptor=None, flavor='SURF'):
        pass

    def get_object(self):
        pass

    def descriptor(self):
        pass

    def quality(self):
        pass

    def ocatve(self):
        pass

    def flavor(self):
        pass

    def angle(self):
        pass

    def draw(self, color=Color.GREEN, width=1):
        pass

    def show(self, color=Color.GREEN):
        pass

    def distance_from(self, point=(-1, -1)):
        pass

    def mean_color(self):
        pass

    def color_distance(self, color=(0, 0, 0)):
        pass

    def perimeter(self):
        pass

    def width(self):
        pass

    def height(self):
        pass

    def radius(self):
        pass

    def diameter(self):
        pass

    def crop(self, no_mask=False):
        pass


class Motion(Feature):
    def __init__(self, img, x, y, dx, dy, wndw):
        pass

    def draw(self, color=Color.GREEN, width=1, normalize=True):
        pass

    def normalize2(self, max_mag):
        pass

    def magnitude(self):
        pass

    def unit_vec(self):
        pass

    def vector(self):
        pass

    def window_size(self):
        pass

    def mean_color(self):
        pass

    def crop(self):
        pass


class KeyPointMatch(Feature):
    def __init__(self):
        pass

    def draw(self, color=Color.GREEN, width=1):
        pass

    def draw_rect(self, color=Color.GREEN, width=1):
        pass

    def crop(self):
        pass

    def mean_color(self):
        pass

    def get_min_rect(self):
        pass

    def get_homography(self):
        pass


class ShapeContextDescriptor(Feature):
    def __init__(self):
        pass

    def draw(self, color=Color.GREEN, width=1):
        pass


class ROI(Feature):
    def __init__(self):
        pass

    def resize(self, w, h=None, percentage=True):
        pass

    def overlaps(self, other):
        pass

    def translate(self, x=0, y=0):
        pass

    def to_xywh(self):
        pass

    def to_tl_br(self):
        pass

    def to_points(self):
        pass

    def to_unit_xywh(self):
        pass

    def to_unit_tl_br(self):
        pass

    def to_unit_points(self):
        pass

    def coord_transform_x(self, x, intype='ROI', output='SRC'):
        pass

    def coord_transform_y(self, y, intype='ROI', output='SRC'):
        pass

    def coord_transform_pts(self, pts, intype='ROI', output='SRC'):
        pass

    def _transform(self, x, img_size, roi_size, offset, intype, output):
        pass

    def split_x(self, x, unit_vals=False, src_vals=False):
        pass

    def split_y(self, y, init_vals=False, src_vals=False):
        pass

    def merge(self, regions):
        pass

    def rebase(self, x, y=None, w=None, h=None):
        pass

    def draw(self, color=Color.GREEN, width=3):
        pass

    def show(self, color=Color.GREEN, width=2):
        pass

    def mean_color(self):
        pass

    def _rebase(self, roi):
        pass

    def _standardize(self, x, y=None, w=None, h=None):
        pass

    def crop(self):
        pass
