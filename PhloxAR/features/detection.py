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
        """
        **SUMMARY**

        Returns the line with endpoints on edges of image. If some endpoints lies inside image
        then those points remain the same without extension to the edges.
        **RETURNS**
        Returns a :py:class:`Line` object. If line does not cross the image's edges or cross at one point returns None.
        **EXAMPLE**
        >>> img = Image("lenna")
        >>> l = Line(img, ((-100, -50), (1000, 25))
        >>> cr_l = l.crop2image_edges()
        """
        pt1, pt2 = self._end_pts
        pt1, pt2 = min(pt1, pt2), max(pt1, pt2)
        x1, y1 = pt1
        x2, y2 = pt2
        w, h = self._image.width - 1, self._image.height - 1
        slope = self.slope

        ep = []
        if slope == float('inf'):
            if 0 <= x1 <= w and 0 <= x2 <= w:
                ep.append((x1, 0))
                ep.append((x2, h))
        elif slope == 0:
            if 0 <= y1 <= w and 0 <= y2 <= w:
                ep.append((0, y1))
                ep.append((w, y2))
        else:
            x = (slope * x1 - y1) / slope  # top edge y = 0
            if 0 <= x <= w:
                ep.append((int(round(x)), 0))

            x = (slope * x1 + h - y1) / slope  # bottom edge y = h
            if 0 <= x <= w:
                ep.append((int(round(x)), h))

            y = -slope * x1 + y1  # left edge x = 0
            if 0 <= y <= h:
                ep.append((0, (int(round(y)))))

            y = slope * (w - x1) + y1  # right edge x = w
            if 0 <= y <= h:
                ep.append((w, (int(round(y)))))

        ep = list(set(
            ep))  # remove duplicates of points if line cross image at corners
        ep.sort()
        if len(ep) == 2:
            # if points lies outside image then change them
            if not (0 < x1 < w and 0 < y1 < h):
                pt1 = ep[0]
            if not (0 < x2 < w and 0 < y2 < h):
                pt2 = ep[1]
        elif len(ep) == 1:
            logger.warning("Line cross the image only at one point")
            return None
        else:
            logger.warning("Line does not cross the image")
            return None

        return Line(self._image, (pt1, pt2))

    @lazy_property
    def vector(self):
        if self._vector is None:
            self._vector = [float(self._end_pts[1][0] - self._end_pts[0][0]),
                            float(self._end_pts[1][1] - self._end_pts[0][1])]

        return self._vector

    def dot(self, other):
        return npy.dot(self.vector, other.vector)

    def cross(self, other):
        return npy.cross(self.vector, other.vector)

    def get_y_intercept(self):
        """
        **SUMMARY**

        Returns the y intercept based on the lines equation.  Note that this point is potentially not contained in the image itself
        **RETURNS**
        Returns a floating point intersection value
        **EXAMPLE**
        >>> img = Image("lena")
        >>> l = Line(img, ((50, 150), (2, 225))
        >>> b = l.get_y_intercept()
        """
        if self._y_intercept is None:
            pt1, pt2 = self._end_pts
            m = self.slope
            # y = mx + b | b = y-mx
            self._y_intercept = pt1[1] - m * pt1[0]
        return self._y_intercept

    def extend2image_edges(self):
        """
        **SUMMARY**

        Returns the line with endpoints on edges of image.
        **RETURNS**
        Returns a :py:class:`Line` object. If line does not lies entirely inside image then returns None.
        **EXAMPLE**
        >>> img = Image("lena")
        >>> l = Line(img, ((50, 150), (2, 225))
        >>> cr_l = l.extend2image_edges()
        """
        pt1, pt2 = self._end_pts
        pt1, pt2 = min(pt1, pt2), max(pt1, pt2)
        x1, y1 = pt1
        x2, y2 = pt2
        w, h = self._image.width - 1, self._image.height - 1
        slope = self.slope

        if not 0 <= x1 <= w or not 0 <= x2 <= w or not 0 <= y1 <= w or not 0 <= y2 <= w:
            logger.warning("At first the line should be cropped")
            return None

        ep = []
        if slope == float('inf'):
            if 0 <= x1 <= w and 0 <= x2 <= w:
                return Line(self._image, ((x1, 0), (x2, h)))
        elif slope == 0:
            if 0 <= y1 <= w and 0 <= y2 <= w:
                return Line(self._image, ((0, y1), (w, y2)))
        else:
            x = (slope * x1 - y1) / slope  # top edge y = 0
            if 0 <= x <= w:
                ep.append((int(round(x)), 0))

            x = (slope * x1 + h - y1) / slope  # bottom edge y = h
            if 0 <= x <= w:
                ep.append((int(round(x)), h))

            y = -slope * x1 + y1  # left edge x = 0
            if 0 <= y <= h:
                ep.append((0, (int(round(y)))))

            y = slope * (w - x1) + y1  # right edge x = w
            if 0 <= y <= h:
                ep.append((w, (int(round(y)))))

        # remove duplicates of points if line cross image at corners
        ep = list(set(ep))
        ep.sort()

        return Line(self._image, ep)

    @property
    def slope(self):
        return self._slope


class Barcode(Feature):
    """
    **SUMMARY**
    The Barcode Feature wrappers the object returned by find_barcode(),
    a zbar symbol
    * The x,y coordinate is the center of the code.
    * points represents the four boundary points of the feature. Note: for
      QR codes, these points are the reference rectangls, and are quadrangular,
      rather than rectangular with other datamatrix types.
    * data is the parsed data of the code.
    """
    _data = ''

    def __init__(self, img, zbsymbol):
        locs = zbsymbol.location
        if len(locs) > 4:
            xs = [l[0] for l in locs]
            ys = [l[1] for l in locs]
            xmax = npy.max(xs)
            xmin = npy.min(xs)
            ymax = npy.max(ys)
            ymin = npy.min(ys)
            points = ((xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin))
        else:
            points = copy(locs)  # hopefully this is in tl clockwise order

        self._data = zbsymbol.data
        self._points = copy(points)
        numpoints = len(self._points)
        self._x = 0
        self._y = 0

        for p in self._points:
            self._x += p[0]
            self._y += p[1]

        if numpoints:
            self._x /= numpoints
            self._y /= numpoints

        super(Barcode, self).__init__(img, 0, 0, points)

    def __repr__(self):
        return "{}.{} at ({}, {}), read data: {}".format(
                self.__class__.__module__, self.__class__.__name__,
                self._x, self._y, self._data
        )

    def draw(self, color=Color.GREEN, width=1):
        """
        **SUMMARY**
        Draws the bounding area of the barcode, given by points.  Note that for
        QR codes, these points are the reference boxes, and so may "stray" into
        the actual code.
        **PARAMETERS**
        * *color* - An RGB color triplet.
        * *width* - if width is less than zero we draw the feature filled in, otherwise we draw the
        contour using the specified width.
        **RETURNS**
        Nothing - this is an inplace operation that modifies the source images drawing layer.
        """
        self._image.draw_line(self._points[0], self._points[1], color, width)
        self._image.draw_line(self._points[1], self._points[2], color, width)
        self._image.draw_line(self._points[2], self._points[3], color, width)
        self._image.draw_line(self._points[3], self._points[0], color, width)

    def length(self):
        """
        **SUMMARY**
        Returns the longest side of the quandrangle formed by the boundary points.
        **RETURNS**
        A floating point length value.
        **EXAMPLE**
        >>> img = Image("mycode.jpg")
        >>> bc = img.findBarcode()
        >>> print(bc[-1].length())
        """
        sqform = spsd.squareform(spsd.pdist(self._points, "euclidean"))
        # get pairwise distances for all points
        # note that the code is a quadrilateral
        return max(sqform[0][1], sqform[1][2], sqform[2][3], sqform[3][0])

    def area(self):
        """
        **SUMMARY**
        Returns the area defined by the quandrangle formed by the boundary points

        **RETURNS**
        An integer area value.
        **EXAMPLE**
        >>> img = Image("mycode.jpg")
        >>> bc = img.findBarcode()
        >>> print(bc[-1].area())
        """
        # calc the length of each side in a square distance matrix
        sqform = spsd.squareform(spsd.pdist(self._points, "euclidean"))

        # squareform returns a N by N matrix
        # boundry line lengths
        a = sqform[0][1]
        b = sqform[1][2]
        c = sqform[2][3]
        d = sqform[3][0]

        # diagonals
        p = sqform[0][2]
        q = sqform[1][3]

        # perimeter / 2
        s = (a + b + c + d) / 2.0

        # i found the formula to do this on wikihow.  Yes, I am that lame.
        # http://www.wikihow.com/Find-the-Area-of-a-Quadrilateral
        return sqrt((s - a) * (s - b) * (s - c) * (s - d) -
                    (a * c + b * d + p * q) * (a * c + b * d - p * q) / 4)


class HaarFeature(Feature):
    """
    The HaarFeature is a rectangle returned by the find_feature function.
    - The x,y coordinates are defined by the center of the bounding rectangle.
    - The classifier property refers to the cascade file used for detection .
    - Points are the clockwise points of the bounding rectangle, starting in
      upper left.
    """
    classifier = None
    _width = None
    _height = None
    neighbors = None
    feature_name = 'None'

    def __init__(self, img, haar_obj, haar_classifier=None, cv2flag=True):
        if cv2flag is False:
            x, y, width, height, self.neighbors = haar_obj
        elif cv2flag is True:
            x, y, width, height = haar_obj

        at_x = x + width / 2
        at_y = y + height / 2  # set location of feature to middle of rectangle.
        points = ((x, y), (x + width, y),
                  (x + width, y + height), (x, y + height))

        # set bounding points of the rectangle
        self.classifier = haar_classifier
        if haar_classifier is not None:
            self.feature_name = haar_classifier.get_name()

        super(HaarFeature, self).__init__(img, at_x, at_y, points)

    def draw(self, color=Color.GREEN, width=1):
        """
        Draw the bounding rectangle, default color is Color.GREEN

        :param color: a RGB color tuple
        :param width: if width is less than zero we draw the feature filled in, otherwise we draw the
                       contour using the specified width.
        :return: None, modify the source images drawing layer.
        """
        self._image.draw_line(self._points[0], self._points[1], color, width)
        self._image.draw_line(self._points[1], self._points[2], color, width)
        self._image.draw_line(self._points[2], self._points[3], color, width)
        self._image.draw_line(self._points[3], self._points[0], color, width)

    def __getstate__(self):
        d = self.__dict__.copy()
        if 'classifier' in d:
            del d['classifier']
        return d

    def mean_color(self):
        """
        Find the mean color of the boundary rectangle

        :return: a RGB tuple that corresponds to the mean color of the feature.

        :Example:
        >>> img = Image('lena')
        >>> face = HaarCascade('face.xml')
        >>> faces = img.find_haar_features(face)
        >>> print(faces[-1].mean_color())
        """
        crop = self._image[self._points[0][0]:self._points[1][0],
                           self._points[0][1]:self._points[2][1]]
        return crop.mean_color()

    def area(self):
        """
        Returns the area of the feature in pixels

        :return: area of feature in pixels.

        :Example:
        >>> img = Image('lena')
        >>> face = HaarCascade('face.xml')
        >>> faces = img.find_haar_features(face)
        >>> print(faces[-1].area())
        """
        return self.width * self.height


class Chessboard(Feature):
    """
    Used for Calibration, it uses a chessboard to calibrate from pixels
    to real world measurements.
    """
    _spcorners = None
    _dims = None

    def __init__(self, img, dim, subpixelscorners):
        self._dims = dim
        self._spcorners = subpixelscorners
        x = npy.average(npy.array(self._spcorners)[:, 0])
        y = npy.average(npy.array(self._spcorners)[:, 1])

        posdiagsorted = sorted(self._spcorners,
                               key=lambda corner: corner[0] + corner[1])
        # sort corners along the x + y axis
        negdiagsorted = sorted(self._spcorners,
                               key=lambda corner: corner[0] - corner[1])
        # sort corners along the x - y axis

        points = (posdiagsorted[0], negdiagsorted[-1],
                  posdiagsorted[-1], negdiagsorted[0])
        super(Chessboard, self).__init__(img, x, y, points)

    def draw(self, no_need_color=None):
        """
        Draws the chessboard corners.

        :param no_need_color:
        :return: None
        """
        cv.DrawChessboardCorners(self._image.bitmap, self._dims,
                                 self._spcorners, 1)

    def area(self):
        """
        **SUMMARY**
        Returns the mean of the distance between corner points in the chessboard
        Given that the chessboard is of a known size, this can be used as a
        proxy for distance from the camera

        :return: the mean distance between the corners.

        :Example:
        >>> img = Image("corners.jpg")
        >>> feats = img.findChessboardCorners()
        >>> print feats[-1].area()
        """
        # note, copying this from barcode means we probably need a subclass of
        # feature called "quandrangle"
        sqform = spsd.squareform(spsd.pdist(self._points, "euclidean"))
        a = sqform[0][1]
        b = sqform[1][2]
        c = sqform[2][3]
        d = sqform[3][0]
        p = sqform[0][2]
        q = sqform[1][3]
        s = (a + b + c + d) / 2.0
        return 2 * sqrt((s - a) * (s - b) * (s - c) * (s - d) -
                        (a * c + b * d + p * q) * (a * c + b * d - p * q) / 4)


class TemplateMatch(Feature):
    """
    **SUMMARY**
    This class is used for template (pattern) matching in images.
    The template matching cannot handle scale or rotation.
    """

    _template_image = None
    _quality = 0
    _w = 0
    _h = 0

    def __init__(self, img, template, location, quality):
        self._template_image = template  # -- KAT - TRYING SOMETHING
        self._image = img
        self._quality = quality
        w = template.width
        h = template.height
        at_x = location[0]
        at_y = location[1]
        points = [(at_x, at_y), (at_x + w, at_y), (at_x + w, at_y + h),
                  (at_x, at_y + h)]

        super(TemplateMatch, self).__init__(img, at_x, at_y, points)

    def _template_overlaps(self, other):
        """
        Returns true if this feature overlaps another template feature.
        """
        (maxx, minx, maxy, miny) = self.extents()
        overlap = False
        for p in other.points:
            if maxx >= p[0] >= minx and maxy >= p[1] >= miny:
                overlap = True
                break

        return overlap

    def consume(self, other):
        """
        Given another template feature, make this feature the size of the two features combined.
        """
        (maxx, minx, maxy, miny) = self.extents()
        (maxx0, minx0, maxy0, miny0) = other.extents()

        maxx = max(maxx, maxx0)
        minx = min(minx, minx0)
        maxy = max(maxy, maxy0)
        miny = min(miny, miny0)
        self._x = minx
        self._y = miny
        self._points = [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
        self._update_extents()

    def rescale(self, w, h):
        """
        This method keeps the feature's center the same but sets a new width and height
        """
        (maxx, minx, maxy, miny) = self.extents()
        xc = minx + ((maxx - minx) / 2)
        yc = miny + ((maxy - miny) / 2)
        x = xc - (w / 2)
        y = yc - (h / 2)
        self._x = x
        self._y = y
        self._points = [(x, y),
                        (x + w, y),
                        (x + w, y + h),
                        (x, y + h)]
        self._update_extents()

    def crop(self):
        (maxx, minx, maxy, miny) = self.extents()
        return self._image.crop(minx, miny, maxx - minx, maxy - miny)

    def draw(self, color=Color.GREEN, width=1):
        """
        **SUMMARY**
        Draw the bounding rectangle, default color green.
        **PARAMETERS**
        * *color* - An RGB color triplet.
        * *width* - if width is less than zero we draw the feature filled in, otherwise we draw the
          contour using the specified width.
        **RETURNS**
        Nothing - this is an inplace operation that modifies the source images drawing layer.
        """
        self._image.dl().rectangle((self._x, self._y),
                                   (self.width(), self.height()),
                                   color=color, width=width)


class Circle(Feature):
    """
    Class for a general circle feature with a center at (x,y) and a radius r
    """
    _radius = 0.00
    _avg_color = None
    _contour = None

    def __init__(self, img, at_x, at_y, r):
        self._radius = r
        points = [(at_x - r, at_y - r), (at_x + r, at_y - r),
                  (at_x + r, at_y + r), (at_x - r, at_y + r)]
        self._avg_color = None
        super(Circle, self).__init__(img, at_x, at_y, points)
        segments = 18
        rng = range(1, segments + 1)
        self._contour = []

        for theta in rng:
            rp = 2.0 * npy.pi * float(theta) / float(segments)
            x = (r * npy.sin(rp)) + at_x
            y = (r * npy.cos(rp)) + at_y
            self._contour.append((x, y))

    def draw(self, color=Color.GREEN, width=1):
        """
        **SUMMARY**
        With no dimension information, color the x,y point for the feature.
        **PARAMETERS**
        * *color* - An RGB color triplet.
        * *width* - if width is less than zero we draw the feature filled in, otherwise we draw the
        contour using the specified width.
        **RETURNS**
        Nothing - this is an inplace operation that modifies the source images drawing layer.
        """
        self._image.dl().circle((self._x, self._y), self._radius, color, width)

    def show(self, color=Color.GREEN):
        """
        **SUMMARY**
        This function will automatically draw the features on the image and show it.
        It is a basically a shortcut function for development and is the same as:
        **PARAMETERS**
        * *color* - the color of the feature as an rgb triplet.
        **RETURNS**
        Nothing - this is an inplace operation that modifies the source images drawing layer.
        **EXAMPLE**
        >>> img = Image("logo")
        >>> feat = img.findCircle()
        >>> feat[0].show()
        """
        self.draw(color)
        self._image.show()

    def distance_from(self, point=(-1, -1)):
        """
        **SUMMARY**
        Given a point (default to center of the image), return the euclidean distance of x,y from this point.
        **PARAMETERS**
        * *point* - The point, as an (x,y) tuple on the image to measure distance from.
        **RETURNS**
        The distance as a floating point value in pixels.
        **EXAMPLE**
        >>> img = Image("OWS.jpg")
        >>> blobs = img.findCircle()
        >>> blobs[-1].distanceFrom(blobs[-2].coordinates())
        """
        if point[0] == -1 or point[1] == -1:
            point = npy.array(self._image.size()) / 2
        return spsd.euclidean(point, [self.x, self.y])

    def mean_color(self):
        """
        **SUMMARY**
        Returns the average color within the circle.
        **RETURNS**
        Returns an RGB triplet that corresponds to the mean color of the feature.
        **EXAMPLE**
        >>> img = Image("lenna")
        >>> c = img.findCircle()
        >>> c[-1].meanColor()
        """
        # generate the mask
        if self._avg_color is None:
            mask = self._image.zeros(1)
            cv.Zero(mask)
            cv.Circle(mask, (self._x, self._y), self._radius,
                      color=(255, 255, 255), thickness=-1)
            temp = cv.Avg(self._image.bitmap, mask)
            self._avg_color = (temp[0], temp[1], temp[2])
        return self._avg_color

    @property
    def area(self):
        """
        Area covered by the feature -- for a pixel, 1
        **SUMMARY**
        Returns a numpy array of the area of each feature in pixels.
        **RETURNS**
        A numpy array of all the positions in the featureset.
        **EXAMPLE**
        >>> img = Image("lenna")
        >>> feats = img.findBlobs()
        >>> xs = feats.coordinates()
        >>> print(xs)
        """
        return self._radius * self._radius * npy.pi

    @property
    def perimeter(self):
        """
        Returns the perimeter of the circle feature in pixels.
        """
        return 2 * npy.pi * self._radius

    @property
    def width(self):
        """
        Returns the width of the feature -- for compliance just r*2
        """
        return self._radius * 2

    @property
    def height(self):
        """
        Returns the height of the feature -- for compliance just r*2
        """
        return self._radius * 2

    @property
    def radius(self):
        """
        Returns the radius of the circle in pixels.
        """
        return self._radius

    @property
    def diameter(self):
        """
        Returns the diameter of the circle in pixels.
        """
        return self._radius * 2

    def crop(self, no_mask=False):
        """
        **SUMMARY**
        This function returns the largest bounding box for an image.
        **PARAMETERS**
        * *noMask* - if noMask=True we return the bounding box image of the circle.
        if noMask=False (default) we return the masked circle with the rest of the area set to black
        **RETURNS**
        The masked circle image.
        """
        if no_mask:
            return self._image.crop(self.x, self.y, self.width, self.height,
                                    centered=True)
        else:
            mask = self._image.zeros(1)
            result = self._image.zeros()
            cv.Zero(mask)
            cv.Zero(result)
            # if you want to shave a bit of time we go do the crop before the blit
            cv.Circle(mask, (self._x, self._y), self._radius,
                      color=(255, 255, 255), thickness=-1)
            cv.Copy(self._image.bitmap, result, mask)
            ret = Image(result)
            ret = ret.crop(self._x, self._y, self.width, self.height,
                                 centered=True)
            return ret


class KeyPoint(Feature):
    """
    The class is place holder for SURF/SIFT/ORB/STAR keypoints.
    """
    _radius = 0.00
    _avg_color = None
    _angle = 0
    _octave = 0
    _response = 0.00
    _flavor = ''
    _descriptor = None
    _keypoint = None

    def __init__(self, img, keypoint, descriptor=None, flavor='SURF'):
        self._keypoint = keypoint
        x = keypoint.pt[1]  # KAT
        y = keypoint.pt[0]
        self._radius = keypoint.size / 2.0
        self._avg_color = None
        self._image = img
        self._angle = keypoint.angle
        self._octave = keypoint.octave
        self._response = keypoint.response
        self._flavor = flavor
        self._descriptor = descriptor
        r = self._radius
        points = ((x + r, y + r), (x + r, y - r),
                  (x - r, y - r), (x - r, y + r))
        super(KeyPoint, self).__init__(img, x, y, points)

        segments = 18
        rng = range(1, segments + 1)
        self._points = []
        for theta in rng:
            rp = 2.0 * npy.pi * float(theta) / float(segments)
            x = (r * npy.sin(rp)) + self.x
            y = (r * npy.cos(rp)) + self.y
            self._points.append((x, y))

    @property
    def keypoint(self):
        """
        Returns the raw keypoint object
        """
        return self._keypoint

    @property
    def descriptor(self):
        """
        Returns the raw keypoint descriptor
        """
        return self._descriptor

    @property
    def quality(self):
        """
        Returns the quality metric for the keypoint object.
        """
        return self._response

    @property
    def octave(self):
        """
        Returns the raw keypoint's octave (if it has)
        """
        return self._octave

    @property
    def flavor(self):
        """
        Returns the type of keypoint as a string (e.g. SURF/MSER/ETC)
        """
        return self._flavor

    def angle(self):
        """
        Return the angle (theta) in degrees of the feature. The default is 0 (horizontal).
        """
        return self._angle

    def draw(self, color=Color.GREEN, width=1):
        """
        **SUMMARY**
        Draw a circle around the feature.  Color tuple is single parameter, default is Green.
        **PARAMETERS**
        * *color* - An RGB color triplet.
        * *width* - if width is less than zero we draw the feature filled in, otherwise we draw the
        contour using the specified width.
        **RETURNS**
        Nothing - this is an inplace operation that modifies the source images drawing layer.
        """
        self._image.dl().circle((self._x, self._y), self._radius, color, width)
        pt1 = (int(self._x), int(self._y))
        pt2 = (int(self._x + (self.radius * sin(radians(self.angle)))),
               int(self._y + (self.radius * cos(radians(self.angle)))))
        self._image.dl().line(pt1, pt2, color, width)

    def show(self, color=Color.GREEN):
        """
        **SUMMARY**
        This function will automatically draw the features on the image and show it.
        It is a basically a shortcut function for development and is the same as:
        >>> img = Image("logo")
        >>> feat = img.find_blobs()
        >>> if feat: feat.draw()
        >>> img.show()
        """
        self.draw(color)
        self._image.show()

    def distance_from(self, point=(-1, -1)):
        """
        **SUMMARY**
        Given a point (default to center of the image), return the euclidean distance of x,y from this point
        """
        if point[0] == -1 or point[1] == -1:
            point = npy.array(self._image.size()) / 2
        return spsd.euclidean(point, [self._x, self._y])

    def mean_color(self):
        """
        **SUMMARY**
        Return the average color within the feature's radius
        **RETURNS**
        Returns an  RGB triplet that corresponds to the mean color of the feature.
        **EXAMPLE**
        >>> img = Image("lenna")
        >>> kp = img.findKeypoints()
        >>> c = kp[0].meanColor()
        """
        # generate the mask
        if self._avg_color is None:
            mask = self._image.zeros(1)
            cv.Zero(mask)
            cv.Circle(mask, (int(self._x), int(self._y)), int(self._radius),
                      color=(255, 255, 255), thickness=-1)
            temp = cv.Avg(self._image.bitmap, mask)
            self._avg_color = (temp[0], temp[1], temp[2])
        return self._avg_color

    def color_distance(self, color=(0, 0, 0)):
        """
        Return the euclidean color distance of the color tuple at x,y from a given color (default black)
        """
        return spsd.euclidean(npy.array(color), npy.array(self.mean_color()))

    @property
    def perimeter(self):
        """
        **SUMMARY**
        Returns the perimeter of the circle feature in pixels.
        """
        return 2 * npy.pi * self._radius

    @property
    def width(self):
        """
        Returns the width of the feature -- for compliance just r*2
        """
        return self._radius * 2

    def height(self):
        """
        Returns the height of the feature -- for compliance just r*2
        """
        return self._radius * 2

    @property
    def radius(self):
        return self._radius

    @property
    def diameter(self):
        return self._radius * 2

    def crop(self, no_mask=False):
        """
        **SUMMARY**
        This function returns the largest bounding box for an image.
        **PARAMETERS**
        * *noMask* - if noMask=True we return the bounding box image of the circle.
        if noMask=False (default) we return the masked circle with the rest of the area set to black
        **RETURNS**
        The masked circle image.
        """
        if no_mask:
            return self._image.crop(self._x, self._y, self.width, self.height,
                                   centered=True)
        else:
            mask = self._image.zeros(1)
            result = self._image.zeros()
            cv.Zero(mask)
            cv.Zero(result)
            # if you want to shave a bit of time we go do the crop before the blit
            cv.Circle(mask, (int(self._x), int(self._y)), int(self._radius),
                      color=(255, 255, 255), thickness=-1)
            cv.Copy(self._image.bitmap, result, mask)
            ret = Image(result)
            ret = ret.crop(self._x, self._y, self.width, self.height,
                                 centered=True)
            return ret


class Motion(Feature):
    """
    The motion feature is used to encapsulate optical flow vectors. The feature
    holds the length and direction of the vector.
    """
    dx = 0.00
    dy = 0.00
    norm_dx = 0.00
    norm_dy = 0.00
    window = 7

    def __init__(self, img, at_x, at_y, dx, dy, wndw):
        """
        img  - the source image.
        at_x - the sample x pixel position on the image.
        at_y - the sample y pixel position on the image.
        dx   - the x component of the optical flow vector.
        dy   - the y component of the optical flow vector.
        wndw - the size of the sample window (we assume it is square).
        """
        self.dx = dx  # the direction of the vector
        self.dy = dy
        self.window = wndw  # the size of the sample window
        sz = wndw / 2
        # so we center at the flow vector
        points = [(at_x + sz, at_y + sz), (at_x - sz, at_y + sz),
                  (at_x + sz, at_y + sz), (at_x + sz, at_y - sz)]
        super(Motion, self).__init__(img, at_x, at_y, points)

    def draw(self, color=Color.GREEN, width=1, normalize=True):
        """
        **SUMMARY**
        Draw the optical flow vector going from the sample point along the length of the motion vector.
        **PARAMETERS**
        * *color* - An RGB color triplet.
        * *width* - if width is less than zero we draw the feature filled in, otherwise we draw the
        contour using the specified width.
        * *normalize* - normalize the vector size to the size of the block (i.e. the biggest optical flow
        vector is scaled to the size of the block, all other vectors are scaled relative to
        the longest vector.
        **RETURNS**
        Nothing - this is an inplace operation that modifies the source images drawing layer.
        """
        new_x = 0
        new_y = 0
        if normalize:
            win = self.window / 2
            w = math.sqrt((win * win) * 2)
            new_x = self.norm_dx * w + self.x
            new_y = self.norm_dy * w + self.y
        else:
            new_x = self._x + self.dx
            new_y = self._y + self.dy

        self._image.dl().line((self.x, self.y), (new_x, new_y), color, width)

    def normalize2(self, max_mag):
        """
        **SUMMARY**
        This helper method normalizes the vector give an input magnitude.
        This is helpful for keeping the flow vector inside the sample window.
        """
        if max_mag == 0:
            self.norm_dx = 0
            self.norm_dy = 0
            return None

        mag = self.magnitude
        new_mag = mag / max_mag
        unit = self.unit_vec
        self.norm_dx = unit[0] * new_mag
        self.norm_dy = unit[1] * new_mag

    @property
    def magnitude(self):
        """
        Returns the magnitude of the optical flow vector.
        """
        return npy.sqrt((self.dx * self.dx) + (self.dy * self.dy))

    @property
    def unit_vec(self):
        """
        Returns the unit vector direction of the flow vector as an (x,y) tuple.
        """
        mag = self.magnitude
        if mag != 0.00:
            return float(self.dx) / mag, float(self.dy) / mag
        else:
            return 0.00, 0.00

    @property
    def vector(self):
        """
        Returns the raw direction vector as an (x,y) tuple.
        """
        return self.dx, self.dy

    @property
    def window_size(self):
        """
        Return the window size that we sampled over.
        """
        return self.window

    def mean_color(self):
        """
        Return the color tuple from x,y
        **SUMMARY**
        Return a numpy array of the average color of the area covered by each Feature.
        **RETURNS**
        Returns an array of RGB triplets the correspond to the mean color of the feature.
        **EXAMPLE**
        >>> img = Image("lenna")
        >>> kp = img.findKeypoints()
        >>> c = kp.meanColor()
        """
        x = int(self.x - (self.window / 2))
        y = int(self.y - (self.window / 2))
        return self._image.crop(x, y, int(self.window),
                                int(self.window)).mean_color()

    def crop(self):
        """
        This function returns the image in the sample window around the flow vector.
        Returns Image
        """
        x = int(self._x - (self.window / 2))
        y = int(self._y - (self.window / 2))

        return self._image.crop(x, y, int(self.window), int(self.window))


class KeyPointMatch(Feature):
    """
    This class encapsulates a keypoint match between images of an object.
    It is used to record a template of one image as it appears in another image
    """
    _min_rect = []
    _avg_color = None
    _homography = []
    _template = None

    def __init__(self, img, template, min_rect, homography):
        self._template = template
        self._min_rect = min_rect
        self._homography = homography
        xmax = 0
        ymax = 0
        xmin = img.width
        ymin = img.height
        for p in min_rect:
            if p[0] > xmax:
                xmax = p[0]
            if p[0] < xmin:
                xmin = p[0]
            if p[1] > ymax:
                ymax = p[1]
            if p[1] < ymin:
                ymin = p[1]

        width = xmax - xmin
        height = ymax - ymin
        at_x = xmin + width / 2
        at_y = ymin + height / 2
        points = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
        super(KeyPointMatch, self).__init__(img, at_x, at_y, points)

    def draw(self, color=Color.GREEN, width=1):
        """
        The default drawing operation is to draw the min bounding
        rectangle in an image.
        **SUMMARY**
        Draw a small circle around the corner.  Color tuple is single parameter, default is Red.
        **PARAMETERS**
        * *color* - An RGB color triplet.
        * *width* - if width is less than zero we draw the feature filled in, otherwise we draw the
        contour using the specified width.
        **RETURNS**
        Nothing - this is an inplace operation that modifies the source images drawing layer.
        """
        self._image.dl().line(self._min_rect[0], self._min_rect[1], color, width)
        self._image.dl().line(self._min_rect[1], self._min_rect[2], color, width)
        self._image.dl().line(self._min_rect[2], self._min_rect[3], color, width)
        self._image.dl().line(self._min_rect[3], self._min_rect[0], color, width)

    def draw_rect(self, color=Color.GREEN, width=1):
        """
            This method draws the axes alligned square box of the template
            match. This box holds the minimum bounding rectangle that describes
            the object. If the minimum bounding rectangle is axes aligned
            then the two bounding rectangles will match.
            """
        self._image.dl().line(self._points[0], self._points[1], color, width)
        self._image.dl().line(self._points[1], self._points[2], color, width)
        self._image.dl().line(self._points[2], self._points[3], color, width)
        self._image.dl().line(self._points[3], self._points[0], color, width)

    def crop(self):
        """
        Returns a cropped image of the feature match. This cropped version is the
        axes aligned box masked to just include the image data of the minimum bounding
        rectangle.
        """
        tl = self.top_left_corner()
        # crop the minbouding rect
        raw = self._image.crop(tl[0], tl[1], self.width, self.height)
        return raw

    def mean_color(self):
        """
        return the average color within the circle
        **SUMMARY**
        Return a numpy array of the average color of the area covered by each Feature.
        **RETURNS**
        Returns an array of RGB triplets the correspond to the mean color of the feature.
        **EXAMPLE**
        >>> img = Image("lena")
        >>> kp = img.find_keypoints()
        >>> c = kp.mean_color()
        """
        if self._avg_color is None:
            tl = self.top_left_corner()
            # crop the minbouding rect
            raw = self._image.crop(tl[0], tl[0], self.width, self.height)
            mask = Image((self.width, self.height))
            mask.dl().polygon(self._min_rect, color=Color.WHITE, filled=TRUE)
            mask = mask.apply_layers()
            ret = cv.Avg(raw.getBitmap(), mask._gray_bitmap_func())
            self._avg_color = ret
        else:
            ret = self._avg_color
        return ret

    @property
    def min_rect(self):
        """
        Returns the minimum bounding rectangle of the feature as a list
        of (x,y) tuples.
        """
        return self._min_rect

    @property
    def homography(self):
        """
        Returns the _homography matrix used to calulate the minimum bounding
        rectangle.
        """
        return self._homography


class ShapeContextDescriptor(Feature):
    _min_rect = []
    _avg_color = None
    _descriptor = None
    _src_blob = None

    def __init__(self, img, point, descriptor, blob):
        self._descriptor = descriptor
        self._sourceBlob = blob
        x = point[0]
        y = point[1]
        points = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1),
                  (x - 1, y + 1)]
        super(ShapeContextDescriptor, self).__init__(img, x, y, points)

    def draw(self, color=Color.GREEN, width=1):
        """
        The default drawing operation is to draw the min bounding
        rectangle in an image.
        **SUMMARY**
        Draw a small circle around the corner.  Color tuple is single parameter, default is Red.
        **PARAMETERS**
        * *color* - An RGB color triplet.
        * *width* - if width is less than zero we draw the feature filled in, otherwise we draw the
          contour using the specified width.
        **RETURNS**
        Nothing - this is an inplace operation that modifies the source images drawing layer.
        """
        self._image.dl().circle((self._x, self._y), 3, color, width)


class ROI(Feature):
    """
    This class creates a region of interest that inherit from one
    or more features or no features at all.
    """
    w = 0
    h = 0
    xtl = 0  # top left x
    ytl = 0  # top left y
    # we are going to assume x,y,w,h is our canonical form
    _sub_features = []
    _mean_color = None

    def __init__(self, x, y=None, w=None, h=None, img=None):
        """
        **SUMMARY**
        This function can handle just about whatever you throw at it
        and makes a it into a feature. Valid input items are tuples and lists
        of x,y points, features, featuresets, two x,y points, and a
        set of x,y,width,height values.
        **PARAMETERS**
        * *x* - this can be just about anything, a list or tuple of x points,
        a corner of the image, a list of (x,y) points, a Feature, a FeatureSet
        * *y* - this is usually a second point or set of y values.
        * *w* - a width
        * *h* - a height.

        **RETURNS**
        Nothing.
        **EXAMPLE**
        >>> img = Image('lenna')
        >>> x,y = npy.where(img.threshold(230).getGrayNumpy() > 128 )
        >>> roi = ROI(zip(x,y),img)
        >>> roi = ROI(x,y,img)
        """
        # After forgetting to set img=Image I put this catch
        # in to save some debugging headache.
        if isinstance(y, Image):
            self._image = y
            y = None
        elif isinstance(w, Image):
            self._image = w
            w = None
        elif isinstance(h, Image):
            self._image = h
            h = None
        else:
            self._image = img

        if img is None and isinstance(x, (Feature, FeatureSet)):
            if isinstance(x, Feature):
                self._image = x.image
            if isinstance(x, FeatureSet) and len(x) > 0:
                self._image = x[0].image

        if isinstance(x, Feature):
            self._sub_features = FeatureSet([x])
        elif isinstance(x, (list, tuple) and len(x) > 0 and
              isinstance(x, Feature)):
            self._sub_features = FeatureSet(x)

        result = self._standardize(x, y, w, h)
        if result is None:
            logger.warning("Could not create an ROI from your data.")
            return
        self._rebase(result)
        super(ROI, self).__init__(img, 0, 0, None)

    def resize(self, w, h=None, percentage=True):
        """
        **SUMMARY**
        Contract/Expand the roi. By default use a percentage, otherwise use pixels.
        This is all done relative to the center of the roi

        **PARAMETERS**
        * *w* - the percent to grow shrink the region is the only parameter, otherwise
                it is the new ROI width
        * *h* - The new roi height in terms of pixels or a percentage.
        * *percentage* - If true use percentages (e.g. 2 doubles the size), otherwise
                         use pixel values.
        * *h* - a height.

        **RETURNS**
        Nothing.
        **EXAMPLE**
        >>> roi = ROI(10,10,100,100,img)
        >>> roi.resize(2)
        >>> roi.show()
        """
        if h is None and isinstance(w, (tuple, list)):
            h = w[1]
            w = w[0]

        if percentage:
            if h is None:
                h = w
            nw = self.w * w
            nh = self.h * h
            nx = self.xtl + ((self.w - nw) / 2.0)
            ny = self.ytl + ((self.h - nh) / 2.0)
            self._rebase([nx, ny, nw, nh])
        else:
            nw = self.w + w
            nh = self.h + h
            nx = self.xtl + ((self.w - nw) / 2.0)
            ny = self.ytl + ((self.h - nh) / 2.0)
            self._rebase([nx, ny, nw, nh])

    def overlaps(self, other):
        for p in other.points:
            if (self.max_x() >= p[0] >= self.min_x() and
                    self.max_y() >= p[1] >= self.min_y()):
                return True
        return False

    def translate(self, x=0, y=0):
        """
        Move the roi.

        **PARAMETERS**
        * *x* - Move the ROI horizontally.
        * *y* - Move the ROI vertically

        **RETURNS**
        Nothing.
        **EXAMPLE**
        >>> roi = ROI(10,10,100,100,img)
        >>> roi.translate(30,30)
        >>> roi.show()
        """
        if x == 0 and y == 0:
            return

        if y == 0 and isinstance(x, (tuple, list)):
            y = x[1]
            x = x[0]

        if isinstance(x, (float, int)) and isinstance(y, (float, int)):
            self._rebase([self.xtl + x, self.ytl + y, self.w, self.h])

    def to_xywh(self):
        """
        Get the ROI as a list of the top left corner's x and y position
        and the roi's width and height in pixels.
        **RETURNS**
        A list of the form [x,y,w,h]
        **EXAMPLE**
        >>> roi = ROI(10,10,100,100,img)
        >>> roi.translate(30,30)
        >>> print(roi.to_xywh())
        """
        return [self.xtl, self.ytl, self.w, self.h]

    def to_tl_br(self):
        """
        Get the ROI as a list of tuples of the ROI's top left
        corner and bottom right corner.
        **RETURNS**
        A list of the form [(x,y),(x,y)]
        **EXAMPLE**
        >>> roi = ROI(10,10,100,100,img)
        >>> roi.translate(30,30)
        >>> print(roi.to_tl_br())
        """
        return [(self.xtl, self.ytl), (self.xtl + self.w, self.ytl + self.h)]

    def to_points(self):
        """
        Get the ROI as a list of four points that make up the bounding rectangle.

        **RETURNS**
        A list of the form [(x,y),(x,y),(x,y),(x,y)]
        **EXAMPLE**
        >>> roi = ROI(10,10,100,100,img)
        >>> print(roi.to_points())
        """
        tl = (self.xtl, self.ytl)
        tr = (self.xtl + self.w, self.ytl)
        br = (self.xtl + self.w, self.ytl + self.h)
        bl = (self.xtl, self.ytl + self.h)
        return [tl, tr, br, bl]

    def to_unit_xywh(self):
        """
        Get the ROI as a list, the values are top left x, to left y,
        width and height. These values are scaled to unit values with
        respect to the source image..


        **RETURNS**
        A list of the form [x,y,w,h]
        **EXAMPLE**
        >>> roi = ROI(10,10,100,100,img)
        >>> print(roi.to_unit_xywh())
        """
        if self._image is None:
            return None

        srcw = float(self._image.width)
        srch = float(self._image.height)
        x, y, w, h = self.to_xywh()
        nx = 0
        ny = 0

        if x != 0:
            nx = x / srcw

        if y != 0:
            ny = y / srch

        return [nx, ny, w / srcw, h / srch]

    def to_unit_tl_br(self):
        """
        Get the ROI as a list of tuples of the ROI's top left
        corner and bottom right corner. These coordinates are in unit
        length values with respect to the source image.
        **RETURNS**
        A list of the form [(x,y),(x,y)]
        **EXAMPLE**
        >>> roi = ROI(10,10,100,100,img)
        >>> roi.translate(30,30)
        >>> print(roi.to_unit_tl_br())
        """

        if self._image is None:
            return None
        srcw = float(self._image.width)
        srch = float(self._image.height)
        x, y, w, h = self.to_xywh()
        nx = 0
        ny = 0
        nw = w / srcw
        nh = h / srch
        if x != 0:
            nx = x / srcw

        if y != 0:
            ny = y / srch

        return [(nx, ny), (nx + nw, ny + nh)]

    def to_unit_points(self):
        """
        Get the ROI as a list of four points that make up the bounding rectangle.
        Each point is represented in unit coordinates with respect to the
        souce image.

        **RETURNS**
        A list of the form [(x,y),(x,y),(x,y),(x,y)]
        **EXAMPLE**
        >>> roi = ROI(10,10,100,100,img)
        >>> print(roi.to_unit_points())
        """
        if self._image is None:
            return None
        srcw = float(self._image.width)
        srch = float(self._image.height)

        pts = self.to_points()
        ret = []
        for p in pts:
            x, y = p
            if x != 0:
                x /= srcw

            if y != 0:
                y /= srch
            ret.append((x, y))
        return ret

    def coord_transform_x(self, x, intype='ROI', output='SRC'):
        """
        Transform a single or a set of x values from one reference frame to another.
        Options are:

        SRC - the coordinates of the source image.
        ROI - the coordinates of the ROI
        ROI_UNIT - unit coordinates in the frame of reference of the ROI
        SRC_UNIT - unit coordinates in the frame of reference of source image.
        **PARAMETERS**
        * *x* - A list of x values or a single x value.
        * *intype* - A string indicating the input format of the data.
        * *output* - A string indicating the output format of the data.
        **RETURNS**
        A list of the transformed values.
        **EXAMPLE**
        >>> img = Image('lenna')
        >>> blobs = img.findBlobs()
        >>> roi = ROI(blobs[0])
        >>> x = roi.crop()..... /find some x values in the crop region
        >>> xt = roi.coord_transform_x(x)
        >>> #xt are no in the space of the original image.
        """
        if self._image is None:
            logger.warning("No image to perform that calculation")
            return None

        if isinstance(x, (float, int)):
            x = [x]

        intype = intype.upper()
        output = output.upper()

        if intype == output:
            return x

        return self._transform(x, self._image.width, self.w, self.xtl, intype,
                               output)

    def coord_transform_y(self, y, intype='ROI', output='SRC'):
        """
        Transform a single or a set of y values from one reference frame to another.
        Options are:

        SRC - the coordinates of the source image.
        ROI - the coordinates of the ROI
        ROI_UNIT - unit coordinates in the frame of reference of the ROI
        SRC_UNIT - unit coordinates in the frame of reference of source image.
        **PARAMETERS**
        * *y* - A list of y values or a single y value.
        * *intype* - A string indicating the input format of the data.
        * *output* - A string indicating the output format of the data.
        **RETURNS**
        A list of the transformed values.
        **EXAMPLE**
        >>> img = Image('lenna')
        >>> blobs = img.findBlobs()
        >>> roi = ROI(blobs[0])
        >>> y = roi.crop()..... /find some y values in the crop region
        >>> yt = roi.coord_transform_y(y)
        >>> #yt are no in the space of the original image.
        """

        if self._image is None:
            logger.warning("No image to perform that calculation")
            return None

        if isinstance(y, (float, int)):
            y = [y]

        intype = intype.upper()
        output = output.upper()

        if intype == output:
            return y
        return self._transform(y, self._image.height, self.h, self.ytl, intype,
                               output)

    def coord_transform_pts(self, pts, intype='ROI', output='SRC'):
        """
        Transform a set of (x,y) values from one reference frame to another.
        Options are:

        SRC - the coordinates of the source image.
        ROI - the coordinates of the ROI
        ROI_UNIT - unit coordinates in the frame of reference of the ROI
        SRC_UNIT - unit coordinates in the frame of reference of source image.
        **PARAMETERS**
        * *pts* - A list of (x,y) values or a single (x,y) value.
        * *intype* - A string indicating the input format of the data.
        * *output* - A string indicating the output format of the data.
        **RETURNS**
        A list of the transformed values.
        **EXAMPLE**
        >>> img = Image('lenna')
        >>> blobs = img.findBlobs()
        >>> roi = ROI(blobs[0])
        >>> pts = roi.crop()..... /find some x,y values in the crop region
        >>> pts = roi.coord_transform_pts(pts)
        >>> #yt are no in the space of the original image.
        """
        if self._image is None:
            logger.warning("No image to perform that calculation")
            return None
        if isinstance(pts, tuple) and len(pts) == 2:
            pts = [pts]
        intype = intype.upper()
        output = output.upper()
        x = [pt[0] for pt in pts]
        y = [pt[1] for pt in pts]

        if intype == output:
            return pts

        x = self._transform(x, self._image.width, self.w, self.xtl, intype,
                            output)
        y = self._transform(y, self._image.height, self.h, self.ytl, intype,
                            output)
        return zip(x, y)

    def _transform(self, x, img_size, roi_size, offset, intype, output):
        xtemp = []
        # we are going to go to src unit coordinates
        # and then we'll go back.
        if intype == "SRC":
            xtemp = [xt / float(img_size) for xt in x]
        elif intype == "ROI":
            xtemp = [(xt + offset) / float(img_size) for xt in x]
        elif intype == "ROI_UNIT":
            xtemp = [((xt * roi_size) + offset) / float(img_size) for xt in x]
        elif intype == "SRC_UNIT":
            xtemp = x
        else:
            logger.warning("Bad Parameter to CoordTransformX")
            return None

        ret = []
        if output == "SRC":
            ret = [int(xt * img_size) for xt in xtemp]
        elif output == "ROI":
            ret = [int((xt * img_size) - offset) for xt in xtemp]
        elif output == "ROI_UNIT":
            ret = [int(((xt * img_size) - offset) / float(roi_size)) for xt in
                      xtemp]
        elif output == "SRC_UNIT":
            ret = xtemp
        else:
            logger.warning("Bad Parameter to CoordTransformX")
            return None

        return ret

    def split_x(self, x, unit_vals=False, src_vals=False):
        """
        **SUMMARY**
        Split the ROI at an x value.
        x can be a list of sequentianl tuples of x split points  e.g [0.3,0.6]
        where we assume the top and bottom are also on the list.
        Use units to split as a percentage (e.g. 30% down).
        The srcVals means use coordinates of the original image.
        **PARAMETERS**
        * *x*-The split point. Can be a single point or a list of points. the type is determined by the flags.
        * *unitVals* - Use unit vals for the split point. E.g. 0.5 means split at 50% of the ROI.
        * *srcVals* - Use x values relative to the source image rather than relative to the ROI.


        **RETURNS**

        Returns a feature set of ROIs split from the source ROI.
        **EXAMPLE**
        >>> roi = ROI(0,0,100,100,img)
        >>> splits = roi.split_x(50) # create two ROIs

        """
        retVal = FeatureSet()
        if unit_vals and src_vals:
            logger.warning("Not sure how you would like to split the feature")
            return None

        if not isinstance(x, (list, tuple)):
            x = [x]

        if unit_vals:
            x = self.coord_transform_x(x, intype="ROI_UNIT", output="SRC")
        elif not src_vals:
            x = self.coord_transform_x(x, intype="ROI", output="SRC")

        for xt in x:
            if xt < self.xtl or xt > self.xtl + self.w:
                logger.warning("Invalid split point.")
                return None

        x.insert(0, self.xtl)
        x.append(self.xtl + self.w)
        for i in range(0, len(x) - 1):
            xstart = x[i]
            xstop = x[i + 1]
            w = xstop - xstart
            retVal.append(ROI(xstart, self.ytl, w, self.h, self._image))
        return retVal

    def split_y(self, y, unit_vals=False, src_vals=False):
        """
        Split the ROI at an x value.
        y can be a list of sequentianl tuples of y split points  e.g [0.3,0.6]
        where we assume the top and bottom are also on the list.
        Use units to split as a percentage (e.g. 30% down).
        The srcVals means use coordinates of the original image.
        **PARAMETERS**
        * *y*-The split point. Can be a single point or a list of points. the type is determined by the flags.
        * *unitVals* - Use unit vals for the split point. E.g. 0.5 means split at 50% of the ROI.
        * *srcVals* - Use x values relative to the source image rather than relative to the ROI.

        **RETURNS**

        Returns a feature set of ROIs split from the source ROI.
        **EXAMPLE**
        >>> roi = ROI(0,0,100,100,img)
        >>> splits = roi.split_y(50) # create two ROIs

        """
        ret = FeatureSet()
        if unit_vals and src_vals:
            logger.warning("Not sure how you would like to split the feature")
            return None

        if not isinstance(y, (list, tuple)):
            y = [y]

        if unit_vals:
            y = self.coord_transform_y(y, intype="ROI_UNIT", output="SRC")
        elif not src_vals:
            y = self.coord_transform_y(y, intype="ROI", output="SRC")

        for yt in y:
            if yt < self.ytl or yt > self.ytl + self.h:
                logger.warning("Invalid split point.")
                return None

        y.insert(0, self.ytl)
        y.append(self.ytl + self.h)

        for i in range(0, len(y) - 1):
            ystart = y[i]
            ystop = y[i + 1]
            h = ystop - ystart
            ret.append(ROI(self.xtl, ystart, self.w, h, self._image))
        return ret

    def merge(self, regions):
        """
        **SUMMARY**

        Combine another region, or regions with this ROI. Everything must be
        in the source image coordinates. Regions can be a ROIs, [ROI], features,
        FeatureSets, or anything that can be cajoled into a region.
        **PARAMETERS**
        * *regions* - A region or list of regions. Regions are just about anything that has position.

        **RETURNS**
        Nothing, but modifies this region.
        **EXAMPLE**
        >>>  blobs = img.find_blobs()
        >>>  roi = ROI(blobs[0])
        >>>  print(roi.to_xywh())
        >>>  roi.merge(blobs[2])
        >>>  print(roi.to_xywh())

        """
        result = self._standardize(regions)

        if result is not None:
            xo, yo, wo, ho = result
            x = npy.min([xo, self.xtl])
            y = npy.min([yo, self.ytl])
            w = npy.max([self.xtl + self.w, xo + wo]) - x
            h = npy.max([self.ytl + self.h, yo + ho]) - y

            if self._image is not None:
                x = npy.clip(x, 0, self._image.width)
                y = npy.clip(y, 0, self._image.height)
                w = npy.clip(w, 0, self._image.width - x)
                h = npy.clip(h, 0, self._image.height - y)

            self._rebase([x, y, w, h])

            if isinstance(regions, ROI):
                self._sub_features += regions
            elif isinstance(regions, Feature):
                self.subFeatures.append(regions)
            elif isinstance(regions, (list, tuple)):
                if isinstance(regions[0], ROI):
                    for r in regions:
                        self._sub_features += r._sub_features
                elif isinstance(regions[0], Feature):
                    for r in regions:
                        self._sub_features.append(r)

    def rebase(self, x, y=None, w=None, h=None):
        """
        Completely alter roi using whatever source coordinates you wish.
        """
        if isinstance(x, Feature):
            self._sub_features.append(x)
        elif (isinstance(x, (list, tuple)) and
                len(x) > 0 and isinstance(x, Feature)):
            self._sub_features += list(x)

        result = self._standardize(x, y, w, h)

        if result is None:
            logger.warning("Could not create an ROI from your data.")
            return

        self._rebase(result)

    def draw(self, color=Color.GREEN, width=3):
        """
        **SUMMARY**
        This method will draw the feature on the source image.
        **PARAMETERS**
        * *color* - The color as an RGB tuple to render the image.
        **RETURNS**
        Nothing.
        **EXAMPLE**
        >>> img = Image("RedDog2.jpg")
        >>> blobs = img.find_blobs()
        >>> blobs[-1].draw()
        >>> img.show()
        """
        x, y, w, h = self.to_xywh()
        self._image.drawRectangle(x,y,w,h,width=width,color=color)

    def show(self, color=Color.GREEN, width=2):
        """
        **SUMMARY**
        This function will automatically draw the features on the image and show it.
        **RETURNS**
        Nothing.
        **EXAMPLE**
        >>> img = Image("logo")
        >>> feat = img.find_blobs()
        >>> feat[-1].show()
        """
        self.draw(color, width)
        self._image.show()

    def mean_color(self):
        """
        **SUMMARY**
        Return the average color within the feature as a tuple.
        **RETURNS**
        An RGB color tuple.
        **EXAMPLE**
        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> for b in blobs:
        >>>    if b.mean_color() == Color.WHITE:
        >>>       print("Found a white thing")
        """
        x, y, w, h = self.to_xywh()
        return self._image.crop(x, y, w, h).mean_color()

    def _rebase(self, roi):
        x, y, w, h = roi
        self._max_x = None
        self._max_y = None
        self._min_x = None
        self._min_y = None
        self._width = None
        self._height = None
        self._extents = None
        self._bbox = None
        self.xtl = x
        self.ytl = y
        self.w = w
        self.h = h
        self._points = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        # WE MAY WANT TO DO A SANITY CHECK HERE
        self._update_extents()

    def _standardize(self, x, y=None, w=None, h=None):
        if isinstance(x, npy.ndarray):
            x = x.tolist()
        if isinstance(y, npy.ndarray):
            y = y.tolist()

        # make the common case fast
        if (isinstance(x, (int, float)) and isinstance(y, (int, float)) and
                isinstance(w, (int, float)) and isinstance(h, (int, float))):
            if self._image is not None:
                x = npy.clip(x, 0, self._image.width)
                y = npy.clip(y, 0, self._image.height)
                w = npy.clip(w, 0, self._image.width - x)
                h = npy.clip(h, 0, self._image.height - y)

                return [x, y, w, h]
        elif isinstance(x, ROI):
            x, y, w, h = x.to_xywh()
        # If it's a feature extract what we need
        elif isinstance(x, FeatureSet) and len(x) > 0:
            # double check that everything in the list is a feature
            features = [feat for feat in x if isinstance(feat, Feature)]
            xmax = npy.max([feat.maxX() for feat in features])
            xmin = npy.min([feat.minX() for feat in features])
            ymax = npy.max([feat.maxY() for feat in features])
            ymin = npy.min([feat.minY() for feat in features])
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin

        elif isinstance(x, Feature):
            feature = x
            x = feature.points[0][0]
            y = feature.points[0][1]
            w = feature.width()
            h = feature.height()

        # [x,y,w,h] (x,y,w,h)
        elif (isinstance(x, (tuple, list)) and len(x) == 4 and
                isinstance(x[0], (int, long, float)) and
                y is None and w is None and h is None):
            x, y, w, h = x
        # x of the form [(x,y),(x1,y1),(x2,y2),(x3,y3)]
        # x of the form [[x,y],[x1,y1],[x2,y2],[x3,y3]]
        # x of the form ([x,y],[x1,y1],[x2,y2],[x3,y3])
        # x of the form ((x,y),(x1,y1),(x2,y2),(x3,y3))
        elif (isinstance(x, (list, tuple)) and
                isinstance(x[0], (list, tuple)) and
                len(x) == 4 and len(x[0]) == 2 and
                y is None is w is None is h is None):
            if (len(x[0]) == 2 and len(x[1]) == 2 and
                    len(x[2]) == 2 and len(x[3]) == 2):
                xmax = npy.max([x[0][0], x[1][0], x[2][0], x[3][0]])
                ymax = npy.max([x[0][1], x[1][1], x[2][1], x[3][1]])
                xmin = npy.min([x[0][0], x[1][0], x[2][0], x[3][0]])
                ymin = npy.min([x[0][1], x[1][1], x[2][1], x[3][1]])
                x = xmin
                y = ymin
                w = xmax - xmin
                h = ymax - ymin
            else:
                logger.warning(
                    "x should be in the form  ((x,y),(x1,y1),(x2,y2),(x3,y3))")
                return None

        # x,y of the form [x1,x2,x3,x4,x5....] and y similar
        elif (isinstance(x, (tuple, list)) and
                isinstance(y, (tuple, list)) and
                len(x) > 4 and len(y) > 4):
            if (isinstance(x[0], (int, long, float)) and
                isinstance(y[0], (int, long, float))):
                xmax = npy.max(x)
                ymax = npy.max(y)
                xmin = npy.min(x)
                ymin = npy.min(y)
                x = xmin
                y = ymin
                w = xmax - xmin
                h = ymax - ymin
            else:
                logger.warning("x should be in the form x = [1,2,3,4,5] "
                               "y =[0,2,4,6,8]")
                return None

        # x of the form [(x,y),(x,y),(x,y),(x,y),(x,y),(x,y)]
        elif isinstance(x, (list, tuple) and len(x) > 4 and
                        len(x[0]) == 2 and y is None and
                        w is None and h is None):
            if isinstance(x[0][0], (int, long, float)):
                xs = [pt[0] for pt in x]
                ys = [pt[1] for pt in x]
                xmax = npy.max(xs)
                ymax = npy.max(ys)
                xmin = npy.min(xs)
                ymin = npy.min(ys)
                x = xmin
                y = ymin
                w = xmax - xmin
                h = ymax - ymin
            else:
                logger.warning("x should be in the form [(x,y),(x,y),(x,y),"
                               "(x,y),(x,y),(x,y)]")
                return None

        # x of the form [(x,y),(x1,y1)]
        elif (isinstance(x, (list, tuple)) and len(x) == 2 and
                isinstance(x[0], (list, tuple)) and
                isinstance(x[1], (list, tuple)) and
                y is None and w is None and h is None):
            if (len(x[0]) == 2 and len(x[1]) == 2):
                xt = npy.min([x[0][0], x[1][0]])
                yt = npy.min([x[0][0], x[1][0]])
                w = npy.abs(x[0][0] - x[1][0])
                h = npy.abs(x[0][1] - x[1][1])
                x = xt
                y = yt
            else:
                logger.warning("x should be in the form [(x1,y1),(x2,y2)]")
                return None

        # x and y of the form (x,y),(x1,y2)
        elif (isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)) and
                      w is None and h is None):
            if len(x) == 2 and len(y) == 2:
                xt = npy.min([x[0], y[0]])
                yt = npy.min([x[1], y[1]])
                w = npy.abs(y[0] - x[0])
                h = npy.abs(y[1] - x[1])
                x = xt
                y = yt

            else:
                logger.warning("if x and y are tuple it should be in the form "
                               "(x1,y1) and (x2,y2)")
                return None

        if y is None or w is None or h is None:
            logger.warning('Not a valid roi')
        elif w <= 0 or h <= 0:
            logger.warning("ROI can't have a negative dimension")
            return None

        if self._image is not None:
            x = npy.clip(x, 0, self._image.width)
            y = npy.clip(y, 0, self._image.height)
            w = npy.clip(w, 0, self._image.width - x)
            h = npy.clip(h, 0, self._image.height - y)

        return [x, y, w, h]

    def crop(self):
        ret = None
        if self._image is not None:
            ret = self._image.crop(self.xtl, self.ytl, self.w, self.h)
        return ret
