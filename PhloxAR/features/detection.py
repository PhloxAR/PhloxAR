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
