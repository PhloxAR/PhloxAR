# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals
from __future__ import division, print_function

from PhloxAR.base import *
from PhloxAR.core.color import Color

__all__ = [
    'Feature', 'FeatureSet'
]


class Feature(object):
    """
    Abstract class which real features descend from.
    Each feature object has:
    a draw() method,
    an image property, referencing the originating Image object,
    x and y coordinates
    default functions for determining angle, area, mean _color, etc.
    these functions assume the feature is 1px
    """
    _x = 0.00  # center x coordinate
    _y = 0.00  # center y coordinate
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
        Return the average _color within the feature as a tuple.
        :return: an RGB _color tuple
        """
        return self._image[self._x, self._y]

    def color_distance(self, color=(0, 0, 0)):
        """
        Return the euclidean _color distance of the _color tuple at x, y from
        a given _color (default black).
        :param color: a RGB tuple to calculate from which to calculate the
                       _color distance.
        :return: a floating point _color distance value.
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
        ret_val = False
        bounds = self._points
        if isinstance(other, Feature):  # A feature
            ret_val = True
            # this isn't completely correct - only tests if points lie
            # in poly, not edges.
            for p in other._points:
                p2 = (int(p[0]), int(p[1]))
                ret_val = self._point_inside_polygon(p2, bounds)
                if not ret_val:
                    break
        # a single point
        elif ((isinstance(other, tuple) and len(other) == 2) or
                  (isinstance(other, npy.ndarray) and other.shape[0] == 2)):
            ret_val = self._point_inside_polygon(other, bounds)

        elif isinstance(other, tuple) and len(other) == 3:  # A circle
            # assume we are in x,y, r format
            ret_val = True
            rr = other[2] * other[2]
            x = other[0]
            y = other[1]
            for p in bounds:
                tst = ((x - p[0]) * (x - p[0])) + ((y - p[1]) * (y - p[1]))
                if tst < rr:
                    ret_val = False
                    break

        elif (isinstance(other, tuple) and len(other) == 4 and
                  (isinstance(other[0], float) or isinstance(other[0], int))):
            ret_val = (self.max_x() <= other[0] + other[2] and
                       self.min_x() >= other[0] and
                       self.max_y() <= other[1] + other[3] and
                       self.min_y() >= other[1])
        elif isinstance(other, list) and len(other) >= 4:  # an arbitrary polygon
            # everything else ....
            ret_val = True
            for p in other:
                tst = self._point_inside_polygon(p, bounds)
                if (not tst):
                    ret_val = False
                    break
        else:
            logger.warning("Did not recognize.")
            return False

        return ret_val

    def overlaps(self, other):
        ret_val = False
        bounds = self._points

        if isinstance(other, Feature):  # A feature
            ret_val = True
            for p in other._points:  # this isn't completely correct - only tests if points lie in poly, not edges.
                ret_val = self._point_inside_polygon(p, bounds)
                if ret_val:
                    break

        elif ((isinstance(other, tuple) and len(other) == 2) or
                  (isinstance(other, npy.ndarray) and other.shape[0] == 2)):
            ret_val = self._point_inside_polygon(other, bounds)

        elif (isinstance(other, tuple) and len(other) == 3 and
                  not isinstance(other[0], tuple)):  # A circle
            # assume we are in x,y, r format
            ret_val = False
            rr = other[2] * other[2]
            x = other[0]
            y = other[1]
            for p in bounds:
                tst = ((x - p[0]) * (x - p[0])) + ((y - p[1]) * (y - p[1]))
                if tst < rr:
                    ret_val = True
                    break

        elif (isinstance(other, tuple) and len(other) == 4 and
                  (isinstance(other[0], float) or isinstance(other[0], int))):
            ret_val = (self.contains((other[0], other[1])) or  # see if we contain any corner
                       self.contains((other[0] + other[2], other[1])) or
                       self.contains((other[0], other[1] + other[3])) or
                       self.contains((other[0] + other[2], other[1] + other[3])))
        elif isinstance(other, list) and len(other) >= 3:  # an arbitrary polygon
            # everything else ....
            ret_val = False
            for p in other:
                tst = self._point_inside_polygon(p, bounds)
                if tst:
                    ret_val = True
                    break
        else:
            logger.warning(
                "Did not recognize.")
            return False

        return ret_val

    def is_contained_within(self, other):
        """
        Return true if the feature is contained withing  the object other,
        where other can be a bounding box, bounding circle, a list of tuples
        in a closed polygon, or any other features.
        :param other: bounding box - (x, y, w, h) where (x, y) is the top left
                       bounding circle - (x, y, r)
                       a list of (x, y) tuples defining a closed polygon
                       any two dimensional feature (e.g. blobs, circle ...
        :return: Bool
        """
        ret_val = True
        bounds = self._points

        if isinstance(other, Feature):  # another feature do the containment test
            ret_val = other.contains(self)
        elif isinstance(other, tuple) and len(other) == 3:  # a circle
            # assume we are in x,y, r format
            rr = other[2] * other[2]  # radius squared
            x = other[0]
            y = other[1]
            for p in bounds:
                tst = ((x - p[0]) * (x - p[0])) + ((y - p[1]) * (y - p[1]))
                if test > rr:
                    ret_val = False
                    break
        elif (isinstance(other, tuple) and len(other) == 4 and  # a bounding box
                  (isinstance(other[0], float) or isinstance(other[0],
                                                             int))):  # we assume a tuple of four is (x,y,w,h)
            ret_val = (self.max_x() <= other[0] + other[2] and
                       self.min_x() >= other[0] and
                       self.max_y() <= other[1] + other[3] and
                       self.min_y() >= other[1])
        elif isinstance(other, list) and len(other) > 2:  # an arbitrary polygon
            # everything else ....
            ret_val = True
            for p in bounds:
                tst = self._point_inside_polygon(p, other)
                if not tst:
                    ret_val = False
                    break
        else:
            logger.warning("Did not recognize.")
            ret_val = False
        return ret_val

    def _point_inside_polygon(self, point, polygon):
        """
        returns true if tuple point (x,y) is inside polygon of the
        form ((a,b),(c,d),...,(a,b)) the polygon should be closed
        """
        if len(polygon) < 3:
            logger.warning("feature._point_inside_polygon - not a valid polygon")
            return False

        if not isinstance(polygon, list):
            logger.warning("feature._point_inside_polygon - not a valid polygon")
            return False

        counter = 0
        ret_val = True
        p1 = None
        poly = copy.deepcopy(polygon)
        poly.append(polygon[0])
        # for p2 in poly:
        N = len(poly)
        p1 = poly[0]
        for i in range(1, N + 1):
            p2 = poly[i % N]
            if point[1] > npy.min((p1[1], p2[1])):
                if point[1] <= npy.max((p1[1], p2[1])):
                    if point[0] <= npy.max((p1[0], p2[0])):
                        if p1[1] != p2[1]:
                            tst = (float((point[1] - p1[1]) * (p2[0] - p1[0])) /
                                   float(((p2[1] - p1[1]) + p1[0])))
                            if p1[0] == p2[0] or point[0] <= tst:
                                counter += 1
            p1 = p2

        if counter % 2 == 0:
            ret_val = False
            return ret_val
        return ret_val

    def bcir(self):
        """
        Calculates the minimum bounding circle of the blob in the image as
        a (x, y, r) tuple.
        :return: a (x, y, r) tuple where (x, y) is the center of the circle
                  and r is the radius
        """
        contour = self.contour()

        points = []

        for pair in contour:
            points.append([pair[0], pair[1]])

        points = npy.array(points)

        cen, rad = cv2.minEnclosingCircle(points)

        return cen[0], cen[1], rad

    def contour(self):
        return []

    @property
    def image(self):
        return self._image

    @property
    def points(self):
        return self._points


class FeatureSet(list):
    """

    FeatureSet is a class extended from Python's list which has special functions so that it is useful for handling feature metadata on an image.
    In general, functions dealing with attributes will return numpy arrays, and functions dealing with sorting or filtering will return new FeatureSets.

    >>> image = Image("/path/to/image.png")
    >>> lines = image.find_lines()  #lines are the feature set
    >>> lines.draw()
    >>> lines.x()
    >>> lines.crop()
    """
    def __getitem__(self, key):
        """

        Returns a FeatureSet when sliced. Previously used to
        return list. Now it is possible to use FeatureSet member
        functions on sub-lists
        """
        if isinstance(key, types.SliceType):
            return FeatureSet(list.__getitem__(self, key))
        else:
            return list.__getitem__(self, key)

    def count(self, **kwargs):
        """
        This function returns the length / count of the all the items in the FeatureSet:
        """
        return len(self)

    def draw(self, color=Color.GREEN, width=1, autocolor=False, alpha=-1):
        """

        Call the draw() method on each feature in the FeatureSet.


        * *_color* - The _color to draw the object. Either an BGR tuple or a member of the :py:class:`Color` class.
        * *width* - The width to draw the feature in pixels. A value of -1 usually indicates a filled region.
        * *autocolor* - If true a _color is randomly selected for each feature.


        Nada. Nothing. Zilch.

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> feats.draw(_color=Color.PUCE, width=3)
        >>> img.show()
        """
        for f in self:
            if autocolor:
                color = Color.random()
            if alpha != -1:
                f.draw(color=color, width=width, alpha=alpha)
            else:
                f.draw(color=color, width=width)

    def show(self, color=Color.GREEN, autocolor=False, width=1):
        """

        This function will automatically draw the features on the image and show it.
        It is a basically a shortcut function for development and is the same as:

        * *_color* - The _color to draw the object. Either an BGR tuple or a member of the :py:class:`Color` class.
        * *width* - The width to draw the feature in pixels. A value of -1 usually indicates a filled region.
        * *autocolor* - If true a _color is randomly selected for each feature.

        Nada. Nothing. Zilch.

        >>> img = Image("logo")
        >>> feat = img.find_blobs()
        >>> if feat: feat.draw()
        >>> img.show()
        """
        self.draw(color, width, autocolor)
        self[-1].image.show()

    def reassign_image(self, img):
        """

        Return a new featureset where the features are assigned to a new image.

        * *img* - the new image to which to assign the feature.
        .. Warning::
          THIS DOES NOT PERFORM A SIZE CHECK. IF YOUR NEW IMAGE IS NOT THE EXACT SAME SIZE YOU WILL CERTAINLY CAUSE ERRORS.

        >>> img = Image("lenna")
        >>> img2 = img.invert()
        >>> l = img.find_lines()
        >>> l2 = img.reassign_image(img2)
        >>> l2.show()
        """
        ret = FeatureSet()
        for i in self:
            ret.append(i.reassign(img))
        return ret

    def x(self):
        """

        Returns a numpy array of the x (horizontal) coordinate of each feature.

        A numpy array.

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> xs = feats.x()
        >>> print(xs)
        """
        return npy.array([f.x for f in self])

    def y(self):
        """

        Returns a numpy array of the y (vertical) coordinate of each feature.

        A numpy array.

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> xs = feats.y()
        >>> print(xs)
        """
        return npy.array([f.y for f in self])

    def coordinates(self):
        """

        Returns a 2d numpy array of the x,y coordinates of each feature.  This
        is particularly useful if you want to use Scipy's Spatial Distance module

        A numpy array of all the positions in the featureset.

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> xs = feats.coordinates()
        >>> print(xs)
        """
        return npy.array([[f.x, f.y] for f in self])

    @property
    def center(self):
        return self.coordinates()

    @property
    def area(self):
        """

        Returns a numpy array of the area of each feature in pixels.

        A numpy array of all the positions in the featureset.

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> xs = feats.area()
        >>> print(xs)
        """
        return npy.array([f.area() for f in self])

    def sort_area(self):
        """

        Returns a new FeatureSet, with the largest area features first.

        A featureset sorted based on area.

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> feats = feats.sort_area()
        >>> print feats[-1] # biggest blob
        >>> print feats[0] # smallest blob
        """
        return FeatureSet(sorted(self, key=lambda f: f.area()))

    def sort_x(self):
        """

        Returns a new FeatureSet, with the smallest x coordinates features first.

        A featureset sorted based on area.

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> feats = feats.sort_x()
        >>> print(feats[-1]) # biggest blob
        >>> print(feats[0]) # smallest blob
        """
        return FeatureSet(sorted(self, key=lambda f: f.x))

    def sort_y(self):
        """

        Returns a new FeatureSet, with the smallest y coordinates features first.

        A featureset sorted based on area.

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> feats = feats.sortY()
        >>> print(feats[-1]) # biggest blob
        >>> print(feats[0]) # smallest blob
        """
        return FeatureSet(sorted(self, key=lambda f: f.y))

    def distance_from(self, point=(-1, -1)):
        """

        Returns a numpy array of the distance each Feature is from a given coordinate.
        Default is the center of the image.

        * *point* - A point on the image from which we will calculate distance.

        A numpy array of distance values.

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> d = feats.distance_from()
        >>> d[0]  #show the 0th blobs distance to the center.

        Make this accept other features to measure from.
        """
        if point[0] == -1 or point[1] == -1 and len(self):
            point = self[0].image.size()

        return spsd.cdist(self.coordinates(), [point])[:, 0]

    def sort_distance(self, point=(-1, -1)):
        """

        Returns a sorted FeatureSet with the features closest to a given coordinate first.
        Default is from the center of the image.

        * *point* - A point on the image from which we will calculate distance.

        A numpy array of distance values.

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> d = feats.sort_distance()
        >>> d[-1].show()  #show the 0th blobs distance to the center.
        """
        return FeatureSet(sorted(self, key=lambda f: f.distance_from(point)))

    def distance_pairs(self):
        """

        Returns the square-form of pairwise distances for the featureset.
        The resulting N x N array can be used to quickly look up distances
        between features.

        A NxN np matrix of distance values.

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> d = feats.distance_pairs()
        >>> print(d)
        """
        return spsd.squareform(spsd.pdist(self.coordinates()))

    def angle(self):
        """

        Return a numpy array of the angles (theta) of each feature.
        Note that theta is given in degrees, with 0 being horizontal.

        An array of angle values corresponding to the features.

        >>> img = Image("lenna")
        >>> l = img.find_lines()
        >>> angs = l.angle()
        >>> print angs
        """
        return npy.array([f.angle() for f in self])

    def sort_angle(self, theta=0):
        """
        Return a sorted FeatureSet with the features closest to a given angle first.
        Note that theta is given in radians, with 0 being horizontal.

        An array of angle values corresponding to the features.

        >>> img = Image("lenna")
        >>> l = img.find_lines()
        >>> l = l.sort_angle()
        >>> print(l)
        """
        return FeatureSet(sorted(self, key=lambda f: abs(f.angle() - theta)))

    def length(self):
        """

        Return a numpy array of the length (longest dimension) of each feature.

        A numpy array of the length, in pixels, of eatch feature object.

        >>> img = Image("Lenna")
        >>> l = img.find_lines()
        >>> lengt = l.length()
        >>> lengt[0] # length of the 0th element.
        """
        return npy.array([f.length() for f in self])

    def sort_length(self):
        """

        Return a sorted FeatureSet with the longest features first.

        A sorted FeatureSet.

        >>> img = Image("Lenna")
        >>> l = img.find_lines().sort_length()
        >>> lengt[-1] # length of the 0th element.
        """
        return FeatureSet(sorted(self, key=lambda f: f.length()))

    def mean_color(self):
        """

        Return a numpy array of the average _color of the area covered by each Feature.

        Returns an array of RGB triplets the correspond to the mean _color of the feature.

        >>> img = Image("lenna")
        >>> kp = img.find_keypoints()
        >>> c = kp.mean_color()
        """
        return npy.array([f.meanColor() for f in self])

    def color_distance(self, color=(0, 0, 0)):
        """

        Return a numpy array of the distance each features average _color is from
        a given _color tuple (default black, so _delta() returns intensity)

        * *_color* - The _color to calculate the distance from.

        The distance of the average _color for the feature from given _color as a numpy array.

        >>> img = Image("lenna")
        >>> circs = img.find_circle()
        >>> d = circs._delta(_color=Color.BLUE)
        >>> print d
        """
        return spsd.cdist(self.mean_color(), [color])[:, 0]

    def sort_color_distance(self, color=(0, 0, 0)):
        """
        Return a sorted FeatureSet with features closest to a given _color first.
        Default is black, so sort_color_distance() will return darkest to brightest
        """
        return FeatureSet(sorted(self, key=lambda f: f.color_distance(color)))

    def filter(self, filterarray):
        """

        Return a FeatureSet which is filtered on a numpy boolean array.  This
        will let you use the attribute functions to easily screen Features out
        of return FeatureSets.

        * *filterarray* - A numpy array, matching  the size of the feature set,
          made of Boolean values, we return the true values and reject the False value.

        The revised feature set.

        Return all lines < 200px
        >>> my_lines.filter(my_lines.length() < 200) # returns all lines < 200px
        >>> my_blobs.filter(my_blobs.area() > 0.9 * my_blobs.length**2) # returns blobs that are nearly square
        >>> my_lines.filter(abs(my_lines.angle()) < numpy.pi / 4) #any lines within 45 degrees of horizontal
        >>> my_corners.filter(my_corners.x() - my_corners.y() > 0) #only return corners in the upper diagonal of the image
        """
        return FeatureSet(list(npy.array(self)[npy.array(filterarray)]))

    def width(self):
        """

        Returns a nparray which is the width of all the objects in the FeatureSet.

        A numpy array of width values.

        >>> img = Image("NotLenna")
        >>> l = img.find_lines()
        >>> l.width()
        """
        return npy.array([f.width for f in self])

    def height(self):
        """
        Returns a nparray which is the height of all the objects in the FeatureSet

        A numpy array of width values.

        >>> img = Image("NotLenna")
        >>> l = img.find_lines()
        >>> l.height()
        """
        return npy.array([f.height for f in self])

    def crop(self):
        """

        Returns a nparray with the cropped features as SimpleCV image.

        A SimpleCV image cropped to each image.

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs(128)
        >>> for b in blobs:
        >>>   newImg = b.crop()
        >>>   newImg.show()
        >>>   time.sleep(1)
        """
        return npy.array([f.crop() for f in self])

    def inside(self, region):
        """

        Return only the features inside the region. where region can be a bounding box,
        bounding circle, a list of tuples in a closed polygon, or any other featutres.

        * *region*
          * A bounding box - of the form (x,y,w,h) where x,y is the upper left corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        Returns a featureset of features that are inside the region.

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[-1]
        >>> lines = img.find_lines()
        >>> inside = lines.inside(b)

        This currently performs a bounding box test, not a full polygon test for speed.
        """
        fs = FeatureSet()
        for f in self:
            if f.is_contained_within(region):
                fs.append(f)
        return fs

    def outside(self, region):
        """

        Return only the features outside the region. where region can be a bounding box,
        bounding circle, a list of tuples in a closed polygon, or any other featutres.

        * *region*
          * A bounding box - of the form (x,y,w,h) where x,y is the upper left corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        Returns a featureset of features that are outside the region.

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[-1]
        >>> lines = img.find_lines()
        >>> outside = lines.outside(b)

        This currently performs a bounding box test, not a full polygon test for speed.
        """
        fs = FeatureSet()
        for f in self:
            if not f.is_contained_within():
                fs.append(f)
        return fs

    def overlaps(self, region):
        """

        Return only the features that overlap or the region. Where region can be a bounding box,
        bounding circle, a list of tuples in a closed polygon, or any other featutres.

        * *region*
          * A bounding box - of the form (x,y,w,h) where x,y is the upper left corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        Returns a featureset of features that overlap the region.

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[-1]
        >>> lines = img.find_lines()
        >>> outside = lines.overlaps(b)

        This currently performs a bounding box test, not a full polygon test
        for speed.
        """
        fs = FeatureSet()
        for f in self:
            if f.overlaps(region):
                fs.append(f)
        return fs

    def above(self, region):
        """

        Return only the features that are above a  region. Where region can be
        a bounding box, bounding circle, a list of tuples in a closed polygon,
        or any other featutres.

        * *region*
          * A bounding box - of the form (x,y,w,h) where x,y is the upper left corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        Returns a featureset of features that are above the region.

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[-1]
        >>> lines = img.find_lines()
        >>> outside = lines.above(b)

        This currently performs a bounding box test, not a full polygon test
        for speed.
        """
        fs = FeatureSet()
        for f in self:
            if f.above(region):
                fs.append(f)
        return fs

    def below(self, region):
        """

        Return only the features below the region. where region can be a
        bounding box, bounding circle, a list of tuples in a closed polygon,
        or any other featutres.

        * *region*
          * A bounding box - of the form (x,y,w,h) where x,y is the upper left
                             corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon e.g. ((x,y),(x,y),..)
          * Any two dimensional feature (e.g. blobs, circle ...)

        Returns a featureset of features that are below the region.

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[-1]
        >>> lines = img.find_lines()
        >>> inside = lines.below(b)

        This currently performs a bounding box test, not a full polygon test for speed.
        """
        fs = FeatureSet()
        for f in self:
            if f.below(region):
                fs.append(f)
        return fs

    def left(self, region):
        """
        Return only the features left of the region. where region can be a
        bounding box, bounding circle, a list of tuples in a closed polygon,
        or any other featutres.

        * *region*
          * A bounding box - of the form (x,y,w,h) where x,y is the upper left corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        Returns a featureset of features that are left of the region.

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[-1]
        >>> lines = img.find_lines()
        >>> left = lines.left(b)

        This currently performs a bounding box test, not a full polygon test for speed.
        """
        fs = FeatureSet()
        for f in self:
            if f.left(region):
                fs.append(f)
        return fs

    def right(self, region):
        """

        Return only the features right of the region. where region can be a bounding box,
        bounding circle, a list of tuples in a closed polygon, or any other featutres.

        * *region*
          * A bounding box - of the form (x,y,w,h) where x,y is the upper left corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        Returns a featureset of features that are right of the region.

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[-1]
        >>> lines = img.find_lines()
        >>> right = lines.right(b)

        This currently performs a bounding box test, not a full polygon test for speed.
        """
        fs = FeatureSet()
        for f in self:
            if f.right(region):
                fs.append(f)
        return fs

    def on_image_edge(self, tolerance=1):
        """
        The method returns a feature set of features that are on or "near" the
        edge of the image. This is really helpful for removing features that are
        edge effects.

        * *tolerance* - the distance in pixels from the edge at which a feature
          qualifies as being "on" the edge of the image.

        Returns a featureset of features that are on the edge of the image.

        >>> img = Image("./sampleimages/EdgeTest1.png")
        >>> blobs = img.find_blobs()
        >>> es = blobs.on_image_edge()
        >>> es.draw(_color=Color.RED)
        >>> img.show()
        """
        fs = FeatureSet()
        for f in self:
            if f.on_image_edge(tolerance):
                fs.append(f)
        return fs

    def top_left_corners(self):
        """

        This method returns the top left corner of each feature's bounding box.

        A numpy array of x,y position values.

        >>> img = Image("./sampleimages/EdgeTest1.png")
        >>> blobs = img.find_blobs()
        >>> tl = img.top_left_corners()
        >>> print(tl[0])
        """
        return npy.array([f.top_left_corner() for f in self])

    def bottom_left_corners(self):
        """

        This method returns the bottom left corner of each feature's bounding box.

        A numpy array of x,y position values.

        >>> img = Image("./sampleimages/EdgeTest1.png")
        >>> blobs = img.find_blobs()
        >>> bl = img.bottom_left_corners()
        >>> print(bl[0])
        """
        return npy.array([f.bottom_left_corner() for f in self])

    def top_left_corners(self):
        """
        This method returns the top left corner of each feature's bounding box.

        A numpy array of x,y position values.

        >>> img = Image("./sampleimages/EdgeTest1.png")
        >>> blobs = img.find_blobs()
        >>> tl = img.bottom_left_corners()
        >>> print(tl[0])
        """
        return npy.array([f.topLeftCorner() for f in self])

    def top_right_corners(self):
        """

        This method returns the top right corner of each feature's bounding box.

        A numpy array of x,y position values.

        >>> img = Image("./sampleimages/EdgeTest1.png")
        >>> blobs = img.find_blobs()
        >>> tr = img.top_right_corners()
        >>> print(tr[0])
        """
        return npy.array([f.topRightCorner() for f in self])

    def bottom_right_corners(self):
        """

        This method returns the bottom right corner of each feature's bounding box.

        A numpy array of x,y position values.

        >>> img = Image("./sampleimages/EdgeTest1.png")
        >>> blobs = img.find_blobs()
        >>> br = img.bottom_right_corners()
        >>> print(br[0])
        """
        return npy.array([f.bottomRightCorner() for f in self])

    def aspect_ratio(self):
        """

        Return the aspect ratio of all the features in the feature set, For our purposes
        aspect ration is max(width,height)/min(width,height).

        A numpy array of the aspect ratio of the features in the featureset.

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> print blobs.aspect_ratio()
        """
        return npy.array([f.aspect_ratio() for f in self])

    def cluster(self, method="kmeans", properties=None, k=3):
        """
        This function clusters the blobs in the featureSet based on the properties.
        Properties can be "_color", "shape" or "position" of blobs.
        Clustering is done using K-Means or Hierarchical clustering(Ward) algorithm.
        

        * *properties* - It should be a list with any combination of "_color", "shape", "position". properties = ["_color","position"]. properties = ["position","shape"]. properties = ["shape"]
        * *method* - if method is "kmeans", it will cluster using K-Means algorithm, if the method is "hierarchical", no need to spicify the number of clusters
        * *k* - The number of clusters(kmeans).

        
        A list of featureset, each being a cluster itself.
        
          >>> img = Image("lenna")
          >>> blobs = img.find_blobs()
          >>> clusters = blobs.cluster(method="kmeans",properties=[__color],k=5)
          >>> for i in clusters:
          >>>     i.draw(c_colorColor.random(),width=5)
          >>> img.show()

        """
        try:
            from sklearn.cluster import KMeans, Ward
            from sklearn import __version__
        except:
            logger.warning("install scikits-learning package")
            return
        X = []  # List of feature vector of each blob
        if not properties:
            properties = ['c_color, 'shape', 'position']
        if k > len(self):
            logger.warning(
                "Number of clusters cannot be greater then the number of blobs in the featureset")
            return
        for i in self:
            featureVector = []
            if 'c_color in properties:
                featureVector.extend(i.mAvgColor)
            if 'shape' in properties:
                featureVector.extend(i.mHu)
            if 'position' in properties:
                featureVector.extend(i.extents())
            if not featureVector:
                logger.warning("properties parameter is not specified properly")
                return
            X.append(featureVector)

        if method == "kmeans":

            # Ignore minor version numbers.
            sklearn_version = re.search(r'\d+\.\d+', __version__).group()

            if (float(sklearn_version) > 0.11):
                k_means = KMeans(init='random', n_clusters=k, n_init=10).fit(X)
            else:
                k_means = KMeans(init='random', k=k, n_init=10).fit(X)
            KClusters = [FeatureSet([]) for i in range(k)]
            for i in range(len(self)):
                KClusters[k_means.labels_[i]].append(self[i])
            return KClusters

        if method == "hierarchical":
            ward = Ward(n_clusters=int(sqrt(len(self)))).fit(
                X)  # n_clusters = sqrt(n)
            WClusters = [FeatureSet([]) for i in range(int(sqrt(len(self))))]
            for i in range(len(self)):
                WClusters[ward.labels_[i]].append(self[i])
            return WClusters

    @property
    def image(self):
        if not len(self):
            return None
        return self[0].image

    @image.setter
    def image(self, i):
        for f in self:
            f.image = i
