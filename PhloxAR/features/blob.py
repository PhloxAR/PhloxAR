# -*- coding:utf-8 -*-
from __future__ import absolute_import, unicode_literals
from __future__ import division, print_function

import re
import warnings
import scipy.stats as sps
import scipy.spatial.distance as spsd
from ..core import Color, Image
from ..base import cv2, np, lazy_property
from .detection import Corner, Line, ShapeContextDescriptor
from .feature import Feature, FeatureSet

__all__ = [
    'Blob', 'BlobMaker'
]


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
        self._scdescriptors = None
        self._contour = []
        self._convex_hull = []
        self._min_rect = [-1, -1, -1, -1, -1]
        self._contour_appx = []
        self._hu = [-1, -1, -1, -1, -1, -1, -1]
        self._perimeter = 0
        self._area = 0
        self.m00 = self.m01 = self.m02 = 0
        self.m10 = self.m11 = self.m12 = 0
        self.m20 = self.m21 = 0
        self._label = 'UNASSIGNED'
        self._label_color = []
        self._avg_color = [-1, -1, -1]
        self._hole_contour = []
        super(Blob, self).__init__(None, -1, -1, [])
        # TODO
        # I would like to clean up the Hull mask parameters
        # it seems to me that we may want the convex hull to be
        # the default way we calculate for area.
    
    def __getstate__(self):
        skip = self.pickle_skip_properties
        tmp_dict = {}
        for k, v in self.__dict__.items():
            if k in skip:
                continue
            else:
                tmp_dict[k] = v

        return tmp_dict

    def __setstate__(self, state):
        keys = []
        for k in state:
            if re.search('__string', k):
                keys.append(k)
            else:
                self.__dict__[k] = state[k]

        for k in keys:
            key = k[:-len('__string')]
            self.__dict__[key] = np.zeros((self.width, self.height), np.uint8)
            self.__dict__[key] = state[k]

    @property
    def perimeter(self):
        """
        Returns the perimeter as an integer number of pixel length.
        :return: integer

        :Example:
        >>> img = Image('lena')
        >>> blobs = img.find_blobs()
        >>> print(blobs[-1].perimeter)
        """
        return self._perimeter

    @property
    def hull(self):
        """
        Returns the convex hull points as a list of x, y tuples
        :return: a list of (x, y) tuples

        :Example:
        >>> img = Image('lena')
        >>> blobs = img.find_blobs()
        >>> print(blobs[-1].hull)
        """
        return self._convex_hull

    @property
    def contour(self):
        """
        Returns the contour points as a list of (x, y) tuples.
        :return: a list of (x, y) tuples

        :Example:
        >>> img = Image('lena')
        >>> blobs = img.find_blobs()
        >>> print(blobs[-1].contour)
        """
        return self._contour

    @property
    def mean_color(self):
        """
        Returns a tuple representing the average color of the blob.
        :return: a RGB triplet of the average blob colors.

        :Example:
        >>> img = Image('lena')
        >>> blobs = img.find_blobs()
        >>> print(blobs[-1].mean_color)
        """
        hack = (self._bbox[0], self._bbox[1], self._bbox[2], self._bbox[3])
        hack_img = cv.SetImageROI(self._image.narray, hack)
        # may need the offset parameter
        avg = cv2.mean(self._image.narray, self._mask._get_gray_narray())
        cv.ResetImageROI(self._image.narray)

        return tuple(reversed(avg[0:3]))

    @property
    def area(self):
        """
        Returns the area of the blob in terms of the number of pixels
        inside the contour
        :return: an integer of the area of the blob in pixels

        :Example:
        >>> img = Image('lena')
        >>> blobs = img.find_blobs()
        >>> print(blobs[-1].area)
        >>> print(blobs[0].area)
        """
        return self._area

    @property
    def min_rect(self):
        """
        Returns the corners for the smallest rotated rectangle to enclose
        the blob. The points are returned as a list of (x, y) tuples.
        """
        ang = self._min_rect[2]
        ang = np.pi*(float(ang) / 180)
        tx = self.min_rect_x()
        ty = self.min_rect_y()
        w = self.min_rect_width() / 2.0
        h = self.min_rect_height() / 2.0

        # [ cos a , -sin a, tx ]
        # [ sin a , cos a , ty ]
        # [ 0     , 0     ,  1 ]
        derp = np.matrix([
            [np.cos(ang), -np.sin(ang), tx],
            [np.sin(ang), np.cos(ang), ty],
            [0, 0, 1]
        ])

        tl = np.matrix([-w, h, 1.0])  # kat gladly supports homo. coord
        tr = np.matrix([w, h, 1.0])
        bl = np.matrix([-w, -h, 1.0])
        br = np.matrix([w, -h, 1.0])
        tlp = derp * tl.T
        trp = derp * tr.T
        blp = derp * bl.T
        brp = derp * br.T

        return ((float(tlp[0, 0]), float(tlp[1, 0])),
                (float(trp[0, 0]), float(trp[1, 0])),
                (float(blp[0, 0]), float(blp[1, 0])),
                (float(brp[0, 0]), float(brp[1, 0])))

    @property
    def width(self):
        return

    @property
    def height(self):
        return

    @property
    def avg_color(self):
        return self._avg_color

    def draw_rect(self, layer=None, color=Color.DEFAULT, width=1, alpha=128):
        """
        Draws the bounding rectangle for the blob.

        :param layer: if layer is not None, the blob is rendered to the layer
                       versus the source image.
        :param color: the color to render the blob's box
        :param width: the width of the drawn blob in pixels
        :param alpha: the alpha value of the rendered blob
                       0 = transparent, 255 = opaque
        :return: None

        :Example:
        >>> img = Image('lena')
        >>> blobs = img.find_blobs()
        >>> blobs[-1].draw_rect(color=Color.RED, width=-1, alpha=128)
        >>> img.show()
        """
        if layer is None:
            layer = self._image.dl()

        if width < 1:
            layer.rectangle(self.top_left_corner(), (self.width, self.height),
                            color, width, filled=True, alpha=alpha)
        else:
            layer.rectangle(self.top_left_corner(), (self.width, self.height),
                            color, width, filled=False, alpha=alpha)

    def draw_min_rect(self, layer=None, color=Color.DEFAULT, width=1, alpha=128):
        """
        Draws the minimum bounding rectangle for the blob.

        :param layer: if layer is not None, the blob is rendered to the layer
                       versus the source image.
        :param color: the color to render the blob's box
        :param width: the width of the drawn blob in pixels
        :param alpha: the alpha value of the rendered blob
                       0 = transparent, 255 = opaque
        :return: None
        """
        if layer is None:
            layer = self._image.dl()

        tl, tr, bl, br = self.min_rect
        layer.line(tl, tr, color, width=width, alpha=alpha, antialias=False)
        layer.line(bl, br, color, width=width, alpha=alpha, antialias=False)
        layer.line(tl, bl, color, width=width, alpha=alpha, antialias=False)
        layer.line(tr, br, color, width=width, alpha=alpha, antialias=False)

    @property
    def angle(self):
        """
        Returns the angle between the horizontal and the minimum enclosing
        rectangle of the blob. The minimum enclosing rectangle IS NOT not the
        bounding box. Use the bounding box for situations where you need only
        an approximation of the objects dimensions. The minimum enclosing
        rectangle is slightly harder to manipulate but gives much better
        information about the blobs dimensions.

        :Example:
        >>> img = Image('lena')
        >>> blobs = img.find_blobs()
        >>> blobs[-1].angle
        """
        ret = 0.00
        if self._min_rect[1][0] < self._min_rect[1][1]:
            ret = self._min_rect[2]
        else:
            ret = 90.00 + self._min_rect[2]

        ret += 90.00

        if ret > 90.00:
            ret -= 180.00

        return ret

    @property
    def min_rect_x(self):
        """
        X coordinate of the centroid for the minimum bounding rectangle.
        :return: an integer of the x position of the centroid of the minimum
                  bounding rectangle.

        :Example:
        >>> img = Image('lena')
        >>> blobs = img.find_blobs()
        >>> print(blobs[-1].min_rect_x)
        """
        return self._min_rect[0][0]

    @property
    def min_rect_y(self):
        """
        Y coordinate of the centroid for the minimum bounding rectangle.
        :return: an integer of the x position of the centroid of the minimum
                  bounding rectangle.

        :Example:
        >>> img = Image('lena')
        >>> blobs = img.find_blobs()
        >>> print(blobs[-1].min_rect_y)
        """
        return self._min_rect[0][1]

    @property
    def min_rect_width(self):
        """
        Return the width of the minimum bounding rectangle for the blob
        """
        # TODO: examples
        return self._min_rect[1][0]

    @property
    def min_rect_height(self):
        """
        Return the height of the minimum bounding rectangle for the blob
        """
        # TODO: examples
        return self._min_rect[1][1]

    def rotate(self, angle):
        """
        Rotate the blob given an  angle in degrees. If you use this method
        most of the blob elements will be rotated in place , however, this
        will "break" drawing back to the original image. To draw the blob
        create a new layer and draw to that layer. Positive rotations are
        counter clockwise.

        :param angle: a floating point angle in degree. Positive is
        anti-clockwise.

        :return: None
        """
        # TODO: examples
        # FIXME: This function should return a blob
        theta = np.pi * (angle / 180.0)
        mode = ""
        point = (self.x, self.y)
        self._image = self._image.rotate(angle, mode, point)
        self._hull_img = self._hull_img.rotate(angle, mode, point)
        self._mask = self._mask.rotate(angle, mode, point)
        self._hull_mask = self._hull_mask.rotate(angle, mode, point)

        self._contour = map(
                lambda x: (
                    x[0] * np.cos(theta) - x[1] * np.sin(theta),
                    x[0] * np.sin(theta) + x[1] * np.cos(theta)
                ),
                self._contour
        )
        self._convex_hull = map(
                lambda x: (
                    x[0] * np.cos(theta) - x[1] * np.sin(theta),
                    x[0] * np.sin(theta) + x[1] * np.cos(theta)
                ),
                self._convex_hull
        )

        if self._hole_contour is not None:
            for h in self._hole_contour:
                h = map(
                        lambda x: (
                            x[0] * np.cos(theta) - x[1] * np.sin(theta),
                            x[0] * np.sin(theta) + x[1] * np.cos(theta)),
                        h
                )

    def draw_apps(self, color=Color.HOTPINK, width=-1, alpha=-1, layer=None):
        if self._contour_appx is None or len(self._contour_appx) == 0:
            return

        if not layer:
            layer = self._image.dl()

        if width < 1:
            layer.polygon(self._contour_appx, color, width, True, True, alpha)
        else:
            layer.polygon(self._contour_appx, color, width, False, True, alpha)

    def draw(self, color=Color.GREEN, width=-1, alpha=-1, layer=None):
        """
        Draw the blob, in the given color, to the appropriate layer By default,
        this draws the entire blob filled in, with holes.  If you provide a
        width, an outline of the exterior and interior contours is drawn.

        :param color: the color to render the blob as a color tuple.
        :param alpha: the alpha value of the rendered blob
                       0 = transparent, 255 = opaque.
        :param width: the width of the drawn blob in pixels, if -1 then filled
                       then the polygon is filled.
        :param layer: source layer, if layer is not None, the blob is rendered
                      to the layer versus the source image.

        :return:
        This method either works on the original source image, or on the drawing
        layer provided. The method does not modify object itself.

        :Example:
        >>> img = Image("lena")
        >>> blobs = img.find_blobs()
        >>> blobs[-2].draw(color=Color.PUCE,width=-1,alpha=128)
        >>> img.show()
        """
        if not layer:
            layer = self._image.dl()

        if width == -1:
            # copy the mask into 3 channels and multiply by the appropriate color
            maskred = np.zeros((self._mask._get_gray_narray().width,
                                self._mask._get_gray_narray().height),
                               np.uint8)

            maskgrn = np.zeros((self._mask._get_gray_narray().width,
                                self._mask._get_gray_narray().height),
                               np.uint8)

            maskblu = np.zeros((self._mask._get_gray_narray().width,
                                self._mask._get_gray_narray().height),
                               np.uint8)

            maskbit = np.zeros((self._mask._get_gray_narray().width,
                                self._mask._get_gray_narray().height, 3),
                               np.uint8)

            cv.ConvertScale(self._mask._get_gray_narray(), maskred,
                            color[0] / 255.0)
            cv.ConvertScale(self._mask._get_gray_narray(), maskgrn,
                            color[1] / 255.0)
            cv.ConvertScale(self._mask._get_gray_narray(), maskblu,
                            color[2] / 255.0)

            cv2.merge((maskblu, maskgrn, maskred), maskbit)

            masksurface = Image(maskbit).surface
            masksurface.set_colorkey(Color.BLACK)
            if alpha != -1:
                masksurface.set_alpha(alpha)
            layer._surface.blit(masksurface, self.top_left_corner())  # KAT HERE
        else:
            self.draw_outline(color, alpha, width, layer)
            self.draw_holes(color, alpha, width, layer)

    def draw_outline(self, color=Color.GREEN, alpha=255, width=1, layer=None):
        """
        Draw the blob contour the provided layer -- if no layer is provided,
        draw to the source image.
        
        :param color: The color to render the blob.
        :param alpha: The alpha value of the rendered poly.
        :param width: The width of the drawn blob in pixels, -1 then the polygon is filled.
        :param layer: if layer is not None, the blob is rendered to the layer versus the source image.
        :return:
        This method either works on the original source image, or on the drawing layer provided.
        The method does not modify object itself.
        
        :Example:
        >>> img = Image("lena")
        >>> blobs = img.find_blobs()
        >>> blobs[-2].drawOutline(color=Color.GREEN,width=3,alpha=128)
        >>> img.show()
        """

        if layer is None:
            layer = self._image.dl()

        if width < 0:
            # blit the blob in
            layer.polygon(self._contour, color, filled=True, alpha=alpha)
        else:
            lastp = self._contour[0]  # this may work better.... than the other
            for nextp in self._contour[1::]:
                layer.line(lastp, nextp, color, width=width, alpha=alpha,
                           antialias=False)
                lastp = nextp
            layer.line(self._contour[0], self._contour[-1], color, width=width,
                       alpha=alpha, antialias=False)

    def draw_holes(self, color=Color.GREEN, alpha=-1, width=-1, layer=None):
        """
        This method renders all of the holes (if any) that are present in the blob.
        
        :param color* - The color to render the blob's holes.
        :param alpha* - The alpha value of the rendered blob hole.
        :param width* - The width of the drawn blob hole in pixels, if w=-1 then the polygon is filled.
        :param layer* - If layer is not None, the blob is rendered to the layer versus the source image.
        :return:
        This method either works on the original source image, or on the drawing layer provided.
        The method does not modify object itself.
        
        :Example:
        >>> img = Image("lena")
        >>> blobs = img.find_blobs(128)
        >>> blobs[-1].draw_holes(color=Color.GREEN,width=3,alpha=128)
        >>> img.show()
        """
        if self._hole_contour is None:
            return
        if layer is None:
            layer = self._image.dl()

        if width < 0:
            # blit the blob in
            for h in self._hole_contour:
                layer.polygon(h, color, filled=True, alpha=alpha)
        else:
            for h in self._hole_contour:
                lastp = h[0]  # this may work better.... than the other
                for nextp in h[1::]:
                    layer.line((int(lastp[0]), int(lastp[1])),
                               (int(nextp[0]), int(nextp[1])), color,
                               width=width, alpha=alpha, antialias=False)
                    lastp = nextp
                layer.line(h[0], h[-1], color, width=width, alpha=alpha,
                           antialias=False)

    def draw_hull(self, color=Color.GREEN, alpha=-1, width=-1, layer=None):
        """
        
        Draw the blob's convex hull to either the source image or to the
        specified layer given by layer.
        
        :param color* - The color to render the blob's convex hull as an RGB triplet.
        :param alpha* - The alpha value of the rendered blob.
        :param width* - The width of the drawn blob in pixels, if w=-1 then the polygon is filled.
        :param layer* - if layer is not None, the blob is rendered to the layer versus the source image.
        :return:
        This method either works on the original source image, or on the drawing layer provided.
        The method does not modify object itself.
        :Example:
        >>> img = Image("lena")
        >>> blobs = img.find_blobs(128)
        >>> blobs[-1].draw_hull(color=Color.GREEN,width=3,alpha=128)
        >>> img.show()
        """
        if layer is None:
            layer = self._image.dl()

        if width < 0:
            # blit the blob in
            layer.polygon(self._convex_hull, color, filled=True, alpha=alpha)
        else:
            # this may work better.... than the other
            lastp = self._convex_hull[0]
            for nextp in self._convex_hull[1::]:
                layer.line(lastp, nextp, color, width=width, alpha=alpha,
                           antialias=False)
                lastp = nextp
            layer.line(self._convex_hull[0], self._convex_hull[-1], color,
                       width=width, alpha=alpha, antialias=False)

    # draw the actual pixels inside the contour to the layer
    def draw_mask2layer(self, layer=None, offset=(0, 0)):
        """
        
        Draw the actual pixels of the blob to another layer. This is handy if you
        want to examine just the pixels inside the contour.
        
        :param layer* - A drawing layer upon which to apply the mask.
        :param offset* -  The offset from the top left corner where we want to place the mask.
        :return:
        This method either works on the original source image, or on the drawing layer provided.
        The method does not modify object itself.
        :Example:
        >>> img = Image("lena")
        >>> blobs = img.find_blobs(128)
        >>> dl = DrawingLayer((img.width,img.height))
        >>> blobs[-1].draw_mask2layer(layer = dl)
        >>> dl.show()
        """
        if layer is not None:
            layer = self._image.dl()

        mx = self._bbox[0] + offset[0]
        my = self._bbox[1] + offset[1]
        layer.blit(self.image, coordinates=(mx, my))
        return None

    def is_square(self, tolerance=0.05, ratiotolerance=0.05):
        """
        
        Given a tolerance, test if the blob is a rectangle, and how close its
        bounding rectangle's aspect ratio is to 1.0.
        
        :param tolerance* - A percentage difference between an ideal rectangle and our hull mask.
        :param ratiotolerance* - A percentage difference of the aspect ratio of our blob and an ideal square.
        :return:
        Boolean True if our object falls within tolerance, false otherwise.

        :Example:
        >>> img = Image("lena")
        >>> blobs = img.find_blobs(128)
        >>> if blobs[-1].is_square():
        >>>     print "it is hip to be square."
        """
        if self.is_rectangle(tolerance) and abs(
                        1 - self.aspect_ratio()) < ratiotolerance:
            return True
        return False

    def is_rectangle(self, tolerance=0.05):
        """
        
        Given a tolerance, test the blob against the rectangle distance to see if
        it is rectangular.
        
        :param tolerance* - The percentage difference between our blob and its idealized bounding box.
        :return:
        Boolean True if the blob is withing the rectangle tolerage, false otherwise.
        :Example:
        >>> img = Image("lena")
        >>> blobs = img.find_blobs(128)
        >>> if blobs[-1].is_rectangle():
        >>>     print "it is hip to be square."
        """
        if self.rectangle_distance() < tolerance:
            return True
        return False

    def rectangle_distance(self):
        """
        
        This compares the hull mask to the bounding rectangle.  Returns the area
        of the blob's hull as a fraction of the bounding rectangle.
        :return:
        The number of pixels in the blobs hull mask over the number of pixels in its bounding box.
        """
        blackcount, whitecount = self._hull_mask.histogram(2)
        return (abs(1.0 - float(whitecount) /
                    (self.min_rect_width() * self.min_rect_height())))

    def is_circle(self, tolerance=0.05):
        """
        
        Test circle distance against a tolerance to see if the blob is circlular.
        
        :param tolerance* - the percentage difference between our blob and an ideal circle.
        :return:
        True if the feature is within tolerance for being a circle, false otherwise.
        """
        if self.circle_distance() < tolerance:
            return True
        return False

    def circle_distance(self):
        """
        
        Compare the hull mask to an ideal circle and count the number of pixels
        that deviate as a fraction of total area of the ideal circle.
        :return:
        The difference, as a percentage, between the hull of our blob and an idealized
        circle of our blob.
        """
        w = self._hull_mask.width
        h = self._hull_mask.height

        idealcircle = Image((w, h))
        radius = min(w, h) / 2
        idealcircle.dl().circle((w / 2, h / 2), radius, filled=True,
                                color=Color.WHITE)
        idealcircle = idealcircle.apply_layers()
        netdiff = (idealcircle - self._hull_mask) + (
        self._hull_mask - idealcircle)
        numblack, numwhite = netdiff.histogram(2)
        return float(numwhite) / (radius * radius * np.pi)

    @property
    def centroid(self):
        """
        Return the centroid (mass-determined center) of the blob. Note that this is differnt from the bounding box center.
        :return:
        An (x,y) tuple that is the center of mass of the blob.
        :Example:
        >>> img = Image("lena")
        >>> blobs = img.find_blobs()
        >>> img.draw_circle((blobs[-1].x,blobs[-1].y),10,color=Color.RED)
        >>> img.draw_circle((blobs[-1].centroid),10,color=Color.BLUE)
        >>> img.show()
        """
        return self.m10 / self.m00, self.m01 / self.m00

    @property
    def radius(self):
        """
        
        Return the radius, the avg distance of each contour point from the centroid
        """
        return float(np.mean(spsd.cdist(self._contour, [self.centroid])))

    @property
    def hull_radius(self):
        """
        Return the radius of the convex hull contour from the centroid
        """
        return float(np.mean(spsd.cdist(self._convex_hull, [self.centroid])))

    @lazy_property
    def image(self):
        # NOTE THAT THIS IS NOT PERFECT - ISLAND WITH A LAKE WITH AN ISLAND WITH A LAKE STUFF
        ret = np.zeros((self._image.width, self._image.height, 3), np.uint8)

        bmp = self._image.narray
        mask = self._mask.narray
        tl = self.top_left_corner()
        cv.SetImageROI(bmp, (tl[0], tl[1], self.width, self.height))
        cv.Copy(bmp, ret, mask)
        cv.ResetImageROI(bmp)
        return Image(ret)

    @lazy_property
    def mask(self):
        # TODO: FIX THIS SO THAT THE INTERIOR CONTOURS GET SHIFTED AND DRAWN

        # Alas - OpenCV does not provide an offset in the fillpoly method for
        # the cv bindings (only cv2 -- which I am trying to avoid). Have to
        # manually do the offset for the ROI shift.
        ret = np.zeros((self._image.width, self._image.height, 3), np.uint8)

        l, t = self.top_left_corner()

        # construct the exterior contour - these are tuples

        cv2.fillPoly(ret, [[(p[0] - l, p[1] - t) for p in self._contour]],
                     (255, 255, 255), 8)

        # construct the hole contoursb
        holes = []
        if self._hole_contour is not None:
            for h in self._hole_contour:  # -- these are lists
                holes.append([(h2[0] - l, h2[1] - t) for h2 in h])

            cv2.fillPoly(ret, holes, (0, 0, 0), 8)
        return Image(ret)

    @lazy_property
    def HullImage(self):
        ret = np.zeros((self._image.width, self._image.height, 3), np.uint8)
        bmp = self._image.narray
        mask = self._hull_mask.narray
        tl = self.top_left_corner()
        cv.SetImageROI(bmp, (tl[0], tl[1], self.width, self.height))
        cv.Copy(bmp, ret, mask)
        cv.ResetImageROI(bmp)
        return Image(ret)

    @lazy_property
    def HullMask(self):
        ret = np.zeros((self._image.width, self._image.height, 3), np.uint8)
        # Alas - OpenCV does not provide an offset in the fillpoly method for
        # the cv bindings (only cv2 -- which I am trying to avoid). Have to
        # manually do the offset for the ROI shift.
        thull = []
        l, t = self.top_left_corner()
        cv.FillPoly(ret, [[(p[0] - l, p[1] - t) for p in self._convex_hull]],
                    (255, 255, 255), 8)
        return Image(ret)

    @property
    def hull_image(self):
        """
        
        The convex hull of a blob is the shape that would result if you snapped a rubber band around
        the blob. So if you had the letter "C" as your blob the convex hull would be the letter "O."
        This method returns an image where the source image around the convex hull of the blob is copied
        ontop a black background.
        :return:
        Returns a PhloxAR Image of the convex hull, cropped to fit.
        >>> img = Image("lena")
        >>> blobs = img.find_blobs()
        >>> blobs[-1].hullImage().show()
        """
        return self._hull_img

    @property
    def hull_mask(self):
        """
        
        The convex hull of a blob is the shape that would result if you snapped a rubber band around
        the blob. So if you had the letter "C" as your blob the convex hull would be the letter "O."
        This method returns an image where the area of the convex hull is white and the rest of the image
        is black. This image is cropped to the size of the blob.
        :return:
        Returns a binary PhloxAR image of the convex hull mask, cropped to fit the blob.
        :Example:
        >>> img = Image("lena")
        >>> blobs = img.find_blobs()
        >>> blobs[-1].hullMask().show()
        """
        return self._hull_mask

    def blob_image(self):
        """
        
        This method automatically copies all of the image data around the blob and puts it in a new
        image. The resulting image has the size of the blob, with the blob data copied in place.
        Where the blob is not present the background is black.
        :return:
        Returns just the image of the blob (cropped to fit).
        :Example:
        >>> img = Image("lena")
        >>> blobs = img.find_blobs()
        >>> blobs[-1].blob_image().show()
        """
        return self._image

    def blob_mask(self):
        """
        
        This method returns an image of the blob's mask. Areas where the blob are present are white
        while all other areas are black. The image is cropped to match the blob area.
        :return:
        Returns a PhloxAR image of the blob's mask, cropped to fit.
        :Example:
        >>> img = Image("lena")
        >>> blobs = img.find_blobs()
        >>> blobs[-1].blob_mask().show()
        """
        return self._mask

    def match(self, otherblob):
        """
        
        Compare the Hu moments between two blobs to see if they match.  Returns
        a comparison factor -- lower numbers are a closer match.
        
        :param otherblob* - The other blob to compare this one to.
        :return:
        A single floating point value that is the match quality.
        :Example:
        >>> cam = Camera()
        >>> img1 = cam.getImage()
        >>> img2 = cam.getImage()
        >>> b1 = img1.find_blobs()
        >>> b2 = img2.find_blobs()
        >>> for ba in b1:
        >>>     for bb in b2:
        >>>         print ba.match(bb)
        """
        # note: this should use cv.MatchShapes -- but that seems to be
        # broken in OpenCV 2.2  Instead, I reimplemented in numpy
        # according to the description in the docs for method I1 (reciprocal log transformed abs diff)
        # return cv.MatchShapes(self.seq, otherblob.seq, cv.CV_CONTOURS_MATCH_I1)

        mySigns = np.sign(self._hu)
        myLogs = np.log(np.abs(self._hu))
        myM = mySigns * myLogs

        otherSigns = np.sign(otherblob.mHu)
        otherLogs = np.log(np.abs(otherblob.mHu))
        otherM = otherSigns * otherLogs

        return np.sum(abs((1 / myM - 1 / otherM)))

    def get_masked_image(self):
        """
        Get the blob size image with the masked blob
        """
        ret = np.zeros((self._image.width, self._image.height, 3), np.uint8)
        bmp = self._image.narray
        mask = self._mask.narray
        tl = self.top_left_corner()
        cv.SetImageROI(bmp, (tl[0], tl[1], self.width, self.height))
        cv.Copy(bmp, ret, mask)
        cv.ResetImageROI(bmp)
        return Image(ret)

    def get_full_masked_image(self):
        """
        Get the full size image with the masked to the blob
        """
        ret = np.zeros((self._image.width, self._image.height, 3), np.uint8)
        bmp = self._image.narray
        mask = self._mask.narray
        tl = self.top_left_corner()
        cv.SetImageROI(ret, (tl[0], tl[1], self.width, self.height))
        cv.SetImageROI(bmp, (tl[0], tl[1], self.width, self.height))
        cv.Copy(bmp, ret, mask)
        cv.ResetImageROI(bmp)
        cv.ResetImageROI(ret)
        return Image(ret)

    def get_full_hull_masked_image(self):
        """
        Get the full size image with the masked to the blob
        """
        ret = np.zeros((self._image.width, self._image.height, 3), np.uint8)
        bmp = self._image.narray
        mask = self._hull_mask.narray
        tl = self.top_left_corner()
        cv.SetImageROI(ret, (tl[0], tl[1], self.width, self.height))
        cv.SetImageROI(bmp, (tl[0], tl[1], self.width, self.height))
        cv.Copy(bmp, ret, mask)
        cv.ResetImageROI(bmp)
        cv.ResetImageROI(ret)
        return Image(ret)

    def get_full_mask(self):
        """
        Get the full sized image mask
        """
        ret = np.zeros((self._image.width, self._image.height, 3), np.uint8)
        mask = self._mask.narray
        tl = self.top_left_corner()
        cv.SetImageROI(ret, (tl[0], tl[1], self.width, self.height))
        cv.Copy(mask, ret)
        cv.ResetImageROI(ret)
        return Image(ret)

    def get_full_hull_mask(self):
        """
        Get the full sized image hull mask
        """
        ret = np.zeros((self._image.width, self._image.height, 3), np.uint8)
        mask = self._hull_mask.narray
        tl = self.top_left_corner()
        cv.SetImageROI(ret, (tl[0], tl[1], self.width, self.height))
        cv.Copy(mask, ret)
        cv.ResetImageROI(ret)
        return Image(ret)

    def get_hull_edge_image(self):
        ret = np.zeros((self._image.width, self._image.height, 3), np.uint8)
        tl = self.top_left_corner()
        translate = [(cs[0] - tl[0], cs[1] - tl[1]) for cs in self._convex_hull]
        cv2.polylines(ret, [translate], 1, (255, 255, 255))
        return Image(ret)

    def get_full_hull_edge_image(self):
        ret = np.zeros((self._image.width, self._image.height, 3), np.uint8)
        cv2.polylines(ret, [self._convex_hull], 1, (255, 255, 255))
        return Image(ret)

    def get_edge_image(self):
        """
        Get the edge image for the outer contour (no inner holes)
        """
        ret = np.zeros((self.width, self.height, 3), np.uint8)
        tl = self.top_left_corner()
        translate = [(cs[0] - tl[0], cs[1] - tl[1]) for cs in self._contour]
        cv2.polylines(ret, [translate], 1, (255, 255, 255))
        return Image(ret)

    def get_full_edge_image(self):
        """
        Get the edge image within the full size image.
        """
        ret = np.zeros((self._image.width, self._image.height, 3), np.uint8)
        cv2.polylines(ret, [self._contour], 1, (255, 255, 255))
        return Image(ret)

    def __repr__(self):
        return "PhloxAR.Features.Blob.Blob object at (%d, %d) with area %d" % (
        self.x, self.y, self.area)

    def _respace_points(self, contour, min_distance=1, max_distance=5):
        p0 = np.array(contour[-1])
        min_d = min_distance ** 2
        max_d = max_distance ** 2
        contour = [p0] + contour[:-1]
        contour = contour[:-1]
        ret = [p0]
        while len(contour) > 0:
            pt = np.array(contour.pop())
            dist = ((p0[0] - pt[0]) ** 2) + ((p0[1] - pt[1]) ** 2)
            if (dist > max_d):  # create the new point
                # get the unit vector from p0 to pt
                # from p0 to pt
                a = float((pt[0] - p0[0]))
                b = float((pt[1] - p0[1]))
                l = np.sqrt((a ** 2) + (b ** 2))
                punit = np.array([a / l, b / l])
                # make it max_distance long and add it to p0
                pn = (max_distance * punit) + p0
                # push the new point onto the return value
                ret.append((pn[0], pn[1]))
                contour.append(pt)  # push the new point onto the contour too
                p0 = pn
            elif dist > min_d:
                p0 = np.array(pt)
                ret.append(pt)
        return ret

    def _filter_scpoints(self, min_distance=3, max_distance=8):
        """
        Go through ever point in the contour and make sure
        that it is no less than min distance to the next point
        and no more than max_distance from the the next point.
        """
        completeContour = self._respace_points(self._contour, min_distance,
                                               max_distance)
        if self._hole_contour is not None:
            for ctr in self._hole_contour:
                completeContour = (completeContour +
                                   self._respace_points(ctr, min_distance,
                                                        max_distance))
        return completeContour

    @property
    def scdescriptor(self):
        if self._scdescriptors is not None:
            return self._scdescriptors, self._complete_contour
        completeContour = self._filter_scpoints()
        descriptor = self._generate_sc(completeContour)
        self._scdescriptors = descriptor
        self._complete_contour = completeContour
        return descriptor, completeContour

    def _generate_sc(self, completeContour, dsz=6, r_bound=[.1, 2.1]):
        """
        Create the shape context objects.
        dsz - The size of descriptor as a dszxdsz histogram
        completeContour - All of the edge points as a long list
        r_bound - Bounds on the log part of the shape context descriptor
        """
        data = []
        for pt in completeContour:  #
            temp = []
            # take each other point in the contour, center it on pt, and covert it to log polar
            for b in completeContour:
                r = np.sqrt((b[0] - pt[0]) ** 2 + (b[1] - pt[1]) ** 2)
                #                if( r > 100 ):
                #                    continue
                if (
                    r == 0.00):  # numpy throws an inf here that mucks the system up
                    continue
                r = np.log10(r)
                theta = np.arctan2(b[0] - pt[0], b[1] - pt[1])
                if np.isfinite(r) and np.isfinite(theta):
                    temp.append((r, theta))
            data.append(temp)

        # UHG!!! need to repeat this for all of the interior contours too
        descriptors = []
        # dsz = 6
        # for each point in the contour
        for d in data:
            test = np.array(d)
            # generate a 2D histrogram, and flatten it out.
            hist, a, b = np.histogram2d(test[:, 0], test[:, 1], dsz,
                                        [r_bound, [np.pi * -1 / 2, np.pi / 2]],
                                        normed=True)
            hist = hist.reshape(1, dsz ** 2)
            if np.all(np.isfinite(hist[0])):
                descriptors.append(hist[0])

        self._scdescriptors = descriptors
        return descriptors

    def get_shape_context(self):
        """
        Return the shape context descriptors as a featureset. Corrently
        this is not used for recognition but we will perhaps use it soon.
        """
        # still need to subsample big contours
        derp = self.scdescriptor
        descriptors, completeContour = self.scdescriptor
        fs = FeatureSet()
        for i in range(0, len(completeContour)):
            fs.append(ShapeContextDescriptor(self._image, completeContour[i],
                                             descriptors[i], self))

        return fs

    def show_correspondence(self, otherBlob, side="left"):
        """
        This is total beta - use at your own risk.
        """
        # We're lazy right now, assume the blob images are the same size
        side = side.lower()
        myPts = self.get_shape_context()
        yourPts = otherBlob.get_shape_context()

        myImg = self._image.copy()
        yourImg = otherBlob.image.copy()

        myPts = myPts.reassign_image(myImg)
        yourPts = yourPts.reassign_image(yourImg)

        myPts.draw()
        myImg = myImg.apply_layers()
        yourPts.draw()
        yourImg = yourImg.apply_layers()

        result = myImg.sideBySide(yourImg, side=side)
        data = self.shapeContextMatch(otherBlob)
        mapvals = data[0]
        color = Color()
        for i in range(0, len(self._complete_contour)):
            lhs = self._complete_contour[i]
            idx = mapvals[i]
            rhs = otherBlob._completeContour[idx]
            if side == "left":
                shift = (rhs[0] + yourImg.width, rhs[1])
                result.drawLine(lhs, shift, color=color.random(),
                                thickness=1)
            elif side == "bottom":
                shift = (rhs[0], rhs[1] + myImg.height)
                result.drawLine(lhs, shift, color=color.random(),
                                thickness=1)
            elif side == "right":
                shift = (rhs[0] + myImg.width, rhs[1])
                result.drawLine(lhs, shift, color=color.random(),
                                thickness=1)
            elif side == "top":
                shift = (lhs[0], lhs[1] + myImg.height)
                result.drawLine(lhs, shift, color=color.random(),
                                thickness=1)

        return result

    def get_match_metric(self, otherBlob):
        """
        This match metric is now deprecated.
        """
        data = self.shapeContextMatch(otherBlob)
        distances = np.array(data[1])
        sd = np.std(distances)
        x = np.mean(distances)
        min = np.min(distances)
        # not sure trimmed mean is perfect
        # realistically we should have some bimodal dist
        # and we want to throw away stuff with awful matches
        # so long as the number of points is not a huge
        # chunk of our points.
        tmean = sps.tmean(distances, (min, x + sd))
        return tmean

    def get_convexity_defects(self, returnPoints=False):
        """
        
        Get Convexity Defects of the contour.
        
        *returnPoints* - Bool(False).
                         If False: Returns FeatureSet of Line(start point, end point)
                         and Corner(far point)
                         If True: Returns a list of tuples
                         (start point, end point, far point)
        :return:
        FeatureSet - A FeatureSet of Line and Corner objects
                     OR
                     A list of (start point, end point, far point)
                     See PARAMETERS.
        :Example:
        >>> img = Image('lena')
        >>> blobs = img.find_blobs()
        >>> blob = blobs[-1]
        >>> lines, farpoints = blob.get_convexity_defects()
        >>> lines.draw()
        >>> farpoints.draw(color=Color.RED, width=-1)
        >>> img.show()
        >>> points = blob.get_convexity_defects(returnPoints=True)
        >>> startpoints = zip(*points)[0]
        >>> endpoints = zip(*points)[0]
        >>> farpoints = zip(*points)[0]
        >>> print startpoints, endpoints, farpoints
        """

        def cv_fallback():
            convex_hull = cv2.convexHull(self._contour)
            defects = cv2.convexityDefects(self._contour, convex_hull)
            points = [(defect[0], defect[1], defect[2]) for defect in defects]
            return points

        try:
            import cv2
            if hasattr(cv2, "convexityDefects"):
                hull = [self._contour.index(x) for x in self._convex_hull]
                hull = np.array(hull).reshape(len(hull), 1)
                defects = cv2.convexityDefects(np.array(self._contour), hull)
                if isinstance(defects, type(None)):
                    warnings.warn(
                        "Unable to find defects. Returning Empty FeatureSet.")
                    defects = []
                points = [(self._contour[defect[0][0]],
                           self._contour[defect[0][1]],
                           self._contour[defect[0][2]]) for defect in defects]
            else:
                points = cv_fallback()
        except ImportError:
            points = cv_fallback()

        if returnPoints:
            return FeatureSet(points)
        else:
            lines = FeatureSet(
                    [Line(self._image, (start, end)) for start, end, far in
                     points])
            farpoints = FeatureSet(
                    [Corner(self._image, far[0], far[1]) for start, end, far in
                     points])
            features = FeatureSet([lines, farpoints])
            return features


class BlobMaker(object):
    """
    Blob maker encapsulates all of the contour extraction process and data, so
    it can be used inside the image class, or extended and used outside the
    image class. The general idea is that the blob maker provides the utilities
    that one would use for blob extraction. Later implementations may include
    tracking and other features.
    """
    _mem_storage = None

    def __init__(self):
        self._mem_storage = cv.CreateMemStorage()

    def extract_with_model(self, img, colormodel, minsize=10, maxsize=0):
        """
        Extract blobs using a color model
        :param img: the input image
        :param colormodel: the color model to use.
        :param minsize: the minimum size of the returned features.
        :param maxsize: the maximum size of the returned features 0=uses the default value.

        Parameters:
            img - Image
            colormodel - ColorModel object
            minsize - Int
            maxsize - Int
        """
        if maxsize <= 0:
            maxsize = img.width * img.height
        gray = colormodel.threshold(img)
        blobs = self.extract_from_binary(gray, img, minArea=minsize,
                                         maxArea=maxsize)
        ret = sorted(blobs, key=lambda x: x.mArea, reverse=True)
        return FeatureSet(ret)

    def extract(self, img, threshval=127, minsize=10, maxsize=0,
                threshblocksize=3, threshconstant=5):
        """
        This method performs a threshold operation on the input image and then
        extracts and returns the blobs.
        img       - The input image (color or b&w)
        threshval - The threshold value for the binarize operation. If threshval = -1 adaptive thresholding is used
        minsize   - The minimum blob size in pixels.
        maxsize   - The maximum blob size in pixels. 0=uses the default value.
        threshblocksize - The adaptive threhold block size.
        threshconstant  - The minimum to subtract off the adaptive threshold
        """
        if maxsize <= 0:
            maxsize = img.width * img.height

        # create a single channel image, thresholded to parameters

        blobs = self.extract_from_binary(
            img.binarize(threshval, 255, threshblocksize,
                         threshconstant).invert(), img, minsize, maxsize)
        ret = sorted(blobs, key=lambda x: x.mArea, reverse=True)
        return FeatureSet(ret)

    def extract_from_binary(self, binaryImg, colorImg, minsize=5, maxsize=-1,
                            appx_level=3):
        """
        This method performs blob extraction given a binary source image that is used
        to get the blob images, and a color source image.
        binarymg- The binary image with the blobs.
        colorImg - The color image.
        minSize  - The minimum size of the blobs in pixels.
        maxSize  - The maximum blob size in pixels.
        * *appx_level* - The blob approximation level - an integer for the maximum distance between the true edge and the approximation edge - lower numbers yield better approximation.
        """
        # If you hit this recursion limit may god have mercy on your soul.
        # If you really are having problems set the value higher, but this means
        # you have over 10,000,000 blobs in your image.
        sys.setrecursionlimit(5000)
        # h_next moves to the next external contour
        # v_next() moves to the next internal contour
        if maxsize <= 0:
            maxsize = colorImg.width * colorImg.height

        ret = []
        test = binaryImg.mean_color
        if test[0] == 0.00 and test[1] == 0.00 and test[2] == 0.00:
            return FeatureSet(ret)

        # There are a couple of weird corner cases with the opencv
        # connect components libraries - when you try to find contours
        # in an all black image, or an image with a single white pixel
        # that sits on the edge of an image the whole thing explodes
        # this check catches those bugs. -KAS
        # Also I am submitting a bug report to Willow Garage - please bare with us.
        ptest = (4 * 255.0) / (
        binaryImg.width * binaryImg.height)  # val if two pixels are white
        if test[0] <= ptest and test[1] <= ptest and test[2] <= ptest:
            return ret

        seq = cv.FindContours(binaryImg._get_gray_narray(),
                              self._mem_storage, cv.CV_RETR_TREE,
                              cv.CV_CHAIN_APPROX_SIMPLE)
        if not list(seq):
            warnings.warn("Unable to find Blobs. Retuning Empty FeatureSet.")
            return FeatureSet([])
        try:
            # note to self
            # http://code.activestate.com/recipes/474088-tail-call-optimization-decorator/
            ret = self._extract_from_binary(seq, False, colorImg, minsize,
                                               maxsize, appx_level)
        except RuntimeError as e:
            logger.warning("You exceeded the recursion limit. This means you "
                           "probably have too many blobs in your image. We "
                           "suggest you do some morphological operations "
                           "(erode/dilate) to reduce the number of blobs in "
                           "your image. This function was designed to max out "
                           "at about 5000 blobs per image.")
        except e:
            logger.warning("PhloxAR Find Blobs Failed - This could be an OpenCV "
                           "python binding issue")
        del seq
        return FeatureSet(ret)

    def _extract_from_binary(self, seq, isaHole, colorImg, minsize, maxsize,
                             appx_level):
        """
        The recursive entry point for the blob extraction. The blobs and holes
        are presented as a tree and we traverse up and across the tree.
        """
        ret = []

        if seq is None:
            return ret

        nextLayerDown = []
        while True:
            # if we aren't a hole then we are an object, so get and
            # return our featuress
            if not isaHole:
                temp = self._extract_data(seq, colorImg, minsize, maxsize,
                                          appx_level)
                if temp is not None:
                    ret.append(temp)

            nextLayer = seq.v_next()

            if nextLayer is not None:
                nextLayerDown.append(nextLayer)

            seq = seq.h_next()

            if seq is None:
                break

        for nextLayer in nextLayerDown:
            ret += self._extract_from_binary(nextLayer, not isaHole, colorImg,
                                             minsize, maxsize, appx_level)

        return ret

    def _extract_data(self, seq, color, minsize, maxsize, appx_level):
        """
        Extract the bulk of the data from a give blob. If the blob's are is too large
        or too small the method returns none.
        """
        if seq is None or not len(seq):
            return None
        area = cv.ContourArea(seq)
        if area < minsize or area > maxsize:
            return None

        ret = Blob()
        ret.image = color
        ret.mArea = area

        ret.mMinRectangle = cv.MinAreaRect2(seq)
        bb = cv.BoundingRect(seq)
        ret.x = bb[0] + (bb[2] / 2)
        ret.y = bb[1] + (bb[3] / 2)
        ret.mPerimeter = cv.ArcLength(seq)
        if seq is not None:  # KAS
            ret.contour = list(seq)
            if ret.contour is not None:
                ret.contourAppx = []
                appx = cv2.approxPolyDP(npy.array([ret.contour], 'float32'),
                                        appx_level, True)
                for p in appx:
                    ret.contourAppx.append((int(p[0][0]), int(p[0][1])))

        # so this is a bit hacky....

        # For blobs that live right on the edge of the image OpenCV reports the position and width
        #   height as being one over for the true position. E.g. if a blob is at (0,0) OpenCV reports
        #   its position as (1,1). Likewise the width and height for the other corners is reported as
        #   being one less than the width and height. This is a known bug.

        xx = bb[0]
        yy = bb[1]
        ww = bb[2]
        hh = bb[3]
        ret.points = [(xx, yy), (xx + ww, yy), (xx + ww, yy + hh),
                         (xx, yy + hh)]
        ret._update_extents()
        chull = cv.ConvexHull2(seq, cv.CreateMemStorage(), return_points=1)
        ret.mConvexHull = list(chull)

        del chull

        moments = cv.Moments(seq)

        # This is a hack for a python wrapper bug that was missing
        # the constants required from the ctype
        ret.m00 = area
        try:
            ret.m10 = moments.m10
            ret.m01 = moments.m01
            ret.m11 = moments.m11
            ret.m20 = moments.m20
            ret.m02 = moments.m02
            ret.m21 = moments.m21
            ret.m12 = moments.m12
        except:
            ret.m10 = cv.GetSpatialMoment(moments, 1, 0)
            ret.m01 = cv.GetSpatialMoment(moments, 0, 1)
            ret.m11 = cv.GetSpatialMoment(moments, 1, 1)
            ret.m20 = cv.GetSpatialMoment(moments, 2, 0)
            ret.m02 = cv.GetSpatialMoment(moments, 0, 2)
            ret.m21 = cv.GetSpatialMoment(moments, 2, 1)
            ret.m12 = cv.GetSpatialMoment(moments, 1, 2)

        ret.hu = cv.GetHuMoments(moments)

        # KAS -- FLAG FOR REPLACE 6/6/2012
        mask = self._get_mask(seq, bb)

        ret.avg_color = self._get_avg(color.bitmap, bb, mask)
        ret.avg_color = ret.avg_color[0:3]

        ret.mHoleContour = self._get_holes(seq)
        ret.mAspectRatio = ret.mMinRectangle[1][0] / \
                              ret.mMinRectangle[1][1]

        return ret

    def _get_holes(self, seq):
        """
        This method returns the holes associated with a blob as a list of tuples.
        """
        ret = None
        holes = seq.v_next()
        if holes is not None:
            ret = [list(holes)]
            while holes.h_next() is not None:
                holes = holes.h_next();
                temp = list(holes)
                if len(temp) >= 3:  # exclude single pixel holes
                    ret.append(temp)
        return ret

    def _get_mask(self, seq, bb):
        """
        Return a binary image of a particular contour sequence.
        """
        # bb = cv.BoundingRect(seq)
        mask = cv.CreateImage((bb[2], bb[3]), cv.IPL_DEPTH_8U, 1)
        cv.Zero(mask)
        cv.DrawContours(mask, seq, 255, 0, 0, thickness=-1,
                        offset=(-1 * bb[0], -1 * bb[1]))
        holes = seq.v_next()
        if holes is not None:
            cv.DrawContours(mask, holes, 0, 255, 0, thickness=-1,
                            offset=(-1 * bb[0], -1 * bb[1]))
            while holes.h_next() is not None:
                holes = holes.h_next()
                if holes is not None:
                    cv.DrawContours(mask, holes, 0, 255, 0, thickness=-1,
                                    offset=(-1 * bb[0], -1 * bb[1]))
        return mask

    def _get_hull_mask(self, hull, bb):
        """
        Return a mask of the convex hull of a blob.
        """
        bb = cv.BoundingRect(hull)
        mask = cv.CreateImage((bb[2], bb[3]), cv.IPL_DEPTH_8U, 1)
        cv.Zero(mask)
        cv.DrawContours(mask, hull, 255, 0, 0, thickness=-1,
                        offset=(-1 * bb[0], -1 * bb[1]))
        return mask

    def _get_avg(self, colorbitmap, bb, mask):
        """
        Calculate the average color of a blob given the mask.
        """
        cv.SetImageROI(colorbitmap, bb)
        # may need the offset parameter
        avg = cv.Avg(colorbitmap, mask)
        cv.ResetImageROI(colorbitmap)
        return avg

    def _get_blob_as_image(self, seq, bb, colorbitmap, mask):
        """
        Return an image that contains just pixels defined by the blob sequence.
        """
        cv.SetImageROI(colorbitmap, bb)
        outputImg = cv.CreateImage((bb[2], bb[3]), cv.IPL_DEPTH_8U, 3)
        cv.Zero(outputImg)
        cv.Copy(colorbitmap, outputImg, mask)
        cv.ResetImageROI(colorbitmap)
        return Image(outputImg)
