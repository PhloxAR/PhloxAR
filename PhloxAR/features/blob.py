# -*- coding:utf-8 -*-
from __future__ import absolute_import, unicode_literals
from __future__ import division, print_function

import scipy.stats as sps

from PhloxAR.base import *
from PhloxAR.core.color import Color
from PhloxAR.core.image import Image
from PhloxAR.features.detection import Corner, Line, ShapeContextDescriptor
from PhloxAR.features.feature import Feature, FeatureSet

__all__ = [
    'Blob'
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
            self.__dict__[key] = cv.CreateImageHeader((self.width, self.height),
                                                      cv.IPL_DEPTH_8U, 1)
            cv.SetData(self.__dict__[key], state[k])

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
        cv.SetImageROI(self._image.bitmap, hack)
        # may need the offset parameter
        avg = cv.Avg(self._image.bitmap, self._mask._get_gray_narray())
        cv.ResetImageROI(self._image.bitmap)

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
        ang = npy.pi*(float(ang) / 180)
        tx = self.min_rect_x()
        ty = self.min_rect_y()
        w = self.min_rect_width() / 2.0
        h = self.min_rect_height() / 2.0

        # [ cos a , -sin a, tx ]
        # [ sin a , cos a , ty ]
        # [ 0     , 0     ,  1 ]
        derp = npy.matrix([
            [cos(ang), -sin(ang), tx],
            [sin(ang), cos(ang), ty],
            [0, 0, 1]
        ])

        tl = npy.matrix([-w, h, 1.0])  # kat gladly supports homo. coord
        tr = npy.matrix([w, h, 1.0])
        bl = npy.matrix([-w, -h, 1.0])
        br = npy.matrix([w, -h, 1.0])
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
        theta = npy.pi * (angle / 180.0)
        mode = ""
        point = (self.x, self.y)
        self._image = self._image.rotate(angle, mode, point)
        self._hull_img = self._hull_img.rotate(angle, mode, point)
        self._mask = self._mask.rotate(angle, mode, point)
        self._hull_mask = self._hull_mask.rotate(angle, mode, point)

        self._contour = map(
                lambda x: (
                    x[0] * npy.cos(theta) - x[1] * npy.sin(theta),
                    x[0] * npy.sin(theta) + x[1] * npy.cos(theta)
                ),
                self._contour
        )
        self._convex_hull = map(
                lambda x: (
                    x[0] * npy.cos(theta) - x[1] * npy.sin(theta),
                    x[0] * npy.sin(theta) + x[1] * npy.cos(theta)
                ),
                self._convex_hull
        )

        if self._hole_contour is not None:
            for h in self._hole_contour:
                h = map(
                        lambda x: (
                            x[0] * npy.cos(theta) - x[1] * npy.sin(theta),
                            x[0] * npy.sin(theta) + x[1] * npy.cos(theta)),
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
            maskred = cv.CreateImage(
                cv.GetSize(self._mask._get_gray_narray()), cv.IPL_DEPTH_8U,
                1
            )
            
            maskgrn = cv.CreateImage(
                cv.GetSize(self._mask._get_gray_narray()), cv.IPL_DEPTH_8U,
                1
            )
            
            maskblu = cv.CreateImage(
                cv.GetSize(self._mask._get_gray_narray()), cv.IPL_DEPTH_8U,
                1
            )

            maskbit = cv.CreateImage(
                cv.GetSize(self._mask._get_gray_narray()), cv.IPL_DEPTH_8U,
                3
            )

            cv.ConvertScale(self._mask._get_gray_narray(), maskred,
                            color[0] / 255.0)
            cv.ConvertScale(self._mask._get_gray_narray(), maskgrn,
                            color[1] / 255.0)
            cv.ConvertScale(self._mask._get_gray_narray(), maskblu,
                            color[2] / 255.0)

            cv.Merge(maskblu, maskgrn, maskred, None, maskbit)

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
            lastp = self._convex_hull[0]  # this may work better.... than the other
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
        if (layer is not None):
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
        return float(numwhite) / (radius * radius * npy.pi)

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
        return float(npy.mean(spsd.cdist(self._contour, [self.centroid])))

    @property
    def hull_radius(self):
        """
        Return the radius of the convex hull contour from the centroid
        """
        return float(npy.mean(spsd.cdist(self._convex_hull, [self.centroid])))

    @lazy_property
    def image(self):
        # NOTE THAT THIS IS NOT PERFECT - ISLAND WITH A LAKE WITH AN ISLAND WITH A LAKE STUFF
        ret = cv.CreateImage((self.width, self.height), cv.IPL_DEPTH_8U,
                                3)
        cv.Zero(ret)
        bmp = self._image.bitmap
        mask = self._mask.bitmap
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

        ret = cv.CreateImage((self.width, self.height), cv.IPL_DEPTH_8U,
                                1)
        cv.Zero(ret)
        l, t = self.top_left_corner()

        # construct the exterior contour - these are tuples

        cv.FillPoly(ret, [[(p[0] - l, p[1] - t) for p in self._contour]],
                    (255, 255, 255), 8)

        # construct the hole contoursb
        holes = []
        if self._hole_contour is not None:
            for h in self._hole_contour:  # -- these are lists
                holes.append([(h2[0] - l, h2[1] - t) for h2 in h])

            cv.FillPoly(ret, holes, (0, 0, 0), 8)
        return Image(ret)

    @lazy_property
    def HullImage(self):
        ret = cv.CreateImage((self.width, self.height), cv.IPL_DEPTH_8U,
                                3)
        cv.Zero(ret)
        bmp = self._image.bitmap
        mask = self._hull_mask.bitmap
        tl = self.top_left_corner()
        cv.SetImageROI(bmp, (tl[0], tl[1], self.width, self.height))
        cv.Copy(bmp, ret, mask)
        cv.ResetImageROI(bmp)
        return Image(ret)

    @lazy_property
    def HullMask(self):
        ret = cv.CreateImage((self.width, self.height), cv.IPL_DEPTH_8U,
                                3)
        cv.Zero(ret)
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

        mySigns = npy.sign(self._hu)
        myLogs = npy.log(npy.abs(self._hu))
        myM = mySigns * myLogs

        otherSigns = npy.sign(otherblob.mHu)
        otherLogs = npy.log(npy.abs(otherblob.mHu))
        otherM = otherSigns * otherLogs

        return npy.sum(abs((1 / myM - 1 / otherM)))

    def get_masked_image(self):
        """
        Get the blob size image with the masked blob
        """
        ret = cv.CreateImage((self.width, self.height), cv.IPL_DEPTH_8U,
                                3)
        cv.Zero(ret)
        bmp = self._image.bitmap
        mask = self._mask.bitmap
        tl = self.top_left_corner()
        cv.SetImageROI(bmp, (tl[0], tl[1], self.width, self.height))
        cv.Copy(bmp, ret, mask)
        cv.ResetImageROI(bmp)
        return Image(ret)

    def get_full_masked_image(self):
        """
        Get the full size image with the masked to the blob
        """
        ret = cv.CreateImage((self._image.width, self._image.height),
                                cv.IPL_DEPTH_8U, 3)
        cv.Zero(ret)
        bmp = self._image.bitmap
        mask = self._mask.bitmap
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
        ret = cv.CreateImage((self._image.width, self._image.height),
                                cv.IPL_DEPTH_8U, 3)
        cv.Zero(ret)
        bmp = self._image.bitmap
        mask = self._hull_mask.bitmap
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
        ret = cv.CreateImage((self._image.width, self._image.height),
                                cv.IPL_DEPTH_8U, 3)
        cv.Zero(ret)
        mask = self._mask.bitmap
        tl = self.top_left_corner()
        cv.SetImageROI(ret, (tl[0], tl[1], self.width, self.height))
        cv.Copy(mask, ret)
        cv.ResetImageROI(ret)
        return Image(ret)

    def get_full_hull_mask(self):
        """
        Get the full sized image hull mask
        """
        ret = cv.CreateImage((self._image.width, self._image.height),
                                cv.IPL_DEPTH_8U, 3)
        cv.Zero(ret)
        mask = self._hull_mask.bitmap
        tl = self.top_left_corner()
        cv.SetImageROI(ret, (tl[0], tl[1], self.width, self.height))
        cv.Copy(mask, ret)
        cv.ResetImageROI(ret)
        return Image(ret)

    def get_hull_edge_image(self):
        ret = cv.CreateImage((self.width, self.height), cv.IPL_DEPTH_8U,
                                3)
        cv.Zero(ret)
        tl = self.top_left_corner()
        translate = [(cs[0] - tl[0], cs[1] - tl[1]) for cs in self._convex_hull]
        cv.PolyLine(ret, [translate], 1, (255, 255, 255))
        return Image(ret)

    def get_full_hull_edge_image(self):
        ret = cv.CreateImage((self._image.width, self._image.height),
                                cv.IPL_DEPTH_8U, 3)
        cv.Zero(ret)
        cv.PolyLine(ret, [self._convex_hull], 1, (255, 255, 255))
        return Image(ret)

    def get_edge_image(self):
        """
        Get the edge image for the outer contour (no inner holes)
        """
        ret = cv.CreateImage((self.width, self.height), cv.IPL_DEPTH_8U,
                                3)
        cv.Zero(ret)
        tl = self.top_left_corner()
        translate = [(cs[0] - tl[0], cs[1] - tl[1]) for cs in self._contour]
        cv.PolyLine(ret, [translate], 1, (255, 255, 255))
        return Image(ret)

    def get_full_edge_image(self):
        """
        Get the edge image within the full size image.
        """
        ret = cv.CreateImage((self._image.width, self._image.height),
                                cv.IPL_DEPTH_8U, 3)
        cv.Zero(ret)
        cv.PolyLine(ret, [self._contour], 1, (255, 255, 255))
        return Image(ret)

    def __repr__(self):
        return "PhloxAR.Features.Blob.Blob object at (%d, %d) with area %d" % (
        self.x, self.y, self.area)

    def _respace_points(self, contour, min_distance=1, max_distance=5):
        p0 = npy.array(contour[-1])
        min_d = min_distance ** 2
        max_d = max_distance ** 2
        contour = [p0] + contour[:-1]
        contour = contour[:-1]
        ret = [p0]
        while len(contour) > 0:
            pt = npy.array(contour.pop())
            dist = ((p0[0] - pt[0]) ** 2) + ((p0[1] - pt[1]) ** 2)
            if (dist > max_d):  # create the new point
                # get the unit vector from p0 to pt
                # from p0 to pt
                a = float((pt[0] - p0[0]))
                b = float((pt[1] - p0[1]))
                l = npy.sqrt((a ** 2) + (b ** 2))
                punit = npy.array([a / l, b / l])
                # make it max_distance long and add it to p0
                pn = (max_distance * punit) + p0
                # push the new point onto the return value
                ret.append((pn[0], pn[1]))
                contour.append(pt)  # push the new point onto the contour too
                p0 = pn
            elif dist > min_d:
                p0 = npy.array(pt)
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
                r = npy.sqrt((b[0] - pt[0]) ** 2 + (b[1] - pt[1]) ** 2)
                #                if( r > 100 ):
                #                    continue
                if (
                    r == 0.00):  # numpy throws an inf here that mucks the system up
                    continue
                r = npy.log10(r)
                theta = npy.arctan2(b[0] - pt[0], b[1] - pt[1])
                if npy.isfinite(r) and npy.isfinite(theta):
                    temp.append((r, theta))
            data.append(temp)

        # UHG!!! need to repeat this for all of the interior contours too
        descriptors = []
        # dsz = 6
        # for each point in the contour
        for d in data:
            test = npy.array(d)
            # generate a 2D histrogram, and flatten it out.
            hist, a, b = npy.histogram2d(test[:, 0], test[:, 1], dsz,
                                        [r_bound, [npy.pi * -1 / 2, npy.pi / 2]],
                                        normed=True)
            hist = hist.reshape(1, dsz ** 2)
            if npy.all(npy.isfinite(hist[0])):
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
        distances = npy.array(data[1])
        sd = npy.std(distances)
        x = npy.mean(distances)
        min = npy.min(distances)
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

        def cvFallback():
            chull = cv.ConvexHull2(self._contour, cv.CreateMemStorage(),
                                   return_points=False)
            defects = cv.ConvexityDefects(self._contour, chull,
                                          cv.CreateMemStorage())
            points = [(defect[0], defect[1], defect[2]) for defect in defects]
            return points

        try:
            import cv2
            if hasattr(cv2, "convexityDefects"):
                hull = [self._contour.index(x) for x in self._convex_hull]
                hull = npy.array(hull).reshape(len(hull), 1)
                defects = cv2.convexityDefects(npy.array(self._contour), hull)
                if isinstance(defects, type(None)):
                    warnings.warn(
                        "Unable to find defects. Returning Empty FeatureSet.")
                    defects = []
                points = [(self._contour[defect[0][0]],
                           self._contour[defect[0][1]],
                           self._contour[defect[0][2]]) for defect in defects]
            else:
                points = cvFallback()
        except ImportError:
            points = cvFallback()

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
