#!/usr/bin/env python
# -*- coding: utf-8 -*_

import sys
import os
import svgwrite
from PhloxAR.color import *
from PhloxAR.base import pg
from PhloxAR.base import npy
from PhloxAR.base import warnings


class DrawingLayer(object):
    """
    A way to mark up Image classes without changing the image data itself.
    This class wraps pygame's Surface class and provides basic drawing
    and text rendering functions.
    """
    _surface = []
    _default_color = 0
    _font_color = 0
    _clear_color = 0
    _font = 0
    _font_name = ''
    _font_size = 0
    _default_alpha = 255
    # used to track the changed value in alpha
    _alpha_delta = 1
    _svg = ''
    width = 0
    height = 0

    def __init__(self, (width, height)):
        if not pg.font.get_init():
            pg.font.init()

        self.width = width
        self.height = height
        self._surface = pg.Surface((width, height), flags=pg.SRCALPHA)
        self._default_alpha = 255
        self._clear_color = pg.Color(0, 0, 0, 0)

        self._surface.fill(self._clear_color)
        self._default_color = Color.BLACK

        self._font_size = 18
        self._font_name = None
        self._font_bold = False
        self._font_italic = False
        self._font_underline = False
        self._font = pg.font.Font(self._font_name, self._font_size)

    def __repr__(self):
        return '<PhloxAR.DrawingLayer object size ({}, {})>'.format(
                self.width, self.height)

    @property
    def default_alpha(self):
        """
        Returns the default alpha value.
        """
        return self._default_alpha

    @default_alpha.setter
    def default_alpha(self, alpha):
        """
        Sets the default alpha value for all methods called on this
        layer.
        """
        if 0 <= alpha <= 255:
            self._default_alpha = alpha

    @property
    def default_color(self):
        return self._default_color

    @default_color.setter
    def default_color(self, color):
        self._default_color = color

    @property
    def svg(self):
        return self._svg.tostring()

    @property
    def font_bold(self):
        return self._font_bold

    @font_bold.setter
    def font_bold(self, bold):
        self._font_bold = bold
        self._font.set_bold(bold)

    @property
    def font_italic(self):
        return self._font_italic

    @font_italic.setter
    def font_italic(self, italic):
        self._font_italic = italic
        self._font.set_italic(italic)

    @property
    def font_underline(self):
        return self._font_underline

    @font_underline.setter
    def font_underline(self, underline):
        self._font_underline = underline
        self._font.set_underline(underline)

    @property
    def font_size(self):
        return self._font_size

    @font_size.setter
    def font_size(self, size):
        self._font_size = size
        self._font = pg.font.Font(self._font_name, self._font_size)

    @property
    def font(self):
        return self._font

    @font.setter
    def font(self, fontface):
        full = pg.font.match_font(fontface)
        self._font_name = full
        self._font = pg.font.Font(self._font_name, self._font_size)

    def list_fonts(self):
        """
        Return a list of strings corresponding to the fonts available
        on the current system.
        """
        return pg.font.get_fonts()

    def set_layer_alpha(self, alpha):
        """
        Sets the alpha value of the entire layer in a single
        pass. Helpful for merging layers with transparency.
        """
        self._surface.set_alpha(alpha)
        # get access to the alpha band of the image
        pixels_alpha = pg.surfarray.pixels_alpha(self._surface)
        # do a floating point multiply, by alpha 100, on each alpha value
        # the truncate the values (convert to integer) and copy back
        # into the surface
        pixels_alpha[...] = (npy.ones(pixels_alpha.shape) * alpha).astype(npy.uint8)

        # unlock the surface
        self._alpha_delta = alpha / 255.0

        del pixels_alpha
        return None

    def _csv_rgb2_pygame_color(self, color, alpha=-1):
        if alpha == -1:
            alpha = self._default_alpha

        if color == Color.DEFAULT:
            color = self._default_color

        ret_val = pg.Color(color[0], color[1], color[2], alpha)

        return ret_val

    def line(self, start, stop, color=Color.DEFAULT, width=1,
             aalias=True, alpha=-1):
        """
        Draw a single line from the (x, y) tuple start to the (x, y) tuple stop.
        :param start: tuple
        :param stop: tuple

        Optional parameters:
        :param color: the object's color as a simple CVColor object, if no
                       is specified the default is used
        :param width:
        :param aalias: draw an anti aliased object of with one
        :param alpha: the alpha blending for the object. If this value is -1
                       then the layer default value is used. A value of 255
                       means opaque, while 0 means transparent
        :return: None
        """
        if aalias and width == 1:
            pg.draw.aaline(self._surface, self._csv_rgb2_pygame_color(color, alpha),
                           start, stop, width)
        else:
            pg.draw.line(self._surface, self._csv_rgb2_pygame_color(color, alpha),
                         start, stop, width)

        start_int = tuple(int(x) for x in start)
        stop_int = tuple(int(x) for x in stop)
        self._svg.add(self._svg.line(start=start_int, end=stop_int))

        return None

    def lines(self, points, color=Color.DEFAULT, aalias=True, alpha=-1, width=1):
        """
        Draw a set of lines from the list of (x, y) tuples points. Lines are
        draw between each successive pair of points.
        :param points: tuple

        Optional parameters:
        :param color: the object's color as a simple CVColor object, if no
                       is specified the default is used
        :param width:
        :param aalias: draw an anti aliased object of with one
        :param alpha: the alpha blending for the object. If this value is -1
                       then the layer default value is used. A value of 255
                       means opaque, while 0 means transparent
        :return: None
        """
        if aalias and width == 1:
            pg.draw.aalines(self._surface, self._csv_rgb2_pygame_color(color, alpha),
                            0, points, width)
        else:
            pg.draw_lines(self._surface, self._csv_rgb2_pygame_color(color, alpha),
                          0, points, width)

        last_point = points[0]

        for point in points[1:]:
            lint = tuple(int(x) for x in last_point)
            pint = tuple(int(x) for x in point)
            self._svg.add(self._svg.line(start=last_point, end=point))
            last_point = point

        return None

    def rectangle(self, pt1=None, pt2=None, color=Color.DEFAULT, width=1,
                  filled=False, alpha=-1, **kwargs):
        """
        Draw a rectangle. By default, using two points to define a rectangle,
        also could use center with its size to define a rectangle, you need to
        specify additional parameters 'center' and 'size' in kwargs.
        :param pt1:
        :param pt2:
        :param color: the object's color as a simple CVColor object, if no
                       is specified the default is used
        :param width: line width
        :param filled: the rectangle is to be filled or not
        :param alpha: the alpha blending for the object. If this value is -1
                       then the layer default value is used. A value of 255 means
                       opaque, while 0 means transparent.
        :param kwargs: if you want to use center and size to specify the
                        rectangle, add it in kwargs.
        :return: None
        """
        if filled:
            width = 0

        if pt1 is None and pt2 is None:
            if kwargs['center'] is None or kwargs['size'] is None:
                warnings.warn("Insufficient parameters.")
                return None
            else:
                x = kwargs['center'][0] - kwargs['size'][0] / 2
                y = kwargs['center'][1] - kwargs['size'][1] / 2
                r = pg.Rect(x, y, kwargs['size'][0], kwargs['size'][1])
                pg.draw.rect(self._surface, self._csv_rgb2_pygame_color(color, alpha),
                             r, width)
                s = tuple(int(x) for x in kwargs['size'])
                self._svg.add(self._svg.rect(insert=(int(x), int(y)), size=s))

                return None
        else:
            w = 0
            h = 0
            x = 0
            y = 0

            if pt1[0] > pt2[0]:
                w = pt1[0] - pt2[0]
                x = pt2[0]
            else:
                w = pt2[0] - pt1[0]
                x = pt1[0]

            if pt1[1] > pt2[1]:
                w = pt1[1] - pt2[1]
                x = pt2[1]
            else:
                w = pt2[1] - pt1[1]
                x = pt1[1]

            r = pg.Rect((x, y), (w, h))
            pg.draw.rect(self._surface, self._csv_rgb2_pygame_color(color, alpha),
                         r, width)
            self._svg.add(self._svg.rect(insert=(int(x), int(y)), size=(int(w), int(h))))

            return None

    def polygon(self):
        pass

    def circle(self):
        pass

    def ellipse(self):
        pass

    def bezier(self):
        pass

    def text(self):
        pass
