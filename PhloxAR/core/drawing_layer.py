# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

import warnings
import numpy as np
import pygame as sdl
import pygame.gfxdraw as gfxdraw
from PhloxAR.core.color import Color

# TODO: DOCUMENT

__all__ = [
    'DrawingLayer'
]


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
        if not sdl.font.get_init():
            sdl.font.init()

        self.width = width
        self.height = height
        self._surface = sdl.Surface((width, height), flags=sdl.SRCALPHA)
        self._default_alpha = 255
        self._clear_color = sdl.Color((0, 0, 0, 0))

        self._surface.fill(self._clear_color)
        self._default_color = Color.BLACK

        self._font_size = 18
        self._font_name = None
        self._font_bold = False
        self._font_italic = False
        self._font_underline = False
        self._font = sdl.font.Font(self._font_name, self._font_size)

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
        self._font = sdl.font.Font(self._font_name, self._font_size)

    @property
    def font(self):
        return self._font

    @font.setter
    def font(self, fontface):
        full = sdl.font.match_font(fontface)
        self._font_name = full
        self._font = sdl.font.Font(self._font_name, self._font_size)

    def list_fonts(self):
        """
        Return a list of strings corresponding to the fonts available
        on the current system.
        """
        return sdl.font.get_fonts()

    def set_layer_alpha(self, alpha):
        """
        Sets the alpha value of the entire layer in a single
        pass. Helpful for merging layers with transparency.
        """
        self._surface.set_alpha(alpha)
        # get access to the alpha band of the image
        pixels_alpha = sdl.surfarray.pixels_alpha(self._surface)
        # do a floating point multiply, by alpha 100, on each alpha value
        # the truncate the values (convert to integer) and copy back
        # into the surface
        pixels_alpha[...] = (np.ones(pixels_alpha.shape) * alpha).astype(np.uint8)

        # unlock the surface
        self._alpha_delta = alpha / 255.0

        del pixels_alpha
        return None

    def _csv_rgb2sdl_color(self, color, alpha=-1):
        if alpha == -1:
            alpha = self._default_alpha

        if color == Color.DEFAULT:
            color = self._default_color

        ret_val = sdl.Color((color[0], color[1], color[2], alpha))

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
            sdl.draw.aaline(self._surface,
                            self._csv_rgb2sdl_color(color, alpha),
                            start, stop, width)
        else:
            sdl.draw.line(self._surface, self._csv_rgb2sdl_color(color, alpha),
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
            sdl.draw.aalines(self._surface, self._csv_rgb2sdl_color(color, alpha),
                            0, points, width)
        else:
            sdl.draw_lines(self._surface, self._csv_rgb2sdl_color(color, alpha),
                          0, points, width)

        last_point = points[0]

        for point in points[1:]:
            lint = tuple(int(x) for x in last_point)
            pint = tuple(int(x) for x in point)
            self._svg.add(self._svg.line(start=last_point, end=point))
            last_point = point

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
                r = sdl.Rect(x, y, kwargs['size'][0], kwargs['size'][1])
                sdl.draw.rect(self._surface, self._csv_rgb2sdl_color(color, alpha),
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

            r = sdl.Rect((x, y), (w, h))
            sdl.draw.rect(self._surface, self._csv_rgb2sdl_color(color, alpha),
                         r, width)
            self._svg.add(self._svg.rect(insert=(int(x), int(y)), size=(int(w), int(h))))

            return None

    def polygon(self, points, color=Color.DEFAULT, width=1, filled=False,
                aalias=True, alpha=-1):
        """
        Draw a polygon from a list of (x, y).
        :param points:
        :param color: the object's color as a simple CVColor object, if no
                       is specified the default is used
        :param width: line width
        :param filled: the rectangle is to be filled or not
        :param alpha: the alpha blending for the object. If this value is -1
                       then the layer default value is used. A value of 255 means
                       opaque, while 0 means transparent.
        :param aalias: draw the edges of the object anti-aliased. Note this
                        does not work when the object is filled.
        :return: None
        """
        # TODO: simplify
        if filled:
            width = 0

        if not filled:
            if aalias and width == 1:
                sdl.draw.aalines(self._surface,
                                 self._csv_rgb2sdl_color(color, alpha),
                                 True, points, width)
            else:
                sdl.draw.lines(self._surface,
                               self._csv_rgb2sdl_color(color, alpha),
                               True, points, width)
        else:
            sdl.draw.polygon(self._surface,
                             self._csv_rgb2sdl_color(color, alpha),
                             points, width)
        return None

    def circle(self, center, radius, color=Color.DEFAULT, width=1, filled=False,
               alpha=-1, aalias=True):
        """
        Draw a circle given a location and a radius.
        :param center:
        :param radius:
        :param color:
        :param width:
        :param filled:
        :param alpha:
        :param aalias:
        :return:
        """
        if filled:
            width = 0
        if aalias == False or width > 1 or filled:
            sdl.draw.circle(self._surface,
                            self._csv_rgb2sdl_color(color, alpha),
                            center, int(radius), int(width))
        else:
            gfxdraw.aacircle(self._surface, int(center[0]), int(center[1]),
                             int(radius), self._csv_rgb2sdl_color(color, alpha))

        cen = tuple(int(x) for x in center)
        self._svg.add(self._svg.circle(center=cen, r=radius))

        return None

    def ellipse(self, center, size, color=Color.DEFAULT, width=1, filled=False,
                alpha=-1):
        """
        Draw an ellipse given a location and a size.
        :param center:
        :param size:
        :param color:
        :param width:
        :param filled:
        :param alpha:
        :return:
        """
        if filled:
            width = 0

        r = sdl.Rect(center[0] - (size[0] / 2), center[1] - (size[1] / 2),
                     size[0], size[1])
        sdl.draw.ellipse(self._surface, self._csv_rgb2sdl_color(color, alpha),
                         r, width)

        cen = tuple(int(x) for x in center)
        sz = tuple(int(x) for x in size)
        self._svg.add(self._svg.ellipse(center=cen, r=sz))

    def bezier(self, points, steps, color=Color.DEFAULT, alpha=-1):
        """
        Draw a bezier _curve based on a control point and a number of steps
        :param points:
        :param steps:
        :param color:
        :param alpha:
        :return:
        """
        gfxdraw.bezier(self._surface, points, steps,
                       self._csv_rgb2sdl_color(color, alpha))

    def text_size(self, text):
        """
        Get text string height and width.
        :param text:
        :return:
        """
        text_surface = self._font.render(
                text, True,
                self._csv_rgb2sdl_color(Color.WHITE, 255)
        )

        return text_surface.get_width(), text_surface.get_height()

    def text(self, text, location, color=Color.DEFAULT, alpha=-1):
        """
        Write a text string at a given location.
        :param text:
        :param location:
        :param color:
        :param alpha:
        :return:
        """
        if len(text) < 0:
            return None

        text_surface = self._font.render(text, True, self._csv_rgb2sdl_color(
            color, alpha
        ))

        if alpha == -1:
            alpha = self._default_alpha
        # this is going to be slow, dumb no active support.
        # get access to the alpha band of the image.
        pixels_alpha = sdl.surfarray.pixels_alpha(text_surface)
        # do a floating point multiply, by alpha 100, on each alpha value.
        # the truncate the values (convert to integer) and copy back
        # into the surface
        pixels_alpha[...] = (pixels_alpha * (alpha / 255.0)).astype(np.uint8)
        # unlock the surface
        del pixels_alpha
        self._surface.blit(text_surface, location)

        # adjust for web
        font_style = 'font-size: {}px;'.format(self._font_size - 7)

        if self._font_bold:
            font_style += 'font-weight: bold;'
        if self._font_italic:
            font_style += 'font-style: italic;'
        if self._font_underline:
            font_style += 'text-decoration: underline;'
        if self._font_name:
            font_style += 'text-family: \"{}\";'.format(self._font_name)

        altered_location = (location[0], location[1] + self.text_size(text)[1])
        alt = tuple(int(x) for x in altered_location)
        self._svg.add(self._svg.text(text, insert=alt, style=font_style))

    def ez_view_text(self, text, location, fgc=Color.WHITE, bgc=Color.BLACK):
        """
        :param text:
        :param location:
        :param fgc:
        :param bgc:
        :return:
        """
        if len(text) < 0:
            return None

        alpha = 255
        text_surface = self._font.render(text, True, self._csv_rgb2sdl_color(
            fgc, alpha), self._csv_rgb2sdl_color(bgc, alpha))
        self._surface.blit(text_surface, location)
        return None

    def sprite(self, img, pos=(0, 0), scale=1.0, rot=0.0, alpha=255):
        """
        sprite draws a sprite (a second small image) onto the current layer.
        The sprite can be loaded directly from a supported image file like a
        gif, jsdl, bmp, or png, or loaded as a surface or SCV image.

        :param img:
        :param pos:
        :param scale:
        :param rot:
        :param alpha:
        :return:
        """
        if not sdl.display.get_init():
            sdl.display.init()

        if isinstance(img, str):
            image = sdl.image.load(img, "RGB")
        elif isinstance(img, PhloxAR.core.image.Image):
            image = img.surface
        else:
            image = img  # we assume we have a surface

        image = image.convert(self._surface)

        if rot != 0.00:
            image = sdl.transform.rotate(image, rot)

        if scale != 1.0:
            scaled_size = (int(image.width * scale), int(image.height * scale))
            image = sdl.transform.scale(image, scaled_size)

        pixels_alpha = sdl.surfarray.pixels_alpha(image)
        pixels_alpha[...] = (pixels_alpha * (alpha / 255.0)).astype(np.uint8)
        del pixels_alpha

        self._surface.blit(image, pos)

    def blit(self, img, pos=(0, 0)):
        """
        Blit one image onto the drawing layer at upper left position
        :param img:
        :param pos:
        :return:
        """
        self._surface.blit(img.get_surface(), pos)

    def replace_overlay(self, overlay):
        """
        Allow user to set the surface manually.
        :param overlay:
        :return:
        """
        self._surface = overlay

    def clear(self):
        """
        Remove all of the drawing on this layer.
        :return:
        """
        self._surface = sdl.Surface((int(self.width), int(self.height)),
                                    flags=sdl.SRCALPHA)

    def render_to_surface(self, surface):
        """
        Blit this layer to another surface.
        :param surface:
        :return: pygame.Surface
        """
        surface.blit(self._surface, (0, 0))
        return surface

    def render_to_layer(self, other):
        """
        Add this layer to another layer.
        :param other:
        :return:
        """
        other.surface.blit(self._surface, (0, 0))

    @property
    def surface(self):
        return self._surface
