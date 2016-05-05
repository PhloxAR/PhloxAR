#!/usr/bin/env python
# -*- coding: utf-8 -*_

import sys
import os
import svgwrite
from PhloxAR.color import *
from PhloxAR.base import pg


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
        if alpha >= 0 and alpha <= 255:
            self._default_alpha = alpha
        return None

    @property
    def layer_alpha(self):
        """
        Returns the default alpha value.
        """