# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import unicode_literals, absolute_import

from PhloxAR.base import *
from PhloxAR.image import *
import random


class Color(object):
    """
    Color is a class which stores commonly used colors.
    """
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    RED = (255, 0, 0)

    LEGO_BLUE = (0, 50, 150)
    LEGO_ORANGE = (255, 150, 40)

    VIOLET = (181, 126, 220)
    ORANGE = (255, 165, 0)
    GREEN = (0, 128, 0)
    GRAY = (128, 128, 128)

    # Extended Colors
    IVORY = (255, 255, 240)
    BEIGE = (245, 245, 220)
    WHEAT = (245, 222, 179)
    TAN = (210, 180, 140)
    KHAKI = (195, 176, 145)
    SILVER = (192, 192, 192)
    CHARCOAL = (70, 70, 70)
    NAVYBLUE = (0, 0, 128)
    ROYALBLUE = (8, 76, 158)
    MEDIUMBLUE = (0, 0, 205)
    AZURE = (0, 127, 255)
    CYAN = (0, 255, 255)
    AQUAMARINE = (127, 255, 212)
    TEAL = (0, 128, 128)
    FORESTGREEN = (34, 139, 34)
    OLIVE = (128, 128, 0)
    LIME = (191, 255, 0)
    GOLD = (255, 215, 0)
    SALMON = (250, 128, 114)
    HOTPINK = (252, 15, 192)
    FUCHSIA = (255, 119, 255)
    PUCE = (204, 136, 153)
    PLUM = (132, 49, 121)
    INDIGO = (75, 0, 130)
    MAROON = (128, 0, 0)
    CRIMSON = (220, 20, 60)
    DEFAULT = (0, 0, 0)
    # These are for the grab cut / findBlobsSmart
    BACKGROUND = (0, 0, 0)
    MAYBE_BACKGROUND = (64, 64, 64)
    MAYBE_FOREGROUND = (192, 192, 192)
    FOREGROUND = (255, 255, 255)
    WATERSHED_FG = (255, 255, 255)  # Watershed foreground
    WATERSHED_BG = (128, 128, 128)  # Watershed background
    WATERSHED_UNSURE = (0, 0, 0)

    colors = [BLACK,
              WHITE,
              BLUE,
              YELLOW,
              RED,
              VIOLET,
              ORANGE,
              GREEN,
              GRAY,
              IVORY,
              BEIGE,
              WHEAT,
              TAN,
              KHAKI,
              SILVER,
              CHARCOAL,
              NAVYBLUE,
              ROYALBLUE,
              MEDIUMBLUE,
              AZURE,
              CYAN,
              AQUAMARINE,
              TEAL,
              FORESTGREEN,
              OLIVE,
              LIME,
              GOLD,
              SALMON,
              HOTPINK,
              FUCHSIA,
              PUCE,
              PLUM,
              INDIGO,
              MAROON,
              CRIMSON,
              DEFAULT
              ]

    @classmethod
    def random(cls):
        """
        :return: a random color tuple.
        """
        r = random.randint(1, len(cls.colors) - 1)
        return cls.colors[r]

    @classmethod
    def rgb_to_hsv(cls, t):
        """
        Convert a rgb color to HSV, OpenCV style (0-180 for hue)
        :param t: an rgb tuple to convert to HSV.
        :return: a color tuple in HSV format.
        """
        hsv = colorsys.rgb_to_hsv(*t)
        return hsv[0] * 180, hsv[1] * 255, hsv[2]

    @classmethod
    def hue_from_rgb(cls, t):
        """
        Get corresponding Hue value of the given RGB value.
        :param t: an rgb tuple to convert to HSV.
        :return: a color tuple in HSV format.
        """
        hue = colorsys.rgb_to_hsv(*t)[0]
        return hue * 180

    @classmethod
    def hue_to_rgb(cls, h):
        """
        Get corresponding RGB values of the given hue.
        :param h: a hue int to convert to RGB.
        :return: a color tuple in RGB format.
        """
        h /= 180.0
        r, g, b = colorsys.hsv_to_rgb(h, 1, 1)

        return round(255.0 * r), round(255.0 * g), round(255.0 * b)

    @classmethod
    def hue_to_bgr(cls, h):
        """
        Get corresponding BGR values of the given hue.
        :param h: a hue int to convert to BGR.
        :return: a color tuple in BGR format.
        """
        return reversed(cls.hue_to_rgb(h))

    @classmethod
    def average_rgb(cls, rgb):
        """
        Get the average of the R, G, B values
        :param rgb: a tuple of RGB values.
        :return: average of RGB.
        """
        return int((rgb[0] + rgb[1] + rgb[2]) / 3)

    @classmethod
    def lightness(cls, rgb):
        """
        Calculate the grayscale value of R, G, B according to lightness method.
        :param rgb: a tuple of RGB values.
        :return: grayscale value.
        """
        return int((max(rgb) + min(rgb)) / 2)

    @classmethod
    def luminosity(cls, rgb):
        """
        Calculate the grayscale value of R, G, B according to luminosity method.
        :param rgb: a tuple of RGB values.
        :return: grayscale value.
        """
        return int((0.21 * rgb[0] + 0.71 * rgb[1] + 0.07 * rgb[2]))


class ColorCurve(object):
    """
    ColorCurve is a color spline class for performing color correction.
    It can takes a Scipy Univariate spline as parameter, or an array with
    at least 4 point pairs.
    Either of these must map in a 255x255 space.  The curve can then be
    used in the applyRGBCurve, applyHSVCurve, and applyIntensityCurve functions.

    Note:
    The points should be in strictly increasing order of their first elements
    (X-coordinates)

    the only property, curve is a linear array with 256 elements from 0 to 255
    """
    curve = ''

    def __init__(self, vals):
        interval = linspace(0, 255, 256)
        if type(vals) == UnivariateSpline:
            self.curve = vals(interval)
        else:
            vals = npy.array(vals)
            spline = UnivariateSpline(vals[:, 0], vals[:, 1], s=1)
            self.curve = npy.maximum(npy.minimum(spline(interval), 255), 0)


class ColorMap(object):
    """
    ColorMap takes a tuple of colors along with the start and end points
    ant it lets you map colors with a range of numbers.
    """
    color = ()
    end_color = ()
    start_map = 0
    end_map = 0
    color_distance = 0
    value_range = 0

    def __init__(self, color, start_map, end_map):
        """

        :param color:
        :param start_map:
        :param end_map:
        """
