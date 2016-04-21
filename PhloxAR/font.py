# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals, absolute_import
from base import *


class Font(object):
    """
    The Font class allows you to create a font object on the image.
    TODO: STYLE
    """
    ITL = 0x0001
    BLD = 0x0010
    BLK = 0x0100
    LGT = 0x1000
    RGR = 0x0000

    _ext = '.ttf'
    _fontpath = './fonts/'
    _fontface = 'lato'
    _fontsize = 16
    # _style = RGR
    _font = None

    # download from https://github.com/google/fonts
    _fonts = [
        'Astloch',
        'Audiowide',
        'AveriaSerifLibre',
        'CaesarDressing',
        'Calligraffitti',
        'CarterOne',
        'JacquesFrancoisShadow',
        'Kranky',
        'LaBelleAurore',
        'Lato',
        'LeagueScript',
        'MarckScript',
        'Monofett',
        'Monoton',
        'NovaMono',
        'NovaScript',
        'OxygenMono',
        'ReenieBeanie',
        'ShadowsIntoLight',
        'SpecialElite',
        'UbuntuMono',
        'VastShadow',
        'VT323',
        'Wallpoet',
        'WireOne'
    ]

    def __init__(self, fontface='Lato', fontsize=16, style='regular'):
        """
        Creates a font object.
        """
        self.set_size(fontsize)
        self.set_font(fontface, style)

    def get_font(self):
        """
        Get the font from object.
        Returns: PIL Image Font
        """
        return self._font.getname()

    def set_font(self, font='Lato', style='regular'):
        """
        Set the name of the font listed in the font family or
        pass the absolute path of the truetype font file.
        """
        if not isinstance(font, basestring):
            print("Please pass a string.")
            return None

        if font in self._fonts:
            self._fontface = font
            f = self._fontpath + self._fontface + '/' + self._fontface + self._ext
            print(font_to_use)
        else:
            self._fontface = font
            f = font

        self._font = pilImageFont.truetype(f, self._fontsize)

    font = property(get_font, set_font)

    def get_size(self):
        """
        Gets the size of the current font.
        Returns: Integer
        """

        return self._fontsize

    def set_size(self, size):
        """
        Set the font point size. i.e. 16pt
        """
        print(type(size))
        if type(size) == int:
            self._fontsize = size
        else:
            print("please provide an integer.")

    size = property(get_size, set_size)

    @property
    def fonts(self):
        """
        Returns the list of builtin fonts.
        """
        return self._fonts

    def print_fonts(self):
        """
        Prints the list of buitlin fonts.
        """
        print("Builtin fonts: ")
        for f in self._fonts:
            print(f)
