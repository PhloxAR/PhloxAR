# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PIL import ImageFont as PILImageFont
from . import const


const.ITL = 0x0001
const.BLD = 0x0010
const.BLK = 0x0100
const.LGT = 0x1000
const.BGR = 0x0000


__all__ = [
    'Font'
]


class Font(object):
    """
    The Font class allows you to create a font object on the image.

    Attributes:
        _ext (string): font file extension
        _fontpath: font file path
        _fontface: font file name
        _fontsize: font size
        _font: current font
        _fonts: available fonts by default

    TODO: STYLE
    """
    # ITL = 0x0001
    # BLD = 0x0010
    # BLK = 0x0100
    # LGT = 0x1000
    # RGR = 0x0000

    _ext = '.ttf'
    _fontpath = './fonts/'
    _fontface = 'Lato'
    _fontsize = 16
    # _style = RGR
    _font = None

    # download from https://github.com/google/fonts
    _fonts = [
        'Astloch', 'Audiowide', 'AveriaSerifLibre',
        'CaesarDressing', 'Calligraffitti', 'CarterOne',
        'JacquesFrancoisShadow',
        'Kranky',
        'LaBelleAurore', 'Lato', 'LeagueScript',
        'MarckScript', 'Monofett', 'Monoton',
        'NovaMono', 'NovaScript',
        'OxygenMono',
        'ReenieBeanie',
        'ShadowsIntoLight', 'SpecialElite',
        'UbuntuMono',
        'VastShadow', 'VT323',
        'Wallpoet', 'WireOne'
    ]

    def __init__(self, fontface='Lato', fontsize=16, style='regular'):
        """
        Creates a new font object, it uses Lato as the default font
        To give it a custom font you can just pass the absolute path
        to the truetype font file.

        Args:
            fontface (string): font face or font path
            fontsize (int): size of the font
            style:
        """
        self.set_size(fontsize)
        self.set_font(fontface, style)

    def get_font(self):
        """
        Get the font from the object to be used in drawing

        Returns:
            PIL Image Font
        """
        return self._font

    def set_font(self, font='Lato'):  # , style='regular'):
        """
        Set the name of the font listed in the font family or
        pass the absolute path of the truetype font file.

        Args:
            font (string, path): font's name or path

        Returns:
            None
        """
        if not isinstance(font, str):
            print("Please pass a string.")

        if font in self._fonts:
            self._fontface = font
            f = self._fontpath + self._fontface + '/' + self._fontface + \
                self._ext
        else:
            self._fontface = font
            f = font

        self._font = PILImageFont.truetype(f, self._fontsize)

    font = property(get_font, set_font)

    @property
    def fontsize(self):
        """
        Gets the size of the current font.

        Returns:
            (int) current font's size
        """
        return self._fontsize

    @fontsize.setter
    def fontsize(self, size):
        """
        Set the font point size.

        Args:
            size (int): font size

        Returns:
            None
        """
        if isinstance(size, (int, float)):
            self._fontsize = int(size)
        else:
            print("Please provide an integer.")

    @property
    def fonts(self):
        """
        Gets the list of builtin fonts.

        Returns:
            (list) fonts' name
        """
        return self._fonts
