# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import unicode_literals, absolute_import

from scipy.interpolate import UnivariateSpline
import PhloxAR.core.image
import numpy as np
import colorsys
import random
import pickle


__all__ = [
    'Color', 'ColorSpace', 'ColorCurve', 'ColorMap', 'ColorModel'
]


class ColorSpace(object):
    UNKNOWN = 0
    RGB = 1
    BGR = 2
    GRAY = 3
    HLS = 4
    HSV = 5
    XYZ = 6
    YCrCb = 7


class Color(object):
    """
    Color is a class which stores commonly used colors.

    Default _color space is RGB.
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

    colors = [
        BLACK, WHITE, BLUE, YELLOW, RED, VIOLET, ORANGE, GREEN, GRAY, IVORY,
        BEIGE, WHEAT, TAN, KHAKI, SILVER, CHARCOAL, NAVYBLUE, ROYALBLUE,
        MEDIUMBLUE, AZURE, CYAN, AQUAMARINE, TEAL, FORESTGREEN, OLIVE, LIME,
        GOLD, SALMON, HOTPINK, FUCHSIA, PUCE, PLUM, INDIGO, MAROON, CRIMSON,
        DEFAULT,
    ]

    @classmethod
    def random(cls):
        """
        Generate a random RGB _color.

        Returns:
            (tuple) a _color.

        Examples:
            >>> _color = Color.random()
        """
        r = random.randint(1, len(cls.colors) - 1)
        return cls.colors[r]

    @classmethod
    def rgb2hsv(cls, color):
        """
        Convert a RGB _color to HSV _color.

        Args:
            color (tuple): RGB _color

        Returns:
            (tuple) a _color in HSV

        Examples:
            >>> _color = Color.random()
            >>> print(_color)
            >>> _color = Color.rgb2hsv(_color)
            >>> print(_color)
        """
        hsv = colorsys.rgb_to_hsv(*color)
        return hsv[0] * 180, hsv[1] * 255, hsv[2]

    @classmethod
    def hue(cls, color):
        """
        Get corresponding Hue value of the given RGB value.

        Args:
            color (tuple): an RGB _color to be converted

        Returns:
            (tuple) a _color in HSV

        Examples:
            >>> _color = Color.random()
            >>> print(_color)
            >>> hue = Color.hue(_color)
            >>> print(hue)
        """
        hue = colorsys.rgb_to_hsv(*color)[0]
        return hue * 180

    @classmethod
    def hue2rgb(cls, hue):
        """
        Get corresponding RGB value of the given hue.

        Args:
            hue (int, float): the hue to be convert

        Returns:
            (tuple) a _color in RGB

        Examples:
            >>> _color = Color.random()
            >>> print(_color)
            >>> hue = Color.hue(_color)
            >>> color1 = Color.hue2rgb(hue)
            >>> print(color1)
        """
        hue /= 180.0
        r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)

        return round(255.0 * r), round(255.0 * g), round(255.0 * b)

    @classmethod
    def hue2bgr(cls, hue):
        """
        Get corresponding BGR value of the given hue

        Args:
            hue (int, float): the hue to be convert

        Returns:
            (tuple) a _color in BGR

        Examples:
            >>> _color = Color.random()
            >>> print(_color)
            >>> hue = Color.hue(_color)
            >>> print(hue)
            >>> color_bgr = Color.hue2bgr(hue)
            >>> print(color_bgr)
        """
        return reversed(cls.hue2rgb(hue))

    @classmethod
    def average(cls, color):
        """
        Averaging a _color.

        Args:
            color (tuple): the _color to be averaged.

        Returns:
            (tuple) averaged _color.

        Examples:
            >>> _color = Color.random()
            >>> print(_color)
            >>> color_averaged = Color.average(_color)
        """
        return int((color[0] + color[1] + color[2]) / 3)

    @classmethod
    def lightness(cls, rgb):
        """
        Calculate the grayscale value of R, G, B according to lightness method.

        Args:
            rgb (tuple): RGB values

        Returns:
            (int) grayscale value

        Examples:
            >>> _color = Color.random()
            >>> print(_color)
            >>> lightness = Color.lightness(_color)
            >>> print(lightness)
        """
        return int((max(rgb) + min(rgb)) / 2)

    @classmethod
    def luminosity(cls, rgb):
        """
        Calculate the grayscale value of R, G, B according to luminosity method.

        Args:
            rgb (tuple): RGB values

        Returns:
            (int) grayscale value

        Examples:
            >>> _color = Color.random()
            >>> print(_color)
            >>> luminosity = Color.luminosity(_color)
            >>> print(luminosity)
        """
        return int((0.21 * rgb[0] + 0.71 * rgb[1] + 0.07 * rgb[2]))


class ColorCurve(object):
    """
    ColorCurve is a _color spline class for performing _color correction.
    It takes a Scipy Univariate spline as parameter, or an array with
    at least 4 point pairs. Either of these must map in a 255x255 space.

    Attributes:
        _curve: a linear array with 256 elements from 0 to 255

    Notes:
    The points should be in strictly increasing order of their first elements
    (X-coordinates)

    Examples:
        >>> curve = ColorCurve([[0,0], [100, 120], [180, 230], [255, 255]])
        >>> img = Image('lena')
        >>> img.apply_intensity_curve(curve)
        >>> img.show()
    """
    _curve = None

    def __init__(self, vals):
        interval = np.linspace(0, 255, 256)
        if isinstance(vals, UnivariateSpline):
            self._curve = vals(interval)
        else:
            vals = np.array(vals)
            spline = UnivariateSpline(vals[:, 0], vals[:, 1], s=1)
            self._curve = np.maximum(0, np.minimum(spline(interval), 255))


class ColorMap(object):
    """
    ColorMap takes a tuple of colors along with the start and end points
    and it lets you map colors with a range of numbers.

    Attributes:
        _color: list of color tuple which need to be mapped
        _start: starting of the range of the number which we map the colors
        _end: end of the range of the number which we map the colors
        _delta: number changes over the color's number
        _val_range: number range

    Examples:
        TODO
    """
    _color = None
    _start = 0
    _end = 0
    _delta = 0
    _val_range = 0

    def __init__(self, color, start, end):
        self._color = np.array(color)

        if self._color.ndim == 1:  # To check if only one _color was passed.
            color = ((color[0], color[1], color[2]), Color.WHITE)
            self._color = np.array(color)

        self._start = float(start)
        self._end = float(end)
        self._val_range = float(end - start)
        self._delta = self._val_range / float(len(self._color) - 1)

    def __getitem__(self, key):
        if key > self._end:
            key = self._end
        elif key < self._start:
            key = self._start

        val = (key - self._start) / self._delta
        alpha = float(val - int(val))
        idx = int(val)

        if idx == len(self._color) - 1:
            color = tuple(self._color[idx])
            return int(color[0]), int(color[1]), int(color[2])

        color = tuple(self._color[idx] * (1 - alpha) + self._color[idx + 1] * alpha)

        return int(color[0]), int(color[1]), int(color[2])


class ColorModel(object):
    """
    The ColorModel is used to model the _color of foreground and background
    objects by using a training set of images.

    You can crate the ColorModel with any number of 'training' images, or
    add images to the model with add() and remove(). The for your data
    images, you can useThresholdImage() to return a segmented picture.
    """
    # TODO: discretize the _color space into smaller intervals
    # TODO: work in HSV space
    _isbkg = True
    _data = {}
    _bits = 1

    def __init__(self, data=None, is_bkg=True):
        self._isbkg = is_bkg
        self._data = data
        self._bits = 1

        if data:
            try:
                [self.add(d) for d in data]
            except TypeError:
                self.add(data)

    def _make_canonical(self, data):
        """
        Turn input types in a common form used by the rest of the class,
        a 4-bit shifted list of unique colors

        Args:
            data (Image, list, tuple, np.array): input data

        Returns:
            shifted list of colors
        """
        if data.__class__.__name__ == 'Image':
            ret = data.narray().reshape(-1, 3)
        elif data.__class__.__name__ == 'list':
            tmp = []

            for d in data:
                t = (d[2], d[1], d[0])
                tmp.append(t)

            ret = np.array(tmp, dtype='uint8')
        elif data.__class__.__name__ == 'tuple':
            ret = np.array((data[2], data[1], data[0]), 'uint8')
        elif data.__class__.__name__ == 'np.array':
            ret = data
        else:
            logger.warning("ColorModel: _color is not in an accepted format!")
            return None

        rs = np.right_shift(ret, self._bits)  # right shift 4 bits

        if len(rs.shape) > 1:
            uniques = np.unique(rs.view([('', rs.dtype)] * rs.shape[1])).view(rs.dtype).reshape(-1, 3)
        else:
            uniques = [rs]

        return dict.fromkeys(map(np.ndarray.tostring, uniques), 1)

    def reset(self):
        """
        Resets the ColorModel, i.e., clears it out the stored values.

        Returns:
            None
        """
        self._data = {}

    def add(self, data):
        """
        Add an image, array, or tuple to the ColorModel.

        Args:
            data: an image, array, or tuple of values to the ColorModel

        Returns:
            None

        Examples:
            >>> model = ColorModel()
            >>> model.add(Image('lena'))
            >>> model.
        """
        self._data.update(self._make_canonical(data))

    def remove(self, data):
        """
        Remove an image, array, or tuple from the model.

        Args:
            data: an image, array, or tuple of values to the ColorModel

        Returns:
            None

        Examples:
            >>> model = ColorModel()
            >>> model.add(Image('lena'))
            >>> model.remove(Color.BLACK)
        """
        self._data = dict.fromkeys(set(self._data) ^ set(
                self._make_canonical(data)), 1)

    def threshold(self, image):
        """
        Perform a threshold operation on the given image. This involves
        iterating over the image and comparing each pixel to the model.
        If the pixel is in the model it is set to be either foreground(white)
        or background(black) based on the setting of _isbkg.

        Args:
            image(Image): the image to perform the threshold on

        Returns:
            (Image) thresholded image.

        Examples:
            >>> model = ColorModel()
            >>> model.add(Color.CYAN)
            >>> model.add(Color.FOREGROUND)
            >>> res = model.threshold(Image('lena'))
            >>> res.show()
        """
        a = 0
        b = 255

        if self._isbkg is False:
            a = 255
            b = 0

        # bit shift down and reshape to Nx3
        rs = np.right_shift(image.narray(), self._bits).reshape(-1, 3)
        # TODO: replace has_key
        mapped = np.array(map(self._data.has_key, map(np.ndarray.tostring, rs)))
        thresh = np.where(mapped, a, b)

        return PhloxAR.core.image.Image(thresh.reshape(image.width, image.height))

    def contains(self, color):
        """
        Decides whether a particular color is in our ColorModel.

        Args:
            color (tuple): a three value Color tuple

        Returns:
            (bool) True if the color is in the model, otherwise False

        Examples:
            >>> model = ColorModel()
            >>> model.add(Color.CYAN)
            >>> model.add(Color.BLACK)
            >>> if model.contains(Color.BLACK):
            >>>     print("Got it!")
        """
        color = np.right_shift(np.cast['uint8'](color[::-1]), self._bits).tostring()
        return color in self._data

    @property
    def isbkg(self):
        """
        Returns ColorModel's isbkg statue

        Returns:
            (bool) isbkg value.
        """
        return self._isbkg

    @isbkg.setter
    def isbkg(self, isbkg):
        """
        Set our model as being foreground or background imagery. I.e. things in
        the model are the foreground and will be marked as white during the
        threshold operation, also things in the model are the background will
        be marked as black.

        Args:
            isbkg (bool): True or False

        Returns:
            None
        """
        self._isbkg = isbkg

    def load(self, filename):
        """
        Load the ColorModel from the specified file.

        Args:
            filename (string): pickled ColoModel file

        Returns:
            if success return the ColorModel data else return None

        Examples:
            >>> model = ColorModel()
            >>> model.load('tmp_color_model.txt')
            >>> model.add(Color.AQUAMARINE, Color.RED, Color.BLACK)
            >>> model.save('my_color_model.txt')
        """
        self._data = pickle.load(open(filename))

    def save(self, filename):
        """
        Save a ColorModel file.

        Args:
            filename (string): file name and path to save the data to

        Returns:
            None

        Examples:
            >>> model = ColorModel()
            >>> model.add(Color.BEIGE, Color.CHARCOAL, Color.AQUAMARINE)
            >>> model.save('my_color_model.txt')
        """
        pickle.dump(self._data, open(filename, 'wb'))

