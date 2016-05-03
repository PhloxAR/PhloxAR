# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PhloxAR.base import npy, warnings
from PhloxAR.image import Image


class DFT(object):
    """
    The DFT class is the refactored class to create DFT filters which can be
    used to filter images by applying Discrete Fourier Transform. This is a
    factory class to create various DFT filter.

    Any of the following parameters can be supplied to create
    a simple DFT object.
    width        - width of the filter
    height       - height of the filter
    channels     - number of channels of the filter
    size         - size of the filter (width, height)
    _numpy       - numpy array of the filter
    _image       - SimpleCV.Image of the filter
    _dia         - diameter of the filter
                      (applicable for gaussian, butterworth, notch)
    _type        - Type of the filter
    _order       - order of the butterworth filter
    _freq_pass   - frequency of the filter (lowpass, highpass, bandpass)
    _x_cutoff_low  - Lower horizontal cut off frequency for lowpassfilter
    _y_cutoff_low  - Lower vertical cut off frequency for lowpassfilter
    _x_cutoff_high - Upper horizontal cut off frequency for highpassfilter
    _y_cutoff_high - Upper vertical cut off frequency for highassfilter

    Example:
    >>> gauss = DFT.create_filter('gaussian', dia=40, size=(64, 64))
    """
    width = 0
    height = 0
    channels = 1
    _numpy = None
    _image = None
    _dia = 0
    _type = ''
    _order = 0
    _freq_pass = 0
    _x_cutoff_low = 0
    _x_cutoff_high = 0
    _y_cutoff_low = 0
    _y_cutoff_high = 0

    def __init__(self, **kwargs):
        for key in kwargs:
            if key == 'width':
                self.width = kwargs[key]
            elif key == 'height':
                self.height = kwargs[key]
            elif key == 'channels':
                self.channels = kwargs[key]
            elif key == 'size':
                self.width, self.height = kwargs[key]
            # numpy array
            elif key == 'array':
                self._numpy = kwargs[key]
            elif key == 'image':
                self._image = kwargs[key]
            elif key == 'dia':
                self._dia = kwargs[key]
            elif key == 'type':
                self._type = kwargs[key]
            elif key == 'order':
                self._order = kwargs[key]
            # frequency
            elif key == 'freq':
                self._freq_pass = kwargs[key]
            elif key == 'x_cutoff_low':
                self._x_cutoff_low = kwargs[key]
            elif key == 'y_cutoff_low':
                self._y_cutoff_low = kwargs[key]
            elif key == 'x_cutoff_high':
                self._x_cutoff_high = kwargs[key]
            elif key == 'y_cutoff_high':
                self._y_cutoff_high = kwargs[key]

    def __repr__(self):
        return ('<PhloxAR.dft object: {} {} filter of size({}, {}) and '
                'channels: %d>'.format(self._type, self._freq_pass, self.width,
                                       self.height, self.channels))

    def __add__(self, other):
        pass

    def __invert__(self):
        pass

    @classmethod
    def create_filter(cls, type, **kwargs):
        """

        :param type: determines the shape of the filter and can be
                      'average', 'disk', 'gaussian', 'log', 'laplacian'
                       'unsharp', 'motion', 'sobel', 'prewitt', 'kirsch'
        :param kwargs:
        :return:
        """
        pass

    def apply_filter(self, image, grayscale=False):
        pass

    def invert(self):
        pass

    def get_image(self):
        pass

    def get_numpy(self):
        pass

    def get_order(self):
        pass

    def get_dia(self):
        pass

    def get_type(self):
        pass

    def stack_filters(self, flt1, flt2):
        pass

    def size(self):
        pass

    def _stack_filters(self, flt1):
        pass


