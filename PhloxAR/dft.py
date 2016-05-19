# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals
from __future__ import division, print_function

from PhloxAR.base import np, warnings
from PhloxAR.core.image import Image


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
    _numpy_array       - numpy array of the filter
    _image       - SimpleCV.Image of the filter
    _dia         - diameter of the filter
                      (applicable for gaussian, butterworth, notch)
    _type        - Type of the filter
    _order       - order of the butterworth filter
    _fpass   - frequency of the filter (low pass, high pass, band pass)
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
    _numpy_array = None
    _image = None
    _dia = 0
    _type = ''
    _order = 0
    _fpass = 0
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
            elif key == 'narray':
                self._numpy_array = kwargs[key]
            elif key == 'image':
                self._image = kwargs[key]
            elif key == 'dia':
                self._dia = kwargs[key]
            elif key == 'type':
                self._type = kwargs[key]
            elif key == 'order':
                self._order = kwargs[key]
            # frequency
            elif key == 'fpass':
                self._fpass = kwargs[key]
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
        if not isinstance(other, type(self)):
            warnings.warn("Provide PhloxAR.DFT object.")
            return None

        if self.size() != other.size():
            warnings.warn("Both PhloxAR.DFT object mush have the same size.")
            return None

        numpy_array = self._numpy_array + other._numpy_array
        image = Image(numpy_array)
        ret = DFT(array=numpy_array, image=image, size=image.size())

        return ret

    def __invert__(self):
        return self.invert()

    def _update(self, flt):
        """
        Update filter's params.
        :param flt: used to update
        :return: None
        """
        self.channels = flt.channels
        self._dia = flt._dia
        self._type = flt._type
        self._order = flt._order
        self._freq_pass = flt._fpass
        self._x_cutoff_high = flt._x_cutoff_high
        self._x_cutoff_low = flt._x_cutoff_low
        self._y_cutoff_high = flt._y_cutoff_high
        self._y_cutoff_low = flt._y_cutoff_low

    @classmethod
    def create_filter(cls, ftype, **kwargs):
        """
        Create a filter according to specific type.
        :param ftype: determines the shape of the filter and can be
                      'gaussian', 'butterworth', 'notch', 'bandpass',
                      'lowpass', 'highpass'
        :param kwargs: other parameters
        :return: DFT filter
        """
        if isinstance(ftype, str):
            if isinstance(kwargs['dia'], list):
                if len(kwargs['dia']) != 3 and len(kwargs['dia']) != 1:
                    warnings.warn("Diameter list must be of size 1 or 3")
                    return None
                if len(kwargs['dia']) == 1:
                    kwargs['dia'] = kwargs['dia'][0]
            if ftype == 'gaussian':
                return cls.gaussian(**kwargs)
            elif ftype == 'butterworth':
                return cls.butterworth(**kwargs)
            elif ftype == 'notch':
                return cls.notch(**kwargs)
            elif ftype == 'bandpass':
                return cls.band_pass(**kwargs)
            elif ftype == 'lowpass':
                return cls.band_pass(**kwargs)
            elif ftype == 'highpass':
                return cls.band_pass(**kwargs)
            else:
                warnings.warn("Not invalid filter type.")
                return None
        else:
            warnings.warn("String type.")
            return None

    @classmethod
    def gaussian(cls, dia=400, size=(64, 64), fpass='low'):
        """
        Create a gaussian filter of given size.
        :param dia: (int) diameter of Gaussian filter
                     (list) provide a list of three diameters to
                     create a 3 channel filter
        :param size: size of the filter (width, height)
        :param fpass: 'high' - high-pass filter
                      'low' - low-pass filter
        :return: DFT filter.
        """
        stacked_filter = DFT()

        if isinstance(dia, list):
            for d in dia:
                stacked_filter = stacked_filter._stack_filters(cls.gaussian(d, size, fpass))

            image = Image(stacked_filter._numpy_array)
            ret_val = DFT(narray=stacked_filter._numpy_array, image=image,
                      dia=dia, channels=len(dia), size=size, type='gaussian',
                      fpass=stacked_filter._freq_pass)
            return ret_val
        else:
            cls._fpass = fpass
            sx, sy = size
            x0 = sx / 2
            y0 = sy / 2
            x, y = np.meshgrid(np.arange(sx), np.arange(sy))
            d = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            flt = 255 * np.exp(-0.5 * (d / dia) ** 2)

            if fpass == 'high':
                flt = 255 - flt

            image = Image(flt)
            ret_val = DFT(size=size, narray=flt, image=image, dia=dia,
                          type='gaussian', fpass=fpass)

    @classmethod
    def butterworth(cls, dia=400, size=(64, 64), order=2, fpass='low'):
        """
        Create a butterworth filter of given size and order.
        :param dia: (int) diameter of Gaussian filter
                     (list) provide a list of three diameters to create
                     a 3 channel filter
        :param size: size of the filter (width, height)
        :param order: order of the filter
        :param fpass: 'high' - high-pass filter
                      'low' - low-pass filter
        :return: DFT filter
        """
        if isinstance(dia, list):
            for d in dia:
                stacked_filter = stacked_filter._stack_filters(
                        cls.butterworth(d, size, fpass))

            image = Image(stacked_filter._numpy_array)
            ret_val = DFT(narray=stacked_filter._numpy_array, image=image,
                          dia=dia, channels=len(dia), size=size,
                          type='butterworth', fpass=stacked_filter._freq_pass)
            return ret_val
        else:
            cls._fpass = fpass
            sx, sy = size
            x0 = sx / 2
            y0 = sy / 2
            x, y = np.meshgrid(np.arange(sx), np.arange(sy))
            d = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            flt = 255 / (1.0 + (d / dia)**(order * 2))

            if fpass == 'high':
                flt = 255 - flt

            image = Image(flt)
            ret_val = DFT(size=size, narray=flt, image=image, dia=dia,
                          type='butterworth', fpass=fpass)

            return ret_val

    @classmethod
    def low_pass(cls, xco, yco=None, size=(64, 64)):
        """
        Create a low pass filter of given size.
        :param xco: x cutoff
                    (int) horizontal cutoff frequency
                    (list) provide a list of three cut off frequencies
                    to create a 3 channel filter
        :param yco: y cutoff
                    (int) vertical cutoff frequency
                    (list) provide a list of three cutoff frequencies
                    to create a 3 channel filter
        :param size: size of the filter (width, height)
        :return: DFT filter
        """
        if isinstance(xco, list):
            if len(xco) != 3 and len(xco) != 1:
                warnings.warn("xco list must be of size 3 or 1")
                return None
            if isinstance(yco, list):
                if len(yco) != 3 and len(yco) != 1:
                    warnings.warn("yco list must be of size 3 or 1")
                    return None
                if len(yco) == 1:
                    yco = [yco[0]] * len(xco)
            else:
                yco = [yco] * len(xco)

            stacked_filter = DFT()

            for xfreq, yfreq in zip(xco, yco):
                stacked_filter = stacked_filter._stack_filters(cls.low_pass(
                        xfreq, yfreq, size))

            image = Image(stacked_filter._numpy_array)
            retVal = DFT(narray=stacked_filter._numpy_array, image=image,
                         xco_low=xco, yco_low=yco, channels=len(xco), size=size,
                         type=stacked_filter._type, order=cls._order,
                         fpass=stacked_filter._fpass)
            return retVal

        w, h = size
        xco = np.clip(int(xco), 0, w / 2)

        if yco is None:
            yco = xco

        yco = np.clip(int(yco), 0, h / 2)
        flt = np.zeros((w, h))
        flt[0:xco, 0:yco] = 255
        flt[0:xco, h - yco:h] = 255
        flt[w - xco:w, 0:yco] = 255
        flt[w - xco:w, h - yco:h] = 255
        img = Image(flt)
        lowpass_filter = DFT(size=size, narray=flt, image=img, type="lowpass",
                      xco_low=xco, yco_low=yco, fpass="lowpass")
        return lowpass_filter


    @classmethod
    def high_pass(cls, xco, yco=None, size=(64, 64)):
        """
        Creates a high pass filter of given size and order.
        :param xco: x cutoff
                    (int) horizontal cutoff frequency
                    (list) provide a list of three cut off frequencies
                    to create a 3 channel filter
        :param yco: y cutoff
                    (int) vertical cutoff frequency
                    (list) provide a list of three cutoff frequencies
                    to create a 3 channel filter
        :param size: size of the filter (width, height)
        :return: DFT filter
        """

        if isinstance(xco, list):
            if len(xco) != 3 and len(xco) != 1:
                warnings.warn("xco list must be of size 3 or 1")
                return None
            if isinstance(yco, list):
                if len(yco) != 3 and len(yco) != 1:
                    warnings.warn("yco list must be of size 3 or 1")
                    return None
                if len(yco) == 1:
                    yco = [yco[0]] * len(xco)
            else:
                yco = [yco] * len(xco)

            stacked_filter = DFT()

            for xfreq, yfreq in zip(xco, yco):
                stacked_filter = stacked_filter._stack_filters(cls.high_pass(
                        xfreq, yfreq, size))

            image = Image(stacked_filter._numpy_array)
            retVal = DFT(narray=stacked_filter._numpy_array, image=image,
                         xco_low=xco, yco_low=yco, channels=len(xco), size=size,
                         type=stacked_filter._type, order=cls._order,
                         fpass=stacked_filter._fpass)
            return retVal

        lowpass = cls.low_pass(xco, yco, size)
        w, h = lowpass.size()
        flt = lowpass._numpy_array
        flt = 255 - flt
        img = Image(flt)
        highpass_filter = DFT(size=size, narray=flt, image=img,
                              type="highpass", xco_high=xco, yco_high=yco,
                              fpass="highpass")
        return highpass_filter

    @classmethod
    def band_pass(cls, xco_low, xco_high, yco_low=None, yco_high=None,
                  size=(64, 64)):
        """
        Create a band filter of given size and order.
        Creates a high pass filter of given size and order.
        :param xco_low: (int) horizontal cutoff frequency
                        (list) provide a list of three cut off frequencies
                        to create a 3 channel filter
        :param yco: (int) vertical cutoff frequency
                    (list) provide a list of three cutoff frequencies
                    to create a 3 channel filter
        :param size: size of the filter (width, height)
        :return: DFT filter
        """

        lowpass = cls.low_pass(xco_low, yco_low, size)
        highpass = cls.high_pass(xco_high, yco_high, size)
        lowpassnumpy = lowpass._numpy_array
        highpassnumpy = highpass._numpy_array
        bandpassnumpy = lowpassnumpy + highpassnumpy
        bandpassnumpy = np.clip(bandpassnumpy, 0, 255)
        img = Image(bandpassnumpy)
        bandpass = DFT(size=size, image=img, narray=bandpassnumpy,
                       type="bandpass", xco_low=xco_low, yco_low=yco_low,
                       xco_high=xco_high, yco_high=yco_high, fpass="bandpass",
                       channels=lowpass.channels)
        return bandpass

    @classmethod
    def notch(cls, dia1, dia2=None, cen=None, size=(64, 64), ftype='lowpass'):
        """
        Creates a disk shaped notch filter of given diameter at given center.
        :param dia1: (int) diameter of the disk shaped notch
                      (list) provide a list of three diameters to create a
                      3 channel filter
        :param dia2: (int) outer diameter of the disk shaped notch used for
                      bandpass filter
                      (list) provide a list of three diameters to create a
                      3 channel filter
        :param cen: tuple (x, y) center of the disk shaped notch
        :param size: size of the filter (width, height)
        :param ftype: lowpass or highpass filter
        :return:
        """
        if isinstance(dia1, list):
            if len(dia1) != 3 and len(dia1) != 1:
                warnings.warn("Diameter list must be of size 1 or 3")
                return None

            if isinstance(dia2, list):
                if len(dia2) != 3 and len(dia2) != 1:
                    warnings.warn("diameter list must be of size 3 or 1")
                    return None
                if len(dia2) == 1:
                    dia2 = [dia2[0]] * len(dia1)
            else:
                dia2 = [dia2] * len(dia1)

        if isinstance(cen, list):
            if len(cen) != 3 and len(cen) != 1:
                warnings.warn("center list must be of size 3 or 1")
                return None
            if len(cen) == 1:
                cen = [cen[0]] * len(dia1)
        else:
            cen = [cen] * len(dia1)

        stacked_filter = DFT()

        for d1, d2, c in zip(dia1, dia2, cen):
            stacked_filter = stacked_filter._stack_filters(cls.notch(d1, d2,
                                                                     c, size,
                                                                     ftype))
        image = Image(stacked_filter._numpy)
        ret_val = DFT(narray=stacked_filter._numpy_array, image=image,
                     dia=dia1 + dia2, channels=len(dia1), size=size,
                     type=stacked_filter._type, fpass=stacked_filter._fpass)

        return ret_val

    def apply_filter(self, image, grayscale=False):
        """
        Apply the DFT filter to given image.
        :param image: PhloxAR.Image
        :param grayscale: if True, perform the operation on the gray version
                           of the image, if False, perform the operation on
                           each channel and the recombine then to create
                           the result.
        :return: filtered image.
        """
        if self.width == 0 or self.height == 0:
            warnings.warn("Empty filter. Returning the image.")
            return image

        w, h = image.size()

        if grayscale:
            image = image.to_gray()

        img = self._image

        if img.size() != image.size():
            img = img.resize(w, h)

        filtered = image.apply_DFT_filter(img)

        return filtered

    def invert(self):
        """
        Invert the filter. All values will be subtracted from 255.
        :return: inverted filter
        """
        flt = self._numpy_array
        flt = 255 - flt
        image = Image(flt)
        inverted = DFT(array=flt, image=image, size=self.size(), type=self._type)
        inverted._update(self)

        return inverted

    @property
    def image(self):
        if self._image is None:
            if self._numpy_array is None:
                warnings.warn("Filter doesn't contain any image.")
            self._image = Image(self._numpy_array)
        return self._image

    @property
    def narray(self):
        """
        Get the numpy array of the filter
        :return: numpy array of the filter
        """
        if self._numpy_array is None:
            if self._image is None:
                warnings.warn("Filter doesn't contain any image. ")
            self._numpy_array = self._image.narray()
        return self._numpy_array

    @property
    def order(self):
        """
        Get order of the butterworth filter
        :return: order of the butterworth filter
        """
        return self._order

    @property
    def dia(self):
        """
        Get diameter of the filter.
        :return: diameter of the filter
        """
        return self._dia

    @property
    def type(self):
        """
        Get type of the filter.
        :return: type of the filter
        """
        return self._type

    def stack_filters(self, flt1, flt2):
        """
        Stack three single channel filters of the same size to create
        a 3 channel filter.
        :param flt1: second filter to be stacked
        :param flt2: third filter to be stacked
        :return: DFT filter
        """
        if not (self.channels == 1 and flt1.channels == 1 and flt2.channels == 1):
            warnings.warn("Filters must have only 1 channel.")
            return None

        if not (self.size() == flt1.size() and self.size() == flt2.size()):
            warnings.warn("All the filters must be of the same size.")
            return None

        numpy_filter = self._numpy_array
        numpy_filter1 = flt1._numpy_array
        numpy_filter2 = flt2._numpy_array
        flt = np.dstack((numpy_filter, numpy_filter1, numpy_filter2))
        image = Image(flt)
        stacked_filter = DFT(size=self.size(), array=flt, image=image, channels=3)

        return stacked_filter

    def size(self):
        pass

    def _stack_filters(self, flt1):
        """
        Stack two filters of same size.
        :param flt1: second filter to be stacked.
        :return: DFT filter
        """
        if isinstance(self._numpy_array, type(None)):
            return flt1

        if not self.size() == flt1.size():
            warnings.warn("All the filters must be of same size.")
            return None

        numpy_array = self._numpy_array
        numpy_array1 = flt1._numpy_array
        flt = np.dstack((numpy_array, numpy_array1))
        stacked_filter = DFT(size=self.size(), array=numpy_array,
                              channels=self.channels+flt1.channels,
                              type=self._type, frequence=self._freq_pass)

        return stacked_filter
