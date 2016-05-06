# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PhloxAR.base import *
from PhloxAR.image import *


class ColorModel(object):
    """
    The ColorModel is used to model the color of foreground and background
    objects by using a training set of images.

    You can crate the color model with any number of 'training' images, or
    add images to the model with add() and remove(). The for your data
    images, you can useThresholdImage() to return a segmented picture.
    """
    # TODO: discretize the color space into smaller intervals
    # TODO: work in HSV space
    _is_bkg = True
    _data = {}
    _bits = 1

    def __init__(self, data=None, is_bkg=True):
        self._is_bkg = is_bkg
        self._data = data
        self._bits = 1

        if data:
            try:
                [self.add(d) for d in data]
            except TypeError:
                self.add(data)

    def _make_canonical(self, data):
        """
        Turn input types in a common form used by the rest of the
        class -- a 4-bit shifted list of unique colors
        :param data:
        :return:
        """
        ret = ''

        if data.__class__.__name__ == 'Image':
            ret = data.narray().reshape(-1, 3)
        elif data.__class__.__name__ == 'cvmat':
            ret = npy.array(data).reshape(-1, 3)
        elif data.__class__.__name__ == 'list':
            tmp = []
            for d in data:
                t = (d[2], d[1], d[0])
                tmp.append(t)
            ret = npy.array(tmp, dtype='uint8')
        elif data.__class__.__name__ == 'tuple':
            ret = npy.array((data[2], data[1], data[0]), 'uint8')
        elif data.__class__.__name__ == 'npy.array':
            ret = data
        else:
            logger.warning("ColorModel: color is not in an accepted format!")
            return None

        rs = npy.right_shift(ret, self._bits)  # right shift 4 bits

        if len(rs.shape) > 1:
            uniques = npy.unique(rs.view([('', rs.dtype)] * rs.shape[1])).view(rs.dtype).reshape(-1, 3)
        else:
            uniques = [rs]

        return dict.fromkeys(map(npy.ndarray.tostring, uniques), 1)

    def reset(self):
        """
        Resets the color model, i.e., clears it out the stored values.
        :return: None
        """
        self._data = {}

    def add(self, data):
        """
        Add an image, array, or tuple to the color model.
        :param data: an image, array, or tuple of values to the color model.
        :return: None
        """
        self._data.update(self._make_canonical(data))

    def remove(self, data):
        """
        Remove an image, array, or tuple from the model.
        :param data: an image, array, or tuple of value.
        :return: None
        """
        self._data = dict.fromkeys(set(self._data) ^ set(self._make_canonical(data), 1))

    def threshold(self, image):
        """
        Perform a threshold operation on the given image. This involves
        iterating over the image and comparing each pixel to the model.
        If the pixel is in the model it is set to be either foreground(white)
        or background(black) based on the setting of _is_bkg.
        :param image: the image to perform the threshold on.
        :return: thresholded image
        """
        a = 0
        b = 255

        if self._is_bkg == False:
            a = 255
            b = 0

        # bit shift down and reshape to Nx3
        rs = npy.right_shift(image.narray(), self._bits).reshape(-1, 3)
        mapped = npy.array(map(self._data.has_key, map(npy.ndarray.tostring, rs)))
        thresh = npy.where(mapped, a, b)

        return Image(thresh.reshape(image.width, image.height))

    def contains(self, c):
        """
        Return true if a particular color is in our color model.
        :param c: a three value color tuple
        :return: True if color is in the model, otherwise False
        """
        return self._data.has_key(npy.right_shift(npy.cast['uint8'](c[::-1]),
                                                  self._bits).tostring())

    def set_is_bkg(self, is_bkg):
        """
        Set image being
        :param is_bkg:
        :return:
        """
        self._is_bkg = is_bkg

    def load(self, filename):
        """
        Load the color model from the specified file.
        :param filename: file name and path to load the data from
        :return: None
        """
        # pickle.load
        self._data = load(open(filename))

    def save(self, filename):
        """
        Save a color model file.
        :param filename: file name and path to save the dat to
        :return: None
        """
        # pickle.dump
        dump(self._data, open(filename, 'wb'))