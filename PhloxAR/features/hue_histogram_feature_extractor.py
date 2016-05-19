# -*- coding: utf-8 -*-
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PhloxAR.base import *
from PhloxAR.features.feature_extractor_base import FeatureExtractorBase


__all__ = [
    'HueHistogramFeatureExtractor'
]


class HueHistogramFeatureExtractor(FeatureExtractorBase):
    """
    Create a Hue Histogram feature extractor. This feature extractor
    takes in an image, gets the hue channel, bins the number of pixels
    with a particular Hue, and returns the results.
    _nbins - the number of Hue bins.
    """
    _nbins = 16

    def __init__(self, bins=16):
        # we define the black (positive) and white (negative) regions of an image
        # to get our haar wavelet
        self._nbins = bins

    def extract(self, img):
        """
        This feature extractor takes in a _color image and returns a normalized _color
        histogram of the pixel counts of each hue.
        """
        img = img.toHLS()
        h = img.getEmpty(1)
        cv.Split(img.getBitmap(), h, None, None, None)
        npa = npy.array(h[:, :])
        npa = npa.reshape(1, npa.shape[0] * npa.shape[1])
        hist = npy.histogram(npa, self._nbins, normed=True, range=(0, 255))
        return hist[0].tolist()

    def get_field_names(self):
        """
        This method gives the names of each field in the feature vector in the
        order in which they are returned. For example, 'xpos' or 'width'
        """
        ret = []
        for i in range(self._nbins):
            name = "Hue" + str(i)
            ret.append(name)
        return ret

    def get_num_fields(self):
        """
        This method returns the total number of fields in the feature vector.
        """
        return self._nbins
