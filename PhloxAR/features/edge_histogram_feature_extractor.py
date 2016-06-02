# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
from __future__ import division, print_function

from PhloxAR.features.feature_extractor_base import FeatureExtractorBase

__all__ = [
    'EdgeHistogramFeatureExtractor'
]


class EdgeHistogramFeatureExtractor(FeatureExtractorBase):
    """
    Create a 1D edge length histogram and 1D edge angle histogram.
    This method takes in an image, applies an edge detector, and calculates
    the length and direction of lines in the image.
    bins = the number of bins
    """
    _nbins = 10

    def __init__(self, bins=10):
        self._nbins = bins

    def extract(self, img):
        """
        Extract the line orientation and and length histogram.
        """
        # I am not sure this is the best normalization constant.
        ret = []
        p = max(img.width, img.height) / 2
        min_line = 0.01 * p
        gap = 0.1 * p
        fs = img.findLines(threshold=10, minlinelength=min_line, maxlinegap=gap)
        ls = fs.length() / p  # normalize to image length
        angs = fs.angle()
        lhist = npy.histogram(ls, self._nbins, normed=True, range=(0, 1))
        ahist = npy.histogram(angs, self._nbins, normed=True, range=(-180, 180))
        ret.extend(lhist[0].tolist())
        ret.extend(ahist[0].tolist())
        return ret

    def get_field_names(self):
        """
        Return the names of all of the length and angle fields.

        This method gives the names of each field in the features vector in the
        order in which they are returned. For example, 'xpos' or 'width'
        """
        ret = []
        for i in range(self._nbins):
            name = "Length" + str(i)
            ret.append(name)
        for i in range(self._nbins):
            name = "Angle" + str(i)
            ret.append(name)

        return ret

    def get_num_fields(self):
        """
        This method returns the total number of fields in the features vector.
        """
        return self._nbins * 2
