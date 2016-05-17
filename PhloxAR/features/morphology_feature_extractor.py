# -*- coding: utf-8 -*-
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PhloxAR.base import *
from PhloxAR.features.blob_maker import BlobMaker
from PhloxAR.features.feature_extractor_base import FeatureExtractorBase


__all__ = [
    'MorphologyFeatureExtractor'
]


class MorphologyFeatureExtractor(FeatureExtractorBase):
    """
    This feature extractor collects some basic morphology information about a
    given image. It is assumed that the object to be recognized is the largest
    object in the image. The user must provide a segmented white on black blob
    image. This operation then straightens the image and collects the data.
    """
    _nbins = 9
    _blob_maker = None
    threshold_operation = None

    def __init__(self, thresh_operation=None):
        """
        The threshold operation is a function of the form
        binaryimg = threshold(img)
        the simplest example would be:
        def binarize_wrap(img):
        """
        self._nbins = 9
        self._blob_maker = BlobMaker()
        self.threshold_operation = thresh_operation

    def setThresholdOperation(self, thresh_op):
        """
        The threshold operation is a function of the form
        binaryimg = threshold(img)

        Example:
        >>> def binarize_wrap(img):
        >>>    return img.binarize()
        """
        self._threshold_operation = thresh_op

    def extract(self, img):
        """
        This method takes in a image and returns some basic morphology
        characteristics about the largest blob in the image. The
        if a color image is provided the threshold operation is applied.
        """
        ret = None
        if self._threshold_operation is not None:
            bitwise = self._threshold_operation(img)
        else:
            bitwise = img.binarize()

        if self._blob_maker is None:
            self._blob_maker = BlobMaker()

        fs = self._blob_maker.extract_from_binary(bitwise, img)
        if fs is not None and len(fs) > 0:
            fs = fs.sort_area()
            ret = []
            ret.append(fs[0].area / fs[0].perimeter)
            ret.append(fs[0].aspect_ratio)
            ret.append(fs[0].hu[0])
            ret.append(fs[0].hu[1])
            ret.append(fs[0].hu[2])
            ret.append(fs[0].hu[3])
            ret.append(fs[0].hu[4])
            ret.append(fs[0].hu[5])
            ret.append(fs[0].hu[6])
        return ret

    def get_field_names(self):
        """
        This method gives the names of each field in the feature vector in the
        order in which they are returned. For example, 'xpos' or 'width'
        """
        ret = []
        ret.append('area over perim')
        ret.append('AR')
        ret.append('Hu0')
        ret.append('Hu1')
        ret.append('Hu2')
        ret.append('Hu3')
        ret.append('Hu4')
        ret.append('Hu5')
        ret.append('Hu6')
        return ret

    def get_num_fields(self):
        """
        This method returns the total number of fields in the feature vector.
        """
        return self._nbins

    def __getstate__(self):
        att = self.__dict__.copy()
        self._blob_maker = None
        del att['_blob_maker']
        return att

    def __setstate__(self, state):
        self.__dict__ = state
        self._blob_maker = BlobMaker()
