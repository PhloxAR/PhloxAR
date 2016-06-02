# -*- coding: utf-8 -*-
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PhloxAR.base import *
from PhloxAR.features.haar_like_feature import *
from PhloxAR.features.feature_extractor_base import FeatureExtractorBase


__all__ = [
    'HaarLikeFeatureExtractor'
]


class HaarLikeFeatureExtractor(FeatureExtractorBase):
    """
    This is used generate Haar like features from an image.  These
    Haar like features are used by a the classifiers of machine learning
    to help identify objects or things in the picture by their features,
    or in this case haar features.
    For a more in-depth review of Haar Like features see:
    http://en.wikipedia.org/wiki/Haar-like_features
    """

    _feature_set = None
    _do45 = True

    def __init__(self, fname=None, do45=True):
        """
        fname - The features file name
        do45 - if this is true we use the regular integral image plus the
        45 degree integral image
        """
        # we define the black (positive) and white (negative) regions of an image
        # to get our haar wavelet
        self._do45 = True
        self._feature_set = None
        if fname is not None:
            self.read_wavelets(fname)

    def read_wavelets(self, fname, nfeats=-1):
        """
        fname = file name
        nfeats = number of features to load from file -1 -> All features
        """
        # We borrowed the wavelet file from  Chesnokov Yuriy
        # He has a great windows tutorial here:
        # http://www.codeproject.com/KB/audio-video/haar_detection.aspx
        # SimpleCV Took a vote and we think he is an all around swell guy!
        # nfeats = number of features to load
        # -1 loads all
        # otherwise loads min(nfeats,features in file)
        self._feature_set = []
        f = open(fname, 'r')
        # line = f.readline()
        # count = int(line)
        temp = f.read()
        f.close()
        data = temp.split()
        count = int(data.pop(0))
        self.mFeatureset = []
        if (nfeats > -1):
            count = min(count, nfeats)
        while len(data) > 0:
            name = data.pop(0)
            nRegions = int(data.pop(0))
            region = []
            for i in range(nRegions):
                region.append(tuple(map(float, data[0:5])))
                data = data[5:]

            feat = HaarLikeFeature(name, region)
            self._feature_set.append(feat)
        return None

    def save_wavelets(self, fname):
        """
        Save wavelets to file
        """
        f = open(fname, 'w')
        f.write(str(len(self._feature_set)) + '\n\n')
        for i in range(len(self._feature_set)):
            self._feature_set[i].writeToFile(f)
        f.close()
        return None

    def extract(self, img):
        """
        This extractor takes in an image, creates the integral image, applies
        the Haar cascades, and returns the result as a features vector.
        """
        regular = img.integralImage()
        retVal = []

        for i in range(len(self._feature_set)):
            retVal.append(self._feature_set[i].apply(regular))
        if self._do45:
            slant = img.integralImage(tilted=True)
            for i in range(len(self._feature_set)):
                retVal.append(self._feature_set[i].apply(regular))
        return retVal

    def get_field_names(self):
        """
        This method gives the names of each field in the features vector in the
        order in which they are returned. For example, 'xpos' or 'width'
        """
        retVal = []
        for i in range(len(self._feature_set)):
            retVal.append(self._feature_set[i].mName)
        if self._do45:
            for i in range(len(self._feature_set)):
                name = "Angle_" + self._feature_set[i].mName
                retVal.append(name)
        return retVal

    def get_num_fields(self):
        """
        This method returns the total number of fields in the features vector.
        """
        mult = 1
        if self._do45:
            mult = 2
        return mult * len(self._feature_set)
