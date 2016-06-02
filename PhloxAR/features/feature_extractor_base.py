# -*- coding: utf-8 -*-
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PhloxAR.base import *
import six


__all__ = [
    'FeatureExtractorBase'
]


@six.add_metaclass(abc.ABCMeta)
class FeatureExtractorBase(object):
    """
    The featureExtractorBase class is a way of abstracting the process of collecting
    descriptive features within an image. A features is some description of the image
    like the mean _color, or the width of a center image, or a histogram of edge
    lengths. This features vectors can then be composed together and used within
    a machine learning algorithm to descriminate between different classes of objects.
    """

    @classmethod
    def load(cls, fname):
        """
        load segmentation settings to file.
        """
        return pickle.load(file(fname))

    def save(self, fname):
        """
        Save segmentation settings to file.
        """
        output = open(fname, 'wb')
        pickle.dump(self, output, 2)  # use two otherwise it borks the system
        output.close()

    @abc.abstractmethod
    def extract(self, img):
        """
        Given an image extract the features vector. The output should be a list
        object of all of the features. These features can be of any interal type
        (string, float, integer) but must contain no sub lists.
        """

    @abc.abstractmethod
    def get_field_names(self):
        """
        This method gives the names of each field in the features vector in the
        order in which they are returned. For example, 'xpos' or 'width'
        """

    @abc.abstractmethod
    def get_num_fields(self):
        """
        This method returns the total number of fields in the features vector.
        """
