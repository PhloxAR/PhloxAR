# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PhloxAR.base import *
from PhloxAR.features.feature import Feature, FeatureSet
from PhloxAR.color import Color
from PhloxAR.image import Image
import six


@six.add_metaclass(abc.ABCMeta)
class SegmentationBase(object):
    @classmethod
    def load(cls, filename):
        """
        Load segmentation settings to file.
        """
        return pickle.load(file(filename))

    def save(self, filename):
        """
        Save segmentation settings to file.
        """
        output = open(filename, 'wb')
        pickle.dump(self, output, 2)
        output.close()

    @abc.abstractmethod
    def add_image(self, img):
        """
        Add a single image to the segmentation algorithm
        """
        return

    @abc.abstractmethod
    def is_ready(self):
        """
        Returns true if the camera has a segmented image ready.
        """
        return False

    @abc.abstractmethod
    def is_error(self):
        """
        Returns true if the segmentation system has detected an error.
        Eventually we'll construct a syntax of errors so this becomes
        more expressive
        """
        return False

    @abc.abstractmethod
    def reset_error(self):
        """
        Clear the previous error.
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Perform a reset of the segmentation systems underlying data.
        """
        pass

    @abc.abstractproperty
    def raw_image(self):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        pass

    @abc.abstractproperty
    def segmented_image(self, white_fg=True):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        pass

    @abc.abstractproperty
    def segmented_blobs(self):
        """
        return the segmented blobs from the fg/bg image
        """
        pass
