# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
from __future__ import division, print_function

from PhloxAR.core.color import ColorModel
from PhloxAR.core.image import Image
from PhloxAR.features.blob_maker import BlobMaker
from PhloxAR.segmentation.segmentation_base import SegmentationBase

__all__ = [
    'ColorSegmentation'
]


class ColorSegmentation(SegmentationBase):
    """
    Perform color segmentation based on a color model or color provided.
    This class uses ColorModel.py to create a color model.
    """
    _color_model = []
    _error = False
    _cur_img = []
    _truth_img = []
    _blob_maker = []

    def __init__(self):
        self._color_model = ColorModel()
        self._error = False
        self._cur_img = Image()
        self._truth_img = Image()
        self._blob_maker = BlobMaker()

    def add_image(self, img):
        """
        Add a single image to the segmentation algorithm
        """
        self._truth_img = img
        self._cur_img = self._color_model.threshold(img)
        return

    def is_ready(self):
        """
        Returns true if the camera has a segmented image ready.
        """
        return True

    def is_error(self):
        """
        Returns true if the segmentation system has detected an error.
        Eventually we'll construct a syntax of errors so this becomes
        more expressive
        """
        return self._error  # need to make a generic error checker

    def reset_error(self):
        """
        Clear the previous error.
        """
        self._error = False
        return

    def reset(self):
        """
        Perform a reset of the segmentation systems underlying data.
        """
        self._color_model.reset()

    @property
    def raw_image(self):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        return self._cur_img

    @property
    def segmented_image(self, white_fg=True):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        return self._cur_img

    @property
    def segmented_blobs(self):
        """
        return the segmented blobs from the fg/bg image
        """
        return self._blob_maker.extract_from_binary(self._cur_img,
                                                    self._truth_img)

    # The following are class specific methods

    def add_model(self, data):
        self._color_model.add(data)

    def sub_model(self, data):
        self._color_model.remove(data)

    def __getstate__(self):
        state = self.__dict__.copy()
        self._blob_maker = None
        del state['_blob_maker']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._blob_maker = BlobMaker()
