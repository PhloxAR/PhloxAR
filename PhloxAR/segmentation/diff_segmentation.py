# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
from __future__ import division, print_function

from PhloxAR.core.image import Image
from PhloxAR.features.blob_maker import BlobMaker
from PhloxAR.segmentation.segmentation_base import SegmentationBase
from ..base import cv2

__all__ = [
    'DiffSegmentation'
]


class DiffSegmentation(SegmentationBase):
    """
    This method will do image segmentation by looking at the difference between
    two frames.
    grayOnly - use only gray images.
    threshold - The value at which we consider the _color difference to
    be significant enough to be foreground imagery.
    The general usage is
    >>> segmentor = DiffSegmentation()
    >>> cam = Camera()
    >>> while 1:
    >>>    segmentor.add_image(cam.getImage())
    >>>    if segmentor.is_ready():
    >>>        img = segmentor.segmented_image
    """
    _error = False
    _last_img = None
    _curr_img = None
    _diff_img = None
    _color_img = None
    _gray_mode = True
    _threshold = 10
    _blob_maker = None

    def __init__(self, gray=False, threshold=(10, 10, 10)):
        self._gray_mode = gray
        self._threshold = threshold
        self._error = False
        self._curr_img = None
        self._last_img = None
        self._diff_img = None
        self._color_img = None
        self._blob_maker = BlobMaker()

    def add_image(self, img):
        """
        Add a single image to the segmentation algorithm
        """
        if img is None:
            return
        if self._last_img is None:
            if self._gray_mode:
                self._last_img = img.to_gray()
                self._diff_img = Image(self._last_img.zeros(1))
                self._curr_img = None
            else:
                self._last_img = img
                self._diff_img = Image(self._last_img.zeros(3))
                self._curr_img = None
        else:
            if self._curr_img is not None:  # catch the first step
                self._last_img = self._curr_img

            if self._gray_mode:
                self._color_img = img
                self._curr_img = img.to_gray()
            else:
                self._color_img = img
                self._curr_img = img

            cv2.absdiff(self._curr_img.narray, self._last_img.narray,
                        self._diff_img.narray)

        return

    def is_ready(self):
        """
        Returns true if the camera has a segmented image ready.
        """
        if self._diff_img is None:
            return False
        else:
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

    def reset(self):
        """
        Perform a reset of the segmentation systems underlying data.
        """
        self._curr_img = None
        self._last_img = None
        self._diff_img = None

    @property
    def raw_image(self):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        return self._diff_img

    @property
    def segmented_image(self, white_fg=True):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        if white_fg:
            ret = self._diff_img.binarize(thresh=self._threshold)
        else:
            ret = self._diff_img.binarize(thresh=self._threshold).invert()
        return ret

    @property
    def segmented_blobs(self):
        """
        return the segmented blobs from the fg/bg image
        """
        ret = []
        if self._color_img is not None and self._diff_img is not None:
            ret = self._blob_maker.extract_from_binary(
                self._diff_img.binarize(thresh=self._threshold), self._color_img
            )
        return ret

    def __getstate__(self):
        state = self.__dict__.copy()
        self._blob_maker = None
        del state['_blob_maker']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._blob_maker = BlobMaker()
