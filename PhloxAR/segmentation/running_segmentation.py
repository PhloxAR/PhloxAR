# -*- coding: utf-8 -*-
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PhloxAR.base import *
from PhloxAR.features.blob_maker import BlobMaker
from PhloxAR.image import Image
from PhloxAR.segmentation.segmentation_base import SegmentationBase


__all__ = [
    'RunningSegmentation'
]


class RunningSegmentation(SegmentationBase):
    """
    RunningSegmentation performs segmentation using a running background model.
    This model uses an accumulator which performs a running average of previous
    frames where: accumulator = ((1-alpha)input_image)+((alpha)accumulator)
    """

    _error = False
    _alpha = 0.1
    _thresh = 10
    _model_img = None
    _diff_img = None
    _curr_img = None
    _color_img = None
    _blob_maker = None
    _gray = True
    _ready = False

    def __init__(self, alpha=0.7, thresh=(20, 20, 20)):
        """
        Create an running background difference.
        alpha - the update weighting where:
        accumulator = ((1-alpha)input_image)+((alpha)accumulator)
        threshold - the foreground background difference threshold.
        """
        self._error = False
        self._ready = False
        self._alpha = alpha
        self._thresh = thresh
        self._model_img = None
        self._diff_img = None
        self._color_img = None
        self._blob_maker = BlobMaker()

    def add_image(self, img):
        """
        Add a single image to the segmentation algorithm
        """
        if img is None:
            return

        self._color_img = img

        if self._model_img is None:
            self._model_img = Image(
                cv.CreateImage((img.width, img.height), cv.IPL_DEPTH_32F, 3))
            self._diff_img = Image(
                cv.CreateImage((img.width, img.height), cv.IPL_DEPTH_32F, 3))
        else:
            # do the difference
            cv.AbsDiff(self._model_img.bitmap, img.float_matrix,
                       self._diff_img.bitmap)
            # update the model
            cv.RunningAvg(img.float_matrix, self._model_img.bitmap,
                          self._alpha)
            self._ready = True
        return

    def is_ready(self):
        """
        Returns true if the camera has a segmented image ready.
        """
        return self._ready

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
        self._model_img = None
        self._diff_img = None

    @property
    def raw_image(self):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        return self._float2int(self._diff_img)

    @property
    def segmented_image(self, white_fg=True):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        img = self._float2int(self._diff_img)
        if white_fg:
            ret = img.binarize(thresh=self._thresh)
        else:
            ret = img.binarize(thresh=self._thresh).invert()
        return ret

    @property
    def segmented_blobs(self):
        """
        return the segmented blobs from the fg/bg image
        """
        ret = []
        if self._color_img is not None and self._diff_img is not None:
            eight_bit = self._float2int(self._diff_img)
            ret = self._blob_maker.extract_from_binary(
                eight_bit.binarize(thresh=self._thresh), self._color_img)

        return ret

    def _float2int(self, img):
        """
        convert a 32bit floating point cv array to an int array
        """
        temp = cv.CreateImage((img.width, img.height), cv.IPL_DEPTH_8U, 3)
        cv.Convert(img.bitmap, temp)

        return Image(temp)

    def __getstate__(self):
        state = self.__dict__.copy()
        self._blob_maker = None
        self._model_img = None
        self._diff_img = None
        del state['_blob_maker']
        del state['_model_img']
        del state['_diff_img']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._blob_maker = BlobMaker()
