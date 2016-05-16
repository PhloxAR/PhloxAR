# -*- coding: utf-8 -*-
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PhloxAR.base import *
from PhloxAR.features.feature import Feature, FeatureSet
from PhloxAR.features.blob_maker import BlobMaker
from PhloxAR.image import Image
from PhloxAR.segmentation.segmentation_base import SegmentationBase


class MOGSegmentation(SegmentationBase):
    """
    Background subtraction using mixture of gaussians.
    For each pixel store a set of gaussian distributions and try to fit new pixels
    into those distributions. One of the distributions will represent the background.

    history - length of the pixel history to be stored
    nMixtures - number of gaussian distributions to be stored per pixel
    backgroundRatio - chance of a pixel being included into the background model
    noiseSigma - noise amount
    learning rate - higher learning rate means the system will adapt faster to new backgrounds
    """

    _error = False
    _diff_img = None
    _color_img = None
    _ready = False

    # OpenCV default parameters
    history = 200
    nMixtures = 5
    backgroundRatio = 0.7
    noiseSigma = 15
    learningRate = 0.7
    bsMOG = None

    def __init__(self, history=200, nMixtures=5, backgroundRatio=0.7,
                 noiseSigma=15, learningRate=0.7):

        try:
            import cv2
        except ImportError:
            raise ImportError(
                "Cannot load OpenCV library which is required by SimpleCV")
            return
        if not hasattr(cv2, 'BackgroundSubtractorMOG'):
            raise ImportError("A newer version of OpenCV is needed")
            return

        self._error = False
        self._ready = False
        self._diff_img = None
        self._color_img = None
        self._blob_maker = BlobMaker()

        self.history = history
        self.nMixtures = nMixtures
        self.backgroundRatio = backgroundRatio
        self.noiseSigma = noiseSigma
        self.learningRate = learningRate

        self._bsmog = cv2.BackgroundSubtractorMOG(history, nMixtures,
                                                  backgroundRatio, noiseSigma)

    def add_image(self, img):
        """
        Add a single image to the segmentation algorithm
        """
        if img is None:
            return

        self._color_img = img
        self._diff_img = Image(
            self._bsmog.apply(img.cvnarray, None, self.learningRate),
            cv2image=True
        )
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
        return self._diff_img

    @property
    def segmented_image(self, white_fg=True):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        return self._diff_img

    @property
    def segmented_blobs(self):
        """
        return the segmented blobs from the fg/bg image
        """
        ret = []
        if self._color_img is not None and self._diff_img is not None:
            ret = self._blob_maker.extract_from_binary(self._diff_img,
                                                       self._color_img)
        return ret

    def __getstate__(self):
        state = self.__dict__.copy()
        self._blob_maker = None
        self._diff_img = None
        del state['_blob_maker']
        del state['_diff_img']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._blob_maker = BlobMaker()
