# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PhloxAR.base import *
from PhloxAR.image import Image, ImageSet, ColorSpace
from PhloxAR.display import Display
from PhloxAR.color import Color
from collections import deque
import time
import ctypes
import subprocess
import cv2
import numpy as npy
import traceback
import sys


# globals
_cameras = []
_camera_polling_thread = ''
_index = []


class FrameSource(object):
    """
    An abstract Camera-type class, for handling multiple types of video
    input. Any sources of image inherit from it.
    """
    _cali_mat = ''  # intrinsic calibration matrix
    _dist_coeff = ''  # distortion matrix
    _thread_cap_time = ''  # the time the last picture was taken
    capture_time = ''  # timestamp of the last aquired image

    def __init__(self):
        return

    def get_property(self, p):
        return None

    def get_all_properties(self):
        return {}

    def get_image(self):
        return None

    def calibrate(self, image_list, grid_size=0.03, dimensions=(8, 5)):
        pass

    def get_camera_matrix(self):
        """
        Return a cvMat of the camera's intrinsic matrix.
        """
        return self._cali_mat

    def undistort(self, image_or_2darray):
        pass

    def get_image_undisort(self):
        return self.undistort(self.get_image())

    def save_calibration(self, filename):
        pass

    def load_calibration(self, filename):
        pass

    def live(self):
        pass



class Camera(FrameSource):
    pass


class FrameBufferThread(threading.Thread):
    pass


class VirtualCamera(FrameSource):
    pass


class JpegStreamReader(threading.Thread):
    pass


class JpegStreamCamera(FrameSource):
    pass


class Scanner(FrameSource):
    pass


class DigitalCamera(FrameSource):
    pass


class ScreenCamera(FrameSource):
    pass


class StereoImage(object):
    pass


class StereoCamera(object):
    pass


class AVTCameraThread(threading.Thread):
    pass


class AVTCamera(FrameSource):
    pass


class AVTCameraInfo(ctypes.Structure):
    pass


class AVTFrame(ctypes.Structure):
    pass


class GigECamera(Camera):
    pass


class VimbaCamera(FrameSource):
    pass


class VimbaCameraThread(threading.Thread):
    pass




