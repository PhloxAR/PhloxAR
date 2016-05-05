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




