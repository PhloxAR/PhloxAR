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
import six


# globals
_gcameras = []
_gcamera_polling_thread = ''
_gindex = []


@six.add_metaclass(abc.ABCMeta)
class FrameSource(object):
    """
    An abstract Camera-type class, for handling multiple types of video
    input. Any sources of image inherit from it.
    """
    _calib_mat = ''  # intrinsic calibration matrix
    _dist_coeff = ''  # distortion matrix
    _thread_cap_time = ''  # the time the last picture was taken
    capture_time = ''  # timestamp of the last acquired image

    @abc.abstractmethod
    def __init__(self):
        return

    @abc.abstractmethod
    def get_property(self, p):
        return None

    @abc.abstractmethod
    def get_all_properties(self):
        return {}

    @abc.abstractmethod
    def get_image(self):
        return None

    @abc.abstractmethod
    def calibrate(self, image_list, grid_size=0.03, dims=(8, 5)):
        """
        Camera calibration will help remove distortion and fish eye effects
        It is agnostic of the imagery source, and can be used with any camera

        The easiest way to run calibration is to run the calibrate.py file
        under the tools directory.

        :param image_list: a list of images of color calibration images
        :param grid_size: the actual grid size of the calibration grid,
                           the unit used will be the calibration unit
                           value (i.e. if in doubt use meters, or U.S. standard)
        :param dims: the count of the 'interior' corners in the calibration
                      grid. So far a grid where there are 4x4 black squares
                      has seven interior corners.
        :return: camera's intrinsic matrix.
        """
        warn_thresh = 1
        n_boards = 0  # number of boards
        board_w = int(dims[1])  # number of horizontal corners
        board_h = int(dims[0])  # number of vertical corners
        n_boards = int(len(image_list))
        board_num= board_w * board_h  # number of total corners
        board_size = (board_w, board_h)  # size of board

        if n_boards < warn_thresh:
            logger.warning('FrameSource.calibrate: We suggest suing 20 or'
                           'more images to perform camera calibration.')

        # creation of memory storages
        image_points = cv.CreateMat(n_boards * board_num, 2, cv.CV_32FC1)
        object_points = cv.CreateMat(n_boards * board_num, 3, cv.CV_32FC1)
        point_counts = cv.CreateMat(n_boards, 1, cv.CV_32SC1)
        intrinsic_mat = cv.CreateMat(3, 3, cv.CV_32FC1)
        dist_coeff = cv.CreateMat(5, 1, cv.CV_32FC1)

        # capture frames of specified properties and modification
        # of matrix values
        i = 0
        z = 0  # to print number of frames
        successes = 0
        img_idx = 0

        # capturing required number of views
        while successes < n_boards:
            found = 0
            img = image_list[img_idx]
            found, corners = cv.FindChessboardCorners(
                img.gray_matrix, board_size,
                cv.CV_CALIB_CB_ADAPTIVE_THRESH | cv.CV_CALIB_CB_FILTER_QUADS
            )

            corners = cv.FindCornerSubPix(
                    img.gray_matrix, corners,
                    (11, 11), (-1, -1),
                    (cv.CV_TERMCRIT_EPS + cv.CV_TERMCRIT_ITER, 30, 0.1)
            )

            # if got a good image, draw chess board
            if found == 1:
                corner_count = len(corners)
                z += 1

            # if got a good image, add to matrix
                if len(corners) == board_num:
                    step = successes * board_num
                    k = step
                    for j in range(board_num):
                        cv.Set2D(image_points, k, 0, corners[j][0])
                        cv.Set2D(image_points, k, 1, corners[j][1])
                        cv.Set2D(object_points, k, 0,
                                 grid_size * (float(j) / float(board_w)))
                        cv.Set2D(object_points, k, 1,
                                 grid_size * (float(j) % float(board_w)))
                        cv.Set2D(object_points, k, 2, 0.0)
                    cv.Set2D(point_counts, successes, 0, board_num)
                    successes += 1

        # now assigning new matrices according to view_count
        if successes < warn_thresh:
            logger.warning('FrameSource.calibrate: You have {} good '
                           'images for calibration, but we recommend '
                           'at least {}'.format(successes, warn_thresh))

        object_points2 = cv.CreateMat(successes * board_num, 3, cv.CV_32FC1)
        image_points2 = cv.CreateMat(successes * board_num, 2, cv.CV_32FC1)
        point_counts2 = cv.CreateMat(successes, 1, cv.CV_32FC1)

        for i in range(successes * board_num):
            cv.Set2D(image_points2, i, 0, cv.Get2D(image_points, i, 0))
            cv.Set2D(image_points2, i, 1, cv.Get2D(image_points, i, 1))
            cv.Set2D(object_points2, i, 0, cv.Get2D(object_points, i, 0))
            cv.Set2D(object_points2, i, 1, cv.Get2D(object_points, i, 1))
            cv.Set2D(object_points2, i, 2, cv.Get2D(object_points, i, 2))

        for i in range(successes):
            cv.Set2D(point_counts2, i, 0, cv.Get2D(point_counts, i, 0))

        cv.Set2D(intrinsic_mat, 0, 0, 1.0)
        cv.Set2D(intrinsic_mat, 1, 1, 1.0)
        rcv = cv.CreateMat(n_boards, 3, cv.CV_64FC1)
        tcv = cv.CreateMat(n_boards, 3, cv.CV_64FC1)
        # camera calibration
        cv.CalibrateCamera2(object_points2, image_points2, point_counts2,
                            (img.width, img.height), intrinsic_mat,
                            dist_coeff, rcv, tcv, 0)

        self._calib_mat = intrinsic_mat
        self._dist_coeff = dist_coeff
        return intrinsic_mat

    @abc.abstractproperty
    def camera_matrix(self):
        """
        Return a cvMat of the camera's intrinsic matrix.
        """
        return self._calib_mat

    @abc.abstractmethod
    def undistort(self, img):
        """
        If given an image, apply the undistortion given by the camera's matrix
        and return the result.
        If given a 1xN 2D cvmat or a 2xN numpy array, it will un-distort points
        of measurement and return them in the original coordinate system.
        :param img: an image or and ndarray
        :return: The undistored image or the undisotreted points.

        :Example:
        >>> img = cam.get_image()
        >>> result = cam.undistort(img)
        """
        if not (isinstance(self._calib_mat, cv.cvmat) and
                isinstance(self._dist_coeff, cv.cvmat)):
            logger.warning('FrameSource.undistort: This operation requires '
                           'calibration, please load the calibration matrix')
            return None

        if isinstance(img, InstanceType) and isinstance(img, Image):
            inimg = img
            ret = inimg.zeros()
            cv.Undistort2(inimg.bitmap, ret, self._calib_mat, self._dist_coeff)
            return Image(ret)
        else:
            mat = None
            if isinstance(img, cv.cvmat):
                mat = img
            else:
                arr = cv.fromarray(npy.array(img))
                mat = cv.CreateMat(cv.GetSize(arr)[1], 1, cv.CV_64FC2)
                cv.Merge(arr[:, 0], arr[:, 1], None, None, mat)

            upoints = cv.CreateMat(cv.GetSize(mat)[1], 1, cv.CV_64FC2)
            cv.UndistortPoints(mat, upoints, self._calib_mat, self._dist_coeff)

            return (npy.array(upoints[:, 0]) * [
                self.camera_matrix[0, 0],
                self.camera_matrix[1, 1] + self.camera_matrix[0, 2],
                self.camera_matrix[1, 2]
            ])[:, 0]

    @abc.abstractmethod
    def get_image_undisort(self):
        """
        Using the overridden get_image method, we retrieve the image and apply
        the undistortion operation
        :return: latest image from the camera after applying undistortion.
        >>> cam = Camera()
        >>> cam.loadCalibration("mycam.xml")
        >>> while True:
        >>>    img = cam.get_image_undisort()
        >>>    img.show()
        """
        return self.undistort(self.get_image())

    @abc.abstractmethod
    def save_calibration(self, filename):
        """
        Save the calibration matrices to file.

        :param filename: file name, without extension
        :return: True, if the file was saved, False otherwise
        """
        ret1 = ret2 = False
        if not isinstance(self._calib_mat, cv.cvmat):
            logger.warning("FrameSource.save_calibration: No calibration matrix"
                           "present, can't save")
        else:
            intr_file = filename + 'Intrinsic.xml'
            cv.Save(intr_file, self._calib_mat)
            ret1 = True

        if not isinstance(self._dist_coeff, cv.cvmat):
            logger.warning("FrameSource.save_calibration: No calibration matrix"
                           "present, can't save")
        else:
            dist_file = filename + 'Distortion.xml'
            cv.Save(dist_file, self._dist_coeff)
            ret2 = True

        return ret1 and ret2

    @abc.abstractmethod
    def load_calibration(self, filename):
        """
        Load a calibration matrix from file.
        The filename should be the stem of the calibration files names.
        e.g. if the calibration files are MyWebcamIntrinsic.xml and
        MyWebcamDistortion.xml then load the calibration file 'MyWebCam'

        :param filename: without extension, which saves the calibration data
        :return: Bool. True - file was loaded, False otherwise.
        """
        intr_file = filename + 'Intrinsic.xml'
        self._calib_mat = cv.Load(intr_file)
        dist_file = filename + 'Distortion.xml'
        self._dist_coeff = cv.Load(dist_file)

        if (isinstance(self._dist_coeff, cv.cvmat) and
                isinstance(self._calib_mat, cv.cvmat)):
            return True

        return False

    @abc.abstractmethod
    def live(self):
        """
        Shows a live view of the camera.

        :Example:
        >>> cam = Camera()
        >>> cam.live()
        Left click will show mouse coordinates and color.
        Right click will kill the live image.
        """
        start_time = time.time()

        img = self.get_image()
        dsp = Display(img.size())
        img.save(dsp)
        col = Color.RED

        while not dsp.is_done():
            img = self.get_image()
            elapsed_time = time.time() - start_time

            if dsp.mouse_l:
                txt1 = 'Coord: ({}, {})'.format(dsp.mouse_x, dsp.mouse_y)
                img.dl().text(txt1, (10, img.height / 2), color=col)
                txt2 = 'Color: {}'.format(img.get_pixel(dsp.mouse_x, dsp.mouse_y))
                img.dl().text(txt2, (10, img.height / 2 + 10), color=col)
                print(txt1 + txt2)

            if 0 < elapsed_time < 5:
                img.dl().text('In live mode', (10, 10), color=col)
                img.dl().text('Left click will show mouse coordinates and color',
                              (10, 20), color=col)
                img.dl().text('Right click will kill the live image', (10, 30),
                              color=col)

            img.save(dsp)
            if dsp.mouse_r:
                print("Closing window!")
                dsp.done = True

        sdl2.quit()


class Camera(FrameSource):
    _cv2_capture = None  # cvCapture object
    _thread = None
    _sdl2_cam = False
    _sdl2_buf = None

    prop_map = {
        "width": cv.CV_CAP_PROP_FRAME_WIDTH,
        "height": cv.CV_CAP_PROP_FRAME_HEIGHT,
        "brightness": cv.CV_CAP_PROP_BRIGHTNESS,
        "contrast": cv.CV_CAP_PROP_CONTRAST,
        "saturation": cv.CV_CAP_PROP_SATURATION,
        "hue": cv.CV_CAP_PROP_HUE,
        "gain": cv.CV_CAP_PROP_GAIN,
        "exposure": cv.CV_CAP_PROP_EXPOSURE
    }

    def __init__(self, cam_idx=-1, prop_set={}, threaded=True, calib_file=''):
        """
        In the camera constructor, cam_idx indicates which camera to connect to
        and prop_set is a dictionary which can be used to set any camera
        attributes, supported props are currently:
        height, width, brightness, contrast, saturation, hue, gain, and exposure

        You can also specify whether you want the FrameBufferThread to
        continuously debuffer the camera.  If you specify True, the camera
        is essentially 'on' at all times.  If you specify off, you will have
        to manage camera buffers.

        :param cam_idx: the index of the camera, these go from 0 upward,
                         and are system specific.
        :param prop_set: the property set for the camera (i.e. a dict of
                          camera properties).
        :Note:
        For most web cameras only the width and height properties are
        supported. Support for all of the other parameters varies by
        camera and operating system.

        :param threaded: if True we constantly debuffer the camera, otherwise
                          the user must do this manually.
        :param calib_file: calibration file to load.
        """
        global _gcameras
        global _gcamera_polling_thread
        global _gindex

        self._index = None
        self._threaded = False
        self._cv2_capture = None

        if platform.system() == 'Linux':
            if -1 in _gindex and cam_idx != -1 and cam_idx not in _gindex:
                process = subprocess.Popen(['lsof /dev/video' + str(cam_idx)],
                                           shell=True, stdout=subprocess.PIPE)
                data = process.communicate()
                if data[0]:
                    cam_idx = -1
            else:
                process = subprocess.Popen(['lsof /dev/video*'], shell=True,
                                           stdout=subprocess.PIPE)
                data = process.communicate()
                if data[0]:
                    cam_idx = int(data[0].split('\n')[1].split()[-1][-1])

        for cam in _gcameras:
            if cam_idx == cam.index:
                self._threaded = cam.threaded
                self._cv2_capture = cam.cv2_capture
                self._index = cam.index
                _gcameras.append(self)
                return

        # to support XIMEA cameras
        if isinstance(cam_idx, str):
            if cam_idx.lower() == 'ximea':
                cam_idx = 1100
                _gindex.append(cam_idx)

        self._cv2_capture = cv.CaptureFromCAM(cam_idx)
        self._index = cam_idx

        if 'delay' in prop_set:
            time.sleep(prop_set['delay'])

        if (platform.system() == 'Linux' and
                ('height' in prop_set or
                         cv.GrabFrame(self._cv2_capture) == False)):
            import pygame.camera as sdl2_cam
            sdl2_cam.init()
            threaded = True  # pygame must be threaded

            if cam_idx == -1:
                cam_idx = 0
                self._index = cam_idx
                _gindex.append(cam_idx)
                print(_gindex)

            if 'height' in prop_set and 'width' in prop_set:
                self._cv2_capture = sdl2_cam.Camera('/dev/video' + str(cam_idx),
                                                    prop_set['width'],
                                                    prop_set['height'])
            else:
                self._cv2_capture = sdl2_cam.Camera('/dev/video' + str(cam_idx))

            try:
                self._cv2_capture.start()
            except Exception as e:
                msg = "Caught exception: {}".format(e)
                logger.warning(msg)
                logger.warning('PhloxAR cannot find camera on your computer!')
                return
            time.sleep(0)
            self._sdl2_buf = self._cv2_capture.get_image()
            self._sdl2_cam = True
        else:
            _gindex.append(cam_idx)
            self._threaded = False

            if platform.system() == 'Windows':
                threaded = False

            if not self._cv2_capture:
                return

            for p in prop_set.keys():
                if p in self.prop_map:
                    cv.SetCaptureProperty(self._cv2_capture, self.prop_map[p],
                                          prop_set[p])

        if threaded:
            self._threaded = True
            _gcameras.append(self)
            if not _gcamera_polling_thread:
                _gcamera_polling_thread = FrameBufferThread()
                _gcamera_polling_thread.daemon = True
                _gcamera_polling_thread.start()
                time.sleep(0)

        if calib_file:
            self.load_calibration(calib_file)

        super(Camera, self).__init__()

    def get_property(self, p):
        """
        Retrieve the value of a given property, wrapper for
        cv.GetCaptureProperty

        :param p: the property to retrieve
        :return: specified property, if it can't be found the method return False

        :Example:
        >>> cam = Camera()
        >>> p = cam.get_property('width')
        """
        if self._sdl2_cam:
            if p.lower() == 'width':
                return self._cv2_capture.get_size()[0]
            elif p.lower() == 'height':
                return self._cv2_capture.get_size()[1]
            else:
                return False

        if p in self.prop_map:
            return cv.GetCaptureProperty(self._cv2_capture, self.prop_map[p])

        return False

    def get_all_properties(self):
        """
        Return all properties from the camera.

        :return: a dict of all the camera properties.
        """
        if self._sdl2_cam:
            return False

        props = {}

        for p in self.prop_map:
            props[p] = self.get_property(p)

        return props

    def get_image(self):
        """
        Retrieve an Image-object from the camera.  If you experience problems
        with stale frames from the camera's hardware buffer, increase the
        flush cache number to dequeue multiple frames before retrieval
        We're working on how to solve this problem.

        :return: a Image.

        :Example:
        >>> cam = Camera()
        >>> while True:
        >>>     cam.get_image().show()
        """

    @property
    def index(self):
        return self._index

    @property
    def threaded(self):
        return self._threaded

    @property
    def sdl2_cam(self):
        return self._sdl2_cam

    @sdl2_cam.setter
    def sdl2_cam(self, value):
        self._sdl2_cam = value

    @property
    def sdl2_buf(self):
        return self._sdl2_buf

    @sdl2_buf.setter
    def sdl2_buf(self, value):
        self._sdl2_buf = value
        
    @property
    def cv2_capture(self):
        return self._cv2_capture

    @cv2_capture.setter
    def cv2_capture(self, value):
        self._cv2_capture = value


class FrameBufferThread(threading.Thread):
    """
    This is a helper thread which continually debuffers the camera frames.
    If you don't do this, cameras may constantly give a frame behind, which
    case problems at low sample rates. This makes sure the frame returned by
    you camera are fresh.
    """
    def run(self):
        global _gcameras
        while True:
            for cam in _gcameras:
                if cam.sdl2_cam:
                    cam.sdl2_buf = cam.cv2_capture.get_image(cam.sdl2_buf)
                else:
                    cv.GrabFrame(cam.cv2_capture)
                cam._thread_capture_time = time.time()
            time.sleep(0.04)  # max 25 fps, if you're lucky



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




