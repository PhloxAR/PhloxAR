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

if sys.version[0] == 2:
    from urllib2 import urlopen, build_opener
    from urllib2 import HTTPBasicAuthHandler, HTTPPasswordMgrWithDefaultRealm
elif sys.version[0] == 3:
    from urllib import urlopen
    from urllib.request import build_opener, HTTPBasicAuthHandler
    from urllib.request import HTTPPasswordMgrWithDefaultRealm


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
    _cap_time = ''  # timestamp of the last acquired image

    def __init__(self):
        return

    def get_property(self, p):
        return None

    def get_all_properties(self):
        return {}

    @abc.abstractmethod
    def get_image(self):
        return None

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

    def camera_matrix(self):
        """
        Return a cvMat of the camera's intrinsic matrix.
        """
        return self._calib_mat

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

        :return: an Image.

        :Example:
        >>> cam = Camera()
        >>> while True:
        >>>     cam.get_image().show()
        """
        if self._sdl2_cam:
            return Image(self._sdl2_buf.copy())

        if not self._threaded:
            cv.GrabFrame(self._cv2_capture)
            self._cap_time = time.time()
        else:
            self._cap_time = self._thread_cap_time

        frame = cv.RetrieveFrame(self._cv2_capture)
        newing = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 3)
        cv.Copy(frame, newing)

        return Image(newing, self)

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
    """
    The virtual camera lets you test algorithms or functions by providing
    a Camera object which is not a physically connected device.

    Currently, VirtualCamera supports "image", "imageset" and "video" source
    types.

    For image, pass the filename or URL to the image
    For the video, the filename
    For imageset, you can pass either a path or a list of [path, extension]
    For directory you treat a directory to show the latest file, an example
    would be where a security camera logs images to the directory,
    calling .get_image() will get the latest in the directory
    """
    _src = None
    _src_type = None
    _last_time = 0

    def __init__(self, src, src_type, start=1):
        """
        The constructor takes a source, and source type.

        :param src: the source of the imagery
        :param src_type: the type of the virtual camera. Valid strings include
                         "image" - a single still image.
                         "video" - a video file.
                         "imageset" - a PhloxAR image set.
                        "directory" - a VirtualCamera for loading a directory
        :param start: the number of the frame that you want to start with.

        :Example:
        >>> vc = VirtualCamera("img.jpg", "image")
        >>> vc = VirtualCamera("video.mpg", "video")
        >>> vc = VirtualCamera("./path_to_images/", "imageset")
        >>> vc = VirtualCamera("video.mpg", "video", 300)
        >>> vc = VirtualCamera("./imgs", "directory")
        """
        super(VirtualCamera, self).__init__()
        self._src = src
        self._src_type = src_type
        self.counter = 0

        if start == 0:
            start = 1

        self._start = start

        if self._src_type not in ['video', 'image', 'imageset', 'directory']:
            print('Error: In VirtualCamera(), Incorrect Source '
                  'option. "{}" \nUsage:'.format(self._src_type))
            print('\tVirtualCamera("filename","video")')
            print('\tVirtualCamera("filename","image")')
            print('\tVirtualCamera("./path_to_images","imageset")')
            print('\tVirtualCamera("./path_to_images","directory")')
            return
        else:
            if isinstance(self._src, str) and not os.path.exists(self._src):
                print('Error: In VirtualCamera()\n\t "{}" was not found'.format(
                    self._src
                ))
                return
        if self._src_type == 'imageset':
            if isinstance(src, ImageSet):
                self._src = src
            elif isinstance(src, (list, str)):
                self._src = ImageSet()
                if isinstance(src, list):
                    self._src.load(*src)
                else:
                    self._src.load(src)
            else:
                warnings.warn('Virtual Camera is unable to figure out the '
                              'contents of your ImageSet, it must be a '
                              'directory, list of directories, or an '
                              'ImageSet object')
        elif self._src_type == 'video':
            self._cv2_capture = cv.CaptureFromFile(self._src)
            cv.SetCaptureProperty(self._cv2_capture, cv.CV_CAP_PROP_POS_FRAMES,
                                  self._start - 1)
        elif self._src_type == 'directory':
            pass

    def get_image(self):
        """
        Retrieve an Image-object from the virtual camera.

        :return: an Image

        :Example:
        >>> cam = VirtualCamera()
        >>> while True:
            ... cam.get_image().show()
        """
        if self._src_type == 'image':
            self.counter += 1
            return Image(self._src, self)
        elif self._src_type == 'imageset':
            print(len(self._src))
            img = self._src[self.counter % len(self._src)]
            self.counter += 1
            return img
        elif self._src_type == 'video':
            # cv.QueryFrame returns None if the video is finished
            frame = cv.QueryFrame(self._cv2_capture)
            if frame:
                img = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 3)
                cv.Copy(frame, img)
                return Image(img, self)
            else:
                return None
        elif self._src_type == 'directory':
            img = self.find_latest_image(self._src, 'bmp')
            self.counter += 1
            return Image(img, self)

    def rewind(self, start=None):
        """
        Rewind the video source back to the given frame.
        Available for only video sources.

        :param start: the number of the frame you want to rwind to,
                       if not provided, the video source would be rewound
                       to the starting frame number you provided or rewound
                       to the beginning.
        :return: None

        :Example:
        >>> cam = VirtualCamera('file.avi', 'video', 120)
        >>> i = 0
        >>> while i < 60:
            ... cam.get_image().show()
            ... i += 1
        >>> cam.rewind()
        """
        if self._src_type == 'video':
            if not start:
                cv.SetCaptureProperty(self._cv2_capture,
                                      cv.CV_CAP_PROP_POS_FRAMES,
                                      self._start - 1)
            else:
                if start == 0:
                    start = 1
                cv.SetCaptureProperty(self._cv2_capture,
                                      cv.CV_CAP_PROP_POS_FRAMES,
                                      self._start - 1)
        else:
            self.counter = 0

    def get_frame(self, frame):
        """
        Get the provided numbered frame from the video source.
        Available for only video sources.

        :param frame: the number of the frame
        :return: Image

        >>> cam = VimbaCamera('file.avi', 'video', 120)
        >>> cam.get_frame(400).show()
        """
        if self._src_type == 'video':
            num_frame = int(cv.GetCaptureProperty(self._cv2_capture,
                                                  cv.CV_CAP_PROP_POS_FRAMES))
            cv.SetCaptureProperty(self._cv2_capture, cv.CV_CAP_PROP_POS_FRAMES,
                                  frame - 1)
            img = self.get_image()
            cv.SetCaptureProperty(self._cv2_capture, cv.CV_CAP_PROP_POS_FRAMES,
                                  num_frame)
            return img
        elif self._src_type == 'imageset':
            img = None
            if frame < len(self._src):
                img = self._src[frame]
            return img
        else:
            return None

    def skip_frames(self, num):
        """
        Skip num number of frames.
        Available for only video sources

        :param num: number of frames to be skipped
        :return: None

        >>> cam = VirtualCamera('file.avi', 'video', 120)
        >>> i = 0
        >>> while i < 60:
            ... cam.get_image().show()
            ... i += 1
        >>> cam.skip_frames(100)
        >>> cam.get_image().show()
        """
        if self._src_type == 'video':
            num_frame = int(cv.GetCaptureProperty(self._cv2_capture,
                                                  cv.CV_CAP_PROP_POS_FRAMES))
            cv.SetCaptureProperty(self._cv2_capture, cv.CV_CAP_PROP_POS_FRAMES,
                                  num_frame + num - 1)
        elif self._src_type == 'imageset':
            self.counter = (self.counter + num) % len(self._src)
        else:
            self.counter += num

    def get_frame_number(self):
        """
        Get the current frame number of the video source.
        Available for only video sources.

        :return: number of frame, integer

        :Example:
        >>> cam = VirtualCamera('file.avi', 'video', 120)
        >>> i = 0
        >>> while i < 60:
            ... cam.get_image().show()
            ... i += 1
        >>> cam.skip_frames(100)
        >>> cam.get_frame_number()
        """
        if self._src_type == 'video':
            num_frame = int(cv.GetCaptureProperty(self._cv2_capture,
                                                  cv.CV_CAP_PROP_POS_FRAMES))
            return num_frame
        else:
            return self.counter

    def get_current_play_time(self):
        """
        Get the current play time in milliseconds of the video source.
        Available for only video sources.

        :return: int, milliseconds of time from beginning of file.

        :Example:
        >>> cam = VirtualCamera("filename.avi", "video", 120)
        >>> i=0
        >>> while i<60:
            ... cam.getImage().show()
            ... i+=1
        >>> cam.skipFrames(100)
        >>> cam.getCurrentPlayTime()
            """
        if self._src_type == 'video':
            milliseconds = int(cv.GetCaptureProperty(self.capture,
                                                     cv.CV_CAP_PROP_POS_MSEC))
            return milliseconds
        else:
            raise ValueError('sources other than video do not have play '
                             'time property')

    def find_latest_image(self, directory='', ext='png'):
        """
        This function finds the latest file in a directory
        with a given extension.

        :param directory: the directory you want to load images from
                           (defaults to current directory)
        :param ext: image extension you want to use (defaults to .png)

        :return: the filename of the latest image

        :Example:
        >>> cam = VirtualCamera('imgs/', 'png') #find all .png files in 'img' directory
        >>> cam.get_image() # Grab the latest image from that directory
        """
        max_mtime = 0
        max_dir = None
        max_file = None
        max_full_path = None
        for dirname, subdirs, files in os.walk(directory):
            for f in files:
                if f.split('.')[-1] == ext:
                    full_path = os.path.join(dirname, f)
                    mtime = os.stat(full_path).st_mtime
                    if mtime > max_mtime:
                        max_mtime = mtime
                        max_dir = dirname
                        max_file = f
                        self._last_time = mtime
                        max_full_path = os.path.abspath(
                            os.path.join(dirname, f))

        # if file is being written, block until mtime is at least 100ms old
        while time.mktime(time.localtime()) - os.stat(
                max_full_path).st_mtime < 0.1:
            time.sleep(0)

        return max_full_path


class JpegStreamReader(threading.Thread):
    """
     A Threaded class for pulling down JPEG streams and breaking up the images.
     This is handy for reading the stream of images from a IP Camera.
     """
    url = ""
    currentframe = ""
    _thread_cap_time = ""

    def run(self):
        f = ''

        if re.search('@', self.url):
            authstuff = re.findall('//(\S+)@', self.url)[0]
            self.url = re.sub("//\S+@", "//", self.url)
            user, password = authstuff.split(":")

            # thank you missing urllib2 manual
            # http://www.voidspace.org.uk/python/articles/urllib2.shtml#id5
            password_mgr = HTTPPasswordMgrWithDefaultRealm()
            password_mgr.add_password(None, self.url, user, password)

            handler = HTTPBasicAuthHandler(password_mgr)
            opener = build_opener(handler)

            f = opener.open(self.url)
        else:
            f = urlopen(self.url)

        headers = f.info()
        if "content-type" in headers:
            # force upcase first char
            headers['Content-type'] = headers['content-type']

        if "Content-type" not in headers:
            logger.warning("Tried to load a JpegStream from " +
                           self.url +
                           ", but didn't find a content-type header!")
            return

        multipart, boundary = headers['Content-type'].split("boundary=")
        if not re.search("multipart", multipart, re.I):
            logger.warning("Tried to load a JpegStream from " +
                           self.url + ", but the content type header was " +
                           multipart + " not multipart/replace!")
            return

        buff = ''
        data = f.readline().strip()
        length = 0
        contenttype = "jpeg"

        # the first frame contains a boundarystring and some header info
        while True:
            # print data
            if re.search(boundary, data.strip()) and len(buff):
                # we have a full jpeg in buffer.  Convert to an image
                if contenttype == "jpeg":
                    self.currentframe = buff
                    self._thread_cap_time = time.time()
                buff = ''

            if re.match("Content-Type", data, re.I):
                # set the content type, if provided (default to jpeg)
                (header, typestring) = data.split(":")
                (junk, contenttype) = typestring.strip().split("/")

            if re.match("Content-Length", data, re.I):
                # once we have the content length, we know how far to go jfif
                (header, length) = data.split(":")
                length = int(length.strip())

            if (re.search("JFIF", data, re.I) or re.search("\xff\xd8\xff\xdb",
                                                           data) or len(
                    data) > 55):
                # we have reached the start of the image
                buff = ''
                if length and length > len(data):
                    buff += data + f.read(
                        length - len(data))  # read the remainder of the image
                    if contenttype == "jpeg":
                        self.currentframe = buff
                        self._thread_cap_time = time.time()
                else:
                    while not re.search(boundary, data):
                        buff += data
                        data = f.readline()

                    endimg, junk = data.split(boundary)
                    buff += endimg
                    data = boundary
                    continue

            data = f.readline()  # load the next (header) line
            time.sleep(0)  # let the other threads go

    @property
    def thread_capture_time(self):
        return self._thread_cap_time


class JpegStreamCamera(FrameSource):
    """
    The JpegStreamCamera takes a URL of a JPEG stream and treats it like a
    camera.  The current frame can always be accessed with getImage()
    Requires the Python Imaging Library:
    http://www.pythonware.com/library/pil/handbook/index.htm

    :Example:
    Using your Android Phone as a Camera. Softwares like IP Webcam can be used.

    >>> cam = JpegStreamCamera("http://192.168.65.101:8080/videofeed") # your IP may be different.
    >>> img = cam.get_image()
    >>> img.show()
    """
    url = ""
    camthread = ""
    capturetime = 0

    def __init__(self, url):
        super(JpegStreamCamera, self).__init__()
        if not PIL_ENABLED:
            logger.warning("You need the Python Image Library (PIL) to use"
                           " the JpegStreamCamera")
            return
        if not url.startswith('http://'):
            url = "http://" + url
        self.url = url
        self.camthread = JpegStreamReader()
        self.camthread.url = self.url
        self.camthread.daemon = True
        self.camthread.start()
        self.capturetime = 0

    def get_image(self):
        """
        Return the current frame of the JpegStream being monitored
        """
        if not self.camthread.thread_capture_time:
            now = time.time()
            while not self.camthread.thread_capture_time:
                if time.time() - now > 5:
                    warnings.warn("Timeout fetching JpegStream at " + self.url)
                    return
                time.sleep(0.1)

        self.capturetime = self.camthread.thread_capture_time
        return Image(pil.open(StringIO(self.camthread.currentframe)), self)


_SANE_INIT = False


class Scanner(FrameSource):
    """
    The Scanner lets you use any supported SANE-compatable scanner as a camera.
    List of supported devices: http://www.sane-projectypes.org/sane-supported-devices.html
    Requires the PySANE wrapper for libsane.  The sane scanner object
    is available for direct manipulation at Scanner.device
    This scanner object is heavily modified from
    https://bitbucket.org/DavidVilla/pysane
    Constructor takes an index (default 0) and a list of SANE options
    (default is color mode).

    :Example:
    >>> scan = Scanner(0, { "mode": "gray" })
    >>> preview = scan.get_preview()
    >>> stuff = preview.find_blobs(minsize = 1000)
    >>> topleft = (npy.min(stuff.x()), npy.min(stuff.y()))
    >>> bottomright = (npy.max(stuff.x()), npy.max(stuff.y()))
    >>> scan.set_roi(topleft, bottomright)
    >>> scan.set_property("resolution", 1200) #set high resolution
    >>> scan.set_property("mode", "color")
    >>> img = scan.get_image()
    >>> scan.set_roi() #reset region of interest
    >>> img.show()
    """
    usbid = None
    manufacturer = None
    model = None
    kind = None
    device = None
    max_x = None
    max_y = None
    preview = None

    def __init__(self, id=0, properties={'mode': 'color'}):
        super(Scanner, self).__init__()
        global _SANE_INIT
        import sane
        if not _SANE_INIT:
            try:
                sane.init()
                _SANE_INIT = True
            except:
                warn("Initializing pysane failed, do you have pysane installed?")
                return

        devices = sane.get_devices()
        if not len(devices):
            warn("Did not find a sane-compatable device")
            return

        self.usbid, self.manufacturer, self.model, self.kind = devices[id]

        self.device = sane.open(self.usbid)
        self.max_x = self.device.br_x
        self.max_y = self.device.br_y #save our extents for later

        for k, v in properties.items():
            setattr(self.device, k, v)

    def get_image(self):
        """
        Retrieve an Image-object from the scanner.  Any ROI set with
        setROI() is taken into account.

        :return: an Image.  Note that whatever the scanner mode is,
        PhloxAR will return a 3-channel, 8-bit image.

        :Example:
        >>> scan = Scanner()
        >>> scan.get_image().show()
        """
        return Image(self.device.scan())

    def get_preview(self):
        """
        Retrieve a preview-quality Image-object from the scanner.

        :return: Image. Note that whatever the scanner mode is, will
                  return a 3-channel, 8-bit image.

        :Example:
        >>> scan = Scanner()
        >>> scan.get_preview().show()
        """
        self.preview = True
        img = Image(self.device.scan())
        self.preview = False
        return img

    def get_all_properties(self):
        """
        Return a list of all properties and values from the scanner

        :return: Dictionary of active options and values. Inactive
        options appear as "None"

        :Example:
        >>> scan = Scanner()
        >>> print(scan.get_all_properties())
        """
        props = {}
        for prop in self.device.optlist:
            val = None
            if hasattr(self.device, prop):
                val = getattr(self.device, prop)
            props[prop] = val

        return props

    def print_properties(self):

        """
        
        Print detailed information about the SANE device properties
        :return:
        Nothing
        :Example:
        >>> scan = Scanner()
        >>> scan.print_properties()
        """
        for prop in self.device.optlist:
            try:
                print(self.device[prop])
            except:
                pass

    def get_property(self, p):
        """
        
        Returns a single property value from the SANE device
        equivalent to Scanner.device.PROPERTY
        :return:
        Value for option or None if missing/inactive
        :Example:
        >>> scan = Scanner()
        >>> print(scan.get_property('mode'))
        color
        """
        if hasattr(self.device, p):
            return getattr(self.device, p)
        return None

    def set_roi(self, topleft=(0, 0), botright=(-1, -1)):
        """
        Sets an ROI for the scanner in the current resolution.  The
        two parameters, topleft and botright, will default to the
        device extents, so the ROI can be reset by calling setROI with
        no parameters.
        The ROI is set by SANE in resolution independent units (default
        MM) so resolution can be changed after ROI has been set.

        :return: None

        :Example:
        >>> scan = Scanner()
        >>> scan.set_roi((50, 50), (100,100))
        >>> scan.get_image().show() # a very small crop on the scanner
        """
        self.device.tl_x = self.px2mm(topleft[0])
        self.device.tl_y = self.px2mm(topleft[1])
        if botright[0] == -1:
            self.device.br_x = self.max_x
        else:
            self.device.br_x = self.px2mm(botright[0])

        if botright[1] == -1:
            self.device.br_y = self.max_y
        else:
            self.device.br_y = self.px2mm(botright[1])

    def set_property(self, prop, val):
        """
        Assigns a property value from the SANE device
        equivalent to Scanner.device.PROPERTY = VALUE

        :return: None

        :Example:
        >>> scan = Scanner()
        >>> print(scan.getProperty('mode'))
        color
        >>> scan.set_property('mode', 'gray')
        """
        setattr(self.device, prop, val)

    def px2mm(self, pixels=1):
        """
        Helper function to convert native scanner resolution to millimeter units

        :return: float value

        :Example:
        >>> scan = Scanner()
        >>> scan.px2mm(scan.device.resolution) #return DPI in DPMM
        """
        return float(pixels * 25.4 / float(self.device.resolution))


class DigitalCamera(FrameSource):
    """
    The DigitalCamera takes a point-and-shoot camera or high-end slr and uses
    it as a Camera.  The current frame can always be accessed with get_preview()
    Requires the PiggyPhoto Library: https://github.com/alexdu/piggyphoto

    :Example:
    >>> cam = DigitalCamera()
    >>> pre = cam.get_preview()
    >>> pre.find_blobs().show()
    >>>
    >>> img = cam.get_image()
    >>> img.show()
    """
    camera = None
    usbid = None
    device = None

    def __init__(self, id=0):
        super(DigitalCamera, self).__init__()
        try:
            import piggyphoto
        except:
            warn("Initializing piggyphoto failed, do you have "
                 "piggyphoto installed?")
            return

        devices = piggyphoto.cameraList(autodetect=True).toList()
        if not len(devices):
            warn("No compatible digital cameras attached")
            return

        self.device, self.usbid = devices[id]
        self.camera = piggyphoto.camera()

    def get_image(self):
        """
        Retrieve an Image-object from the camera with the highest
        quality possible.

        :return: an Image.

        :Example:
        >>> cam = DigitalCamera()
        >>> cam.get_image().show()
        """
        fd, path = tempfile.mkstemp()
        self.camera.capture_image(path)
        img = Image(path)
        os.close(fd)
        os.remove(path)
        return img

    def get_preview(self):
        """
        Retrieve an Image-object from the camera with the preview quality
        from the camera.


        :return: an Image.

        :Example:
        >>> cam = DigitalCamera()
        >>> cam.get_preview().show()
        """
        fd, path = tempfile.mkstemp()
        self.camera.capture_preview(path)
        img = Image(path)
        os.close(fd)
        os.remove(path)

        return img


class ScreenCamera(object):
    """
    ScreenCapture is a camera class would allow you to capture all or part of
    the screen and return it as a color image. Requires the pyscreenshot
    Library: https://github.com/vijaym123/pyscreenshot

    :Example:
    >>> sc = ScreenCamera()
    >>> res = sc.get_resolution()
    >>> print(res)
    >>> img = sc.get_image()
    >>> img.show()
    """
    _roi = None

    def __init__(self):
        if not PYSCREENSHOT_ENABLED:
            warn("Initializing pyscreenshot failed. Install pyscreenshot from"
                 " https://github.com/vijaym123/pyscreenshot")
            return

    def get_resolution(self):
        """
        returns the resolution of the screenshot of the screen.

        :Example:
        >>> img = ScreenCamera()
        >>> res = img.get_resolution()
        >>> print(res)
        """
        return Image(pyscreenshot.grab()).size()

    @property
    def roi(self):
        return self._roi

    @roi.setter
    def roi(self, roi):
        """
        To set the region of interest.

        :param roi: a tuple of size 4. where region of interest is to the
                     center of the screen.

        :Examples:
        >>> sc = ScreenCamera()
        >>> res = sc.get_resolution()
        >>> sc.roi = (res[0]/4, res[1]/4, res[0]/2, res[1]/2)
        >>> img = sc.get_image()
        >>> img.show()
        """
        if isinstance(roi, tuple) and len(roi) == 4:
            self._roi = roi

    def get_image(self):
        """
        Returns a Image object capturing the current screenshot of the screen.

        :return: the region of interest if ROI is set, otherwise returns
                  the original capture of the screenshot.

        :Examples:
        >>> sc = ScreenCamera()
        >>> img = sc.get_image()
        >>> img.show()
        """
        img = Image(pyscreenshot.grab())
        try:
            if self._roi:
                img = img.crop(self._roi, centered=True)
        except Exception:
            print("Error croping the image. ROI specified is not correctypes.")
            return None
        return img


class StereoImage(object):
    pass


class StereoCamera(object):
    pass


class AVTCameraThread(threading.Thread):
    camera = None
    run = True
    verbose = False
    lock = None
    logger = None
    framerate = 0

    def __init__(self, camera):
        super(AVTCameraThread, self).__init__()
        self._stop = threading.Event()
        self.camera = camera
        self.lock = threading.Lock()
        self.name = 'Thread-Camera-ID-' + str(self.camera.uniqueid)

    def run(self):
        counter = 0
        timestamp = time.time()

        while self.run:
            self.lock.acquire()
            self.camera.run_command("AcquisitionStart")
            frame = self.camera._get_frame(1000)

            if frame:
                img = Image(pil.fromstring(self.camera.imgformat,
                                           (self.camera.width,
                                            self.camera.height),
                                           frame.ImageBuffer[
                                           :int(frame.ImageBufferSize)]))
                self.camera._buffer.appendleft(img)

            self.camera.run_command("AcquisitionStop")
            self.lock.release()
            counter += 1
            time.sleep(0.01)

            if time.time() - timestamp >= 1:
                self.camera.framerate = counter
                counter = 0
                timestamp = time.time()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()


AVTCameraErrors = [
    ("ePvErrSuccess", "No error"),
    ("ePvErrCameraFault", "Unexpected camera fault"),
    ("ePvErrInternalFault", "Unexpected fault in PvApi or driver"),
    ("ePvErrBadHandle", "Camera handle is invalid"),
    ("ePvErrBadParameter", "Bad parameter to API call"),
    ("ePvErrBadSequence", "Sequence of API calls is incorrect"),
    ("ePvErrNotFound", "Camera or attribute not found"),
    ("ePvErrAccessDenied", "Camera cannot be opened in the specified mode"),
    ("ePvErrUnplugged", "Camera was unplugged"),
    ("ePvErrInvalidSetup", "Setup is invalid (an attribute is invalid)"),
    ("ePvErrResources", "System/network resources or memory not available"),
    ("ePvErrBandwidth", "1394 bandwidth not available"),
    ("ePvErrQueueFull", "Too many frames on queue"),
    ("ePvErrBufferTooSmall", "Frame buffer is too small"),
    ("ePvErrCancelled", "Frame cancelled by user"),
    ("ePvErrDataLost", "The data for the frame was lost"),
    ("ePvErrDataMissing", "Some data in the frame is missing"),
    ("ePvErrTimeout", "Timeout during wait"),
    ("ePvErrOutOfRange", "Attribute value is out of the expected range"),
    ("ePvErrWrongType", "Attribute is not this type (wrong access function)"),
    ("ePvErrForbidden", "Attribute write forbidden at this time"),
    ("ePvErrUnavailable", "Attribute is not available at this time"),
    ("ePvErrFirewall", "A firewall is blocking the traffic (Windows only)"),
  ]


def pverr(errcode):
    if errcode:
        raise Exception(": ".join(AVTCameraErrors[errcode]))


class AVTCamera(FrameSource):
    """
    AVTCamera is a ctypes wrapper for the Prosilica/Allied Vision cameras,
    such as the "manta" series.
    These require the PvAVT binary driver from Allied Vision:
    http://www.alliedvisiontec.com/us/products/1108.html
    Note that as of time of writing the new VIMBA driver is not available
    for Mac/Linux - so this uses the legacy PvAVT drive
    Props to Cixelyn, whos py-avt-pvapi module showed how to get much
    of this working https://bitbucket.org/Cixelyn/py-avt-pvapi
    All camera properties are directly from the PvAVT manual -- if not
    specified it will default to whatever the camera state is.  Cameras
    can either by

    :Example:
    >>> cam = AVTCamera(0, {"width": 656, "height": 492})
    >>>
    >>> img = cam.get_image()
    >>> img.show()
    """

    _buffer = None  # Buffer to store images
    _buffersize = 10  # Number of images to keep in the rolling image buffer for threads
    _lastimage = None  # Last image loaded into memory
    _thread = None
    _framerate = 0
    threaded = False
    _pvinfo = {}
    _properties = {
        "AcqEndTriggerEvent": ("Enum", "R/W"),
        "AcqEndTriggerMode": ("Enum", "R/W"),
        "AcqRecTriggerEvent": ("Enum", "R/W"),
        "AcqRecTriggerMode": ("Enum", "R/W"),
        "AcqStartTriggerEvent": ("Enum", "R/W"),
        "AcqStartTriggerMode": ("Enum", "R/W"),
        "FrameRate": ("Float32", "R/W"),
        "FrameStartTriggerDelay": ("Uint32", "R/W"),
        "FrameStartTriggerEvent": ("Enum", "R/W"),
        "FrameStartTriggerMode": ("Enum", "R/W"),
        "FrameStartTriggerOverlap": ("Enum", "R/W"),
        "AcquisitionFrameCount": ("Uint32", "R/W"),
        "AcquisitionMode": ("Enum", "R/W"),
        "RecorderPreEventCount": ("Uint32", "R/W"),
        "ConfigFileIndex": ("Enum", "R/W"),
        "ConfigFilePowerup": ("Enum", "R/W"),
        "DSPSubregionBottom": ("Uint32", "R/W"),
        "DSPSubregionLeft": ("Uint32", "R/W"),
        "DSPSubregionRight": ("Uint32", "R/W"),
        "DSPSubregionTop": ("Uint32", "R/W"),
        "DefectMaskColumnEnable": ("Enum", "R/W"),
        "ExposureAutoAdjustTol": ("Uint32", "R/W"),
        "ExposureAutoAlg": ("Enum", "R/W"),
        "ExposureAutoMax": ("Uint32", "R/W"),
        "ExposureAutoMin": ("Uint32", "R/W"),
        "ExposureAutoOutliers": ("Uint32", "R/W"),
        "ExposureAutoRate": ("Uint32", "R/W"),
        "ExposureAutoTarget": ("Uint32", "R/W"),
        "ExposureMode": ("Enum", "R/W"),
        "ExposureValue": ("Uint32", "R/W"),
        "GainAutoAdjustTol": ("Uint32", "R/W"),
        "GainAutoMax": ("Uint32", "R/W"),
        "GainAutoMin": ("Uint32", "R/W"),
        "GainAutoOutliers": ("Uint32", "R/W"),
        "GainAutoRate": ("Uint32", "R/W"),
        "GainAutoTarget": ("Uint32", "R/W"),
        "GainMode": ("Enum", "R/W"),
        "GainValue": ("Uint32", "R/W"),
        "LensDriveCommand": ("Enum", "R/W"),
        "LensDriveDuration": ("Uint32", "R/W"),
        "LensVoltage": ("Uint32", "R/V"),
        "LensVoltageControl": ("Uint32", "R/W"),
        "IrisAutoTarget": ("Uint32", "R/W"),
        "IrisMode": ("Enum", "R/W"),
        "IrisVideoLevel": ("Uint32", "R/W"),
        "IrisVideoLevelMax": ("Uint32", "R/W"),
        "IrisVideoLevelMin": ("Uint32", "R/W"),
        "VsubValue": ("Uint32", "R/C"),
        "WhitebalAutoAdjustTol": ("Uint32", "R/W"),
        "WhitebalAutoRate": ("Uint32", "R/W"),
        "WhitebalMode": ("Enum", "R/W"),
        "WhitebalValueRed": ("Uint32", "R/W"),
        "WhitebalValueBlue": ("Uint32", "R/W"),
        "EventAcquisitionStart": ("Uint32", "R/C 40000"),
        "EventAcquisitionEnd": ("Uint32", "R/C 40001"),
        "EventFrameTrigger": ("Uint32", "R/C 40002"),
        "EventExposureEnd": ("Uint32", "R/C 40003"),
        "EventAcquisitionRecordTrigger": ("Uint32", "R/C 40004"),
        "EventSyncIn1Rise": ("Uint32", "R/C 40010"),
        "EventSyncIn1Fall": ("Uint32", "R/C 40011"),
        "EventSyncIn2Rise": ("Uint32", "R/C 40012"),
        "EventSyncIn2Fall": ("Uint32", "R/C 40013"),
        "EventSyncIn3Rise": ("Uint32", "R/C 40014"),
        "EventSyncIn3Fall": ("Uint32", "R/C 40015"),
        "EventSyncIn4Rise": ("Uint32", "R/C 40016"),
        "EventSyncIn4Fall": ("Uint32", "R/C 40017"),
        "EventOverflow": ("Uint32", "R/C 65534"),
        "EventError": ("Uint32", "R/C"),
        "EventNotification": ("Enum", "R/W"),
        "EventSelector": ("Enum", "R/W"),
        "EventsEnable1": ("Uint32", "R/W"),
        "BandwidthCtrlMode": ("Enum", "R/W"),
        "ChunkModeActive": ("Boolean", "R/W"),
        "NonImagePayloadSize": ("Unit32", "R/V"),
        "PayloadSize": ("Unit32", "R/V"),
        "StreamBytesPerSecond": ("Uint32", "R/W"),
        "StreamFrameRateConstrain": ("Boolean", "R/W"),
        "StreamHoldCapacity": ("Uint32", "R/V"),
        "StreamHoldEnable": ("Enum", "R/W"),
        "TimeStampFrequency": ("Uint32", "R/C"),
        "TimeStampValueHi": ("Uint32", "R/V"),
        "TimeStampValueLo": ("Uint32", "R/V"),
        "Height": ("Uint32", "R/W"),
        "RegionX": ("Uint32", "R/W"),
        "RegionY": ("Uint32", "R/W"),
        "Width": ("Uint32", "R/W"),
        "PixelFormat": ("Enum", "R/W"),
        "TotalBytesPerFrame": ("Uint32", "R/V"),
        "BinningX": ("Uint32", "R/W"),
        "BinningY": ("Uint32", "R/W"),
        "CameraName": ("String", "R/W"),
        "DeviceFirmwareVersion": ("String", "R/C"),
        "DeviceModelName": ("String", "R/W"),
        "DevicePartNumber": ("String", "R/C"),
        "DeviceSerialNumber": ("String", "R/C"),
        "DeviceVendorName": ("String", "R/C"),
        "FirmwareVerBuild": ("Uint32", "R/C"),
        "FirmwareVerMajor": ("Uint32", "R/C"),
        "FirmwareVerMinor": ("Uint32", "R/C"),
        "PartClass": ("Uint32", "R/C"),
        "PartNumber": ("Uint32", "R/C"),
        "PartRevision": ("String", "R/C"),
        "PartVersion": ("String", "R/C"),
        "SerialNumber": ("String", "R/C"),
        "SensorBits": ("Uint32", "R/C"),
        "SensorHeight": ("Uint32", "R/C"),
        "SensorType": ("Enum", "R/C"),
        "SensorWidth": ("Uint32", "R/C"),
        "UniqueID": ("Uint32", "R/C"),
        "Strobe1ControlledDuration": ("Enum", "R/W"),
        "Strobe1Delay": ("Uint32", "R/W"),
        "Strobe1Duration": ("Uint32", "R/W"),
        "Strobe1Mode": ("Enum", "R/W"),
        "SyncIn1GlitchFilter": ("Uint32", "R/W"),
        "SyncInLevels": ("Uint32", "R/V"),
        "SyncOut1Invert": ("Enum", "R/W"),
        "SyncOut1Mode": ("Enum", "R/W"),
        "SyncOutGpoLevels": ("Uint32", "R/W"),
        "DeviceEthAddress": ("String", "R/C"),
        "HostEthAddress": ("String", "R/C"),
        "DeviceIPAddress": ("String", "R/C"),
        "HostIPAddress": ("String", "R/C"),
        "GvcpRetries": ("Uint32", "R/W"),
        "GvspLookbackWindow": ("Uint32", "R/W"),
        "GvspResentPercent": ("Float32", "R/W"),
        "GvspRetries": ("Uint32", "R/W"),
        "GvspSocketBufferCount": ("Enum", "R/W"),
        "GvspTimeout": ("Uint32", "R/W"),
        "HeartbeatInterval": ("Uint32", "R/W"),
        "HeartbeatTimeout": ("Uint32", "R/W"),
        "MulticastEnable": ("Enum", "R/W"),
        "MulticastIPAddress": ("String", "R/W"),
        "PacketSize": ("Uint32", "R/W"),
        "StatDriverType": ("Enum", "R/V"),
        "StatFilterVersion": ("String", "R/C"),
        "StatFrameRate": ("Float32", "R/V"),
        "StatFramesCompleted": ("Uint32", "R/V"),
        "StatFramesDropped": ("Uint32", "R/V"),
        "StatPacketsErroneous": ("Uint32", "R/V"),
        "StatPacketsMissed": ("Uint32", "R/V"),
        "StatPacketsReceived": ("Uint32", "R/V"),
        "StatPacketsRequested": ("Uint32", "R/V"),
        "StatPacketResent": ("Uint32", "R/V")
    }

    class AVTCameraInfo(ctypes.Structure):
        """
        AVTCameraInfo is an internal ctypes.Structure-derived class which
        contains metadata about cameras on the local network.
        Properties include:
        - UniqueId
        - CameraName
        - ModelName
        - PartNumber
        - SerialNumber
        - FirmwareVersion
        - PermittedAccess
        - InterfaceId
        - InterfaceType
        """
        _fields_ = [
            ("StructVer", ctypes.c_ulong),
            ("UniqueId", ctypes.c_ulong),
            ("CameraName", ctypes.c_char * 32),
            ("ModelName", ctypes.c_char * 32),
            ("PartNumber", ctypes.c_char * 32),
            ("SerialNumber", ctypes.c_char * 32),
            ("FirmwareVersion", ctypes.c_char * 32),
            ("PermittedAccess", ctypes.c_long),
            ("InterfaceId", ctypes.c_ulong),
            ("InterfaceType", ctypes.c_int)
        ]

        def __repr__(self):
            return "<PhloxAR.Camera.AVTCameraInfo - UniqueId: %s>" % (self.UniqueId)

    class AVTFrame(ctypes.Structure):
        def __init__(self, buffersize):
            self.ImageBuffer = ctypes.create_string_buffer(buffersize)
            self.ImageBufferSize = ctypes.c_ulong(buffersize)
            self.AncillaryBuffer = 0
            self.AncillaryBufferSize = 0
            self.img = None
            self.hasImage = False
            self.frame = None

        _fields_ = [
            ("ImageBuffer", ctypes.POINTER(ctypes.c_char)),
            ("ImageBufferSize", ctypes.c_ulong),
            ("AncillaryBuffer", ctypes.c_int),
            ("AncillaryBufferSize", ctypes.c_int),
            ("Context", ctypes.c_int * 4),
            ("_reserved1", ctypes.c_ulong * 8),

            ("Status", ctypes.c_int),
            ("ImageSize", ctypes.c_ulong),
            ("AncillarySize", ctypes.c_ulong),
            ("Width", ctypes.c_ulong),
            ("Height", ctypes.c_ulong),
            ("RegionX", ctypes.c_ulong),
            ("RegionY", ctypes.c_ulong),
            ("Format", ctypes.c_int),
            ("BitDepth", ctypes.c_ulong),
            ("BayerPattern", ctypes.c_int),
            ("FrameCount", ctypes.c_ulong),
            ("TimestampLo", ctypes.c_ulong),
            ("TimestampHi", ctypes.c_ulong),
            ("_reserved2", ctypes.c_ulong * 32)
        ]

    def __del__(self):
        # This function should disconnect from the AVT Camera
        pverr(self.dll.PvCameraClose(self.handle))

    def __init__(self, camera_id=-1, properties={}, threaded=False):
        super(AVTCamera, self).__init__()
        import platform

        if platform.system() == "Windows":
            self.dll = ctypes.windll.LoadLibrary("PvAPI.dll")
        elif platform.system() == "Darwin":
            self.dll = ctypes.CDLL("libPvAPI.dylib", ctypes.RTLD_GLOBAL)
        else:
            self.dll = ctypes.CDLL("libPvAPI.so")

        if not self._pvinfo.get("initialized", False):
            self.dll.PvInitialize()
            self._pvinfo['initialized'] = True
        # initialize.  Note that we rely on listAllCameras being the next
        # call, since it blocks on cameras initializing

        camlist = self.list_all_cameras()

        if not len(camlist):
            raise Exception("Couldn't find any cameras with the PvAVT driver.  "
                            "Use SampleViewer to confirm you have one connected.")

        if camera_id < 9000:  # camera was passed as an index reference
            if camera_id == -1:  # accept -1 for "first camera"
                camera_id = 0

            camera_id = camlist[camera_id].UniqueId

        camera_id = long(camera_id)
        self.handle = ctypes.c_uint()
        init_count = 0
        while self.dll.PvCameraOpen(camera_id, 0, ctypes.byref(
                self.handle)) != 0:  # wait until camera is availble
            if init_count > 4:  # Try to connect 5 times before giving up
                raise Exception(
                    'Could not connect to camera, please verify with SampleViewer you can connect')
            init_count += 1
            time.sleep(1)  # sleep and retry to connect to camera in a second

        pverr(self.dll.PvCaptureStart(self.handle))
        self.uniqueid = camera_id

        self.set_property("AcquisitionMode", "SingleFrame")
        self.set_property("FrameStartTriggerMode", "Freerun")

        if properties.get("mode", "RGB") == 'gray':
            self.set_property("PixelFormat", "Mono8")
        else:
            self.set_property("PixelFormat", "Rgb24")

        # give some compatablity with other cameras
        if properties.get("mode", ""):
            properties.pop("mode")

        if properties.get("height", ""):
            properties["Height"] = properties["height"]
            properties.pop("height")

        if properties.get("width", ""):
            properties["Width"] = properties["width"]
            properties.pop("width")

        for p in properties:
            self.set_property(p, properties[p])

        if threaded:
            self._thread = AVTCameraThread(self)
            self._thread.daemon = True
            self._buffer = deque(maxlen=self._buffersize)
            self._thread.start()
            self.threaded = True
        self.frame = None
        self._refresh_frame_stats()

    def restart(self):
        """
        This tries to restart the camera thread
        """
        self._thread.stop()
        self._thread = AVTCameraThread(self)
        self._thread.daemon = True
        self._buffer = deque(maxlen=self._buffersize)
        self._thread.start()

    def list_all_cameras(self):
        """
        List all cameras attached to the host

        :return:
        List of AVTCameraInfo objects, otherwise empty list
        """
        camlist = (self.AVTCameraInfo * 100)()
        starttime = time.time()
        while int(camlist[0].UniqueId) == 0 and time.time() - starttime < 10:
            self.dll.PvCameraListEx(ctypes.byref(camlist), 100, None,
                                    ctypes.sizeof(self.AVTCameraInfo))
            time.sleep(0.1)  # keep checking for cameras until timeout

        return [cam for cam in camlist if cam.UniqueId != 0]

    def run_command(self, command):
        """
        Runs a PvAVT Command on the camera
        Valid Commands include:
        - FrameStartTriggerSoftware
        - AcquisitionAbort
        - AcquisitionStart
        - AcquisitionStop
        - ConfigFileLoad
        - ConfigFileSave
        - TimeStampReset
        - TimeStampValueLatch

        :return: 0 on success

        :Example
        >>> c = AVTCamera()
        >>> c.run_command("TimeStampReset")
        """
        return self.dll.PvCommandRun(self.handle, command)

    def get_property(self, name):
        """
        This retrieves the value of the AVT Camera attribute
        There are around 140 properties for the AVT Camera, so reference the
        AVT Camera and Driver Attributes pdf that is provided with
        the driver for detailed information
        Note that the error codes are currently ignored, so empty values
        may be returned.

        :Example:
        >>>c = AVTCamera()
        >>>print(c.get_property("ExposureValue"))
        """
        valtype, perm = self._properties.get(name, (None, None))

        if not valtype:
            return None

        val = ''
        err = 0
        if valtype == "Enum":
            val = ctypes.create_string_buffer(100)
            vallen = ctypes.c_long()
            err = self.dll.PvAttrEnumGet(self.handle, name, val, 100,
                                         ctypes.byref(vallen))
            val = str(val[:vallen.value])
        elif valtype == "Uint32":
            val = ctypes.c_uint()
            err = self.dll.PvAttrUint32Get(self.handle, name, ctypes.byref(val))
            val = int(val.value)
        elif valtype == "Float32":
            val = ctypes.c_float()
            err = self.dll.PvAttrFloat32Get(self.handle, name, ctypes.byref(val))
            val = float(val.value)
        elif valtype == "String":
            val = ctypes.create_string_buffer(100)
            vallen = ctypes.c_long()
            err = self.dll.PvAttrStringGet(self.handle, name, val, 100,
                                           ctypes.byref(vallen))
            val = str(val[:vallen.value])
        elif valtype == "Boolean":
            val = ctypes.c_bool()
            err = self.dll.PvAttrBooleanGet(self.handle, name, ctypes.byref(val))
            val = bool(val.value)

        # TODO, handle error codes

        return val

    # TODO, implement the PvAttrRange* functions
    # def getPropertyRange(self, name)

    def get_all_properties(self):
        """
        This returns a dict with the name and current value of the
        documented PvAVT attributes
        CAVEAT: it addresses each of the properties individually, so
        this may take time to run if there's network latency

        :Example:
        >>>c = AVTCamera(0)
        >>>props = c.get_all_properties()
        >>>print(props['ExposureValue'])
        """
        props = {}
        for p in self._properties.keys():
            props[p] = self.get_property(p)

        return props

    def set_property(self, name, value, skip_buffer_size_check=False):
        """
        This sets the value of the AVT Camera attribute.
        There are around 140 properties for the AVT Camera, so reference the
        AVT Camera and Driver Attributes pdf that is provided with
        the driver for detailed information
        By default, we will also refresh the height/width and bytes per
        frame we're expecting -- you can manually bypass this if you want speed
        Returns the raw PvAVT error code (0 = success)

        :Example:
        >>>c = AVTCamera()
        >>>c.set_property("ExposureValue", 30000)
        >>>c.get_image().show()
        """
        valtype, perm = self._properties.get(name, (None, None))

        if not valtype:
            return None

        if valtype == "Uint32":
            err = self.dll.PvAttrUint32Set(self.handle, name,
                                           ctypes.c_uint(int(value)))
        elif valtype == "Float32":
            err = self.dll.PvAttrFloat32Set(self.handle, name,
                                            ctypes.c_float(float(value)))
        elif valtype == "Enum":
            err = self.dll.PvAttrEnumSet(self.handle, name, str(value))
        elif valtype == "String":
            err = self.dll.PvAttrStringSet(self.handle, name, str(value))
        elif valtype == "Boolean":
            err = self.dll.PvAttrBooleanSet(self.handle, name,
                                            ctypes.c_bool(bool(value)))

        # just to be safe, re-cache the camera metadata
        if not skip_buffer_size_check:
            self._refresh_frame_stats()

        return err

    def get_image(self, timeout=5000):
        """
        Extract an Image from the Camera, returning the value.  No matter
        what the image characteristics on the camera, the Image returned
        will be RGB 8 bit depth, if camera is in greyscale mode it will
        be 3 identical channels.

        :Example:
        >>>c = AVTCamera()
        >>>c.get_image().show()
        """

        if self.frame is not None:
            st = time.time()
            try:
                pverr(self.dll.PvCaptureWaitForFrameDone(self.handle,
                                                         ctypes.byref(self.frame),
                                                         timeout))
            except Exception as e:
                print("Exception waiting for frame:", e)
                print("Time taken:", time.time() - st)
                self.frame = None
                raise e
            img = self.unbuffer()
            self.frame = None
            return img
        elif self.threaded:
            self._thread.lock.acquire()
            try:
                img = self._buffer.pop()
                self._lastimage = img
            except IndexError:
                img = self._lastimage
            self._thread.lock.release()

        else:
            self.run_command("AcquisitionStart")
            frame = self._get_frame(timeout)
            img = Image(pil.fromstring(self.imgformat,
                                       (self.width, self.height),
                                       frame.ImageBuffer[
                                       :int(frame.ImageBufferSize)]))
            self.run_command("AcquisitionStop")
        return img

    def setup_async_mode(self):
        self.set_property('AcquisitionMode', 'SingleFrame')
        self.set_property('FrameStartTriggerMode', 'Software')

    def setup_sync_mode(self):
        self.set_property('AcquisitionMode', 'Continuous')
        self.set_property('FrameStartTriggerMode', 'FreeRun')

    def unbuffer(self):
        img = Image(pil.fromstring(self.imgformat,
                                   (self.width, self.height),
                                   self.frame.ImageBuffer[:int(self.frame.ImageBufferSize)]))

        return img

    def _refresh_frame_stats(self):
        self.width = self.get_property("Width")
        self.height = self.get_property("Height")
        self.buffersize = self.get_property("TotalBytesPerFrame")
        self.pixelformat = self.get_property("PixelFormat")
        self.imgformat = 'RGB'
        if self.pixelformat == 'Mono8':
            self.imgformat = 'L'

    def _get_frame(self, timeout=5000):
        # return the AVTFrame object from the camera, timeout in ms
        # need to multiply by bitdepth
        try:
            frame = self.AVTFrame(self.buffersize)
            pverr(self.dll.PvCaptureQueueFrame(self.handle, ctypes.byref(frame),
                                               None))
            st = time.time()
            try:
                pverr(self.dll.PvCaptureWaitForFrameDone(self.handle,
                                                         ctypes.byref(frame),
                                                         timeout))
            except Exception as e:
                print("Exception waiting for frame:", e)
                print("Time taken:", time.time() - st)
                raise e

        except Exception as e:
            print("Exception aquiring frame:", e)
            raise e

        return frame

    def acquire(self):
        self.frame = self.AVTFrame(self.buffersize)
        try:
            self.run_command("AcquisitionStart")
            pverr(
                self.dll.PvCaptureQueueFrame(self.handle,
                                             ctypes.byref(self.frame),
                                             None))
            self.run_command("AcquisitionStop")
        except Exception as e:
            print("Exception aquiring frame:", e)
            raise (e)


class GigECamera(Camera):
    """
    GigE Camera driver via Aravis
    """
    def __init__(self, camera_id=None, properties={}, threaded=False):
        super(GigECamera, self).__init__()
        try:
            from gi.repository import Aravis
        except ImportError:
            print("GigE is supported by the Aravis library, download and "
                  "build from https://github.com/sightmachine/aravis")
            print("Note that you need to set GI_TYPELIB_PATH=$GI_TYPELIB_PATH:"
                  "(PATH_TO_ARAVIS)/src for the GObject Introspection")
            sys.exit()

        self._cam = Aravis.Camera.new(None)

        self._pixel_mode = "RGB"
        if properties.get("mode", False):
            self._pixel_mode = properties.pop("mode")

        if self._pixel_mode == "gray":
            self._cam.set_pixel_format(Aravis.PIXEL_FORMAT_MONO_8)
        else:
            self._cam.set_pixel_format(
                Aravis.PIXEL_FORMAT_BAYER_BG_8)  # we'll use bayer (basler cams)
            # TODO, deal with other pixel formats

        if properties.get("roi", False):
            roi = properties['roi']
            self._cam.set_region(*roi)
            # TODO, check sensor size

        if properties.get("width", False):
            # TODO, set internal function to scale results of getimage
            pass

        if properties.get("framerate", False):
            self._cam.set_frame_rate(properties['framerate'])

        self._stream = self._cam.create_stream(None, None)

        payload = self._cam.get_payload()
        self._stream.push_buffer(Aravis.Buffer.new_allocate(payload))
        [x, y, width, height] = self._cam.get_region()
        self._height, self._width = height, width

    def get_image(self):

        camera = self._cam
        camera.start_acquisition()
        buff = self._stream.pop_buffer()
        self._cap_time = buff.timestamp_ns / 1000000.0
        img = npy.fromstring(ctypes.string_at(buff.data_address(), buff.size),
                             dtype=npy.uint8).reshape(self._height, self._width)
        rgb = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
        self._stream.push_buffer(buff)
        camera.stop_acquisition()
        # TODO, we should handle software triggering (separate capture and get image events)

        return Image(rgb)

    def get_property_list(self):
        l = [
            'available_pixel_formats',
            'available_pixel_formats_as_display_names',
            'available_pixel_formats_as_strings',
            'binning',
            'device_id',
            'exposure_time',
            'exposure_time_bounds',
            'frame_rate',
            'frame_rate_bounds',
            'gain',
            'gain_bounds',
            'height_bounds',
            'model_name',
            'payload',
            'pixel_format',
            'pixel_format_as_string',
            'region',
            'sensor_size',
            'trigger_source',
            'vendor_name',
            'width_bounds'
        ]
        return l

    def get_property(self, p=None):
        """
        This function get's the properties available to the camera
        Usage:
          > camera.getProperty('region')
          > (0, 0, 128, 128)
  
        Available Properties:
          see function camera.get_property_list()
        """
        if p is None:
            print("You need to provide a property, available properties are:")
            print("")
            for p in self.get_property_list():
                print(p)
            return

        stringval = "get_{}".format(p)
        try:
            return getattr(self._cam, stringval)()
        except Exception:
            print('Property {} does not appear to exist'.format(p))
            return None

    def set_property(self, p=None, *args):
        """
        This function sets the property available to the camera
        Usage:
          > camera.setProperty('region',(256,256))
        Available Properties:
          see function camera.get_property_list()
        """

        if p is None:
            print("You need to provide a property, available properties are:")
            print("")
            for p in self.get_property_list():
                print(p)
            return

        if len(args) <= 0:
            print("You must provide a value to set")
            return

        stringval = "set_{}".format(p)

        try:
            return getattr(self._cam, stringval)(*args)
        except Exception:
            print('Property {} does not appear to exist or value is not '
                  'in correct format'.format(p))
            return None

    def get_all_properties(self):
        """
        This function just prints out all the properties available to the camera
        """

        for p in self.get_property():
            print("{}: {}".format(p, self.get_property(p)))


class VimbaCamera(FrameSource):
    """
    VimbaCamera is a wrapper for the Allied Vision cameras, such as
    the "manta" series. This requires the

    1) Vimba SDK provided from Allied Vision
       http://www.alliedvisiontec.com/us/products/software/vimba-sdk.html
    2) Pyvimba Python library
       TODO: <INSERT URL>
       Note that as of time of writing, the VIMBA driver is not available
       for Mac.
    All camera properties are directly from the Vimba SDK manual -- if not
    specified it will default to whatever the camera state is.

    :Example:
    >>> cam = VimbaCamera(0, {"width": 656, "height": 492})
    >>>
    >>> img = cam.get_image()
    >>> img.show()
    """

    def _setupVimba(self):
        from pymba import Vimba

        self._vimba = Vimba()
        self._vimba.startup()
        system = self._vimba.getSystem()
        if system.GeVTLIsPresent:
            system.runFeatureCommand("GeVDiscoveryAllOnce")
            time.sleep(0.2)

    def __del__(self):
        # This function should disconnect from the Vimba Camera
        if self._camera is not None:
            if self.threaded:
                self._thread.stop()
                time.sleep(0.2)

            if self._frame is not None:
                self._frame.revokeFrame()
                self._frame = None

            self._camera.closeCamera()

        self._vimba.shutdown()

    def shutdown(self):
        """
        You must call this function if you are using threaded=true when you are finished
        to prevent segmentation fault
        """
        # REQUIRED TO PREVENT SEGMENTATION FAULT FOR THREADED=True
        if self._camera:
            self._camera.closeCamera()

        self._vimba.shutdown()

    def __init__(self, camera_id=-1, properties={}, threaded=False):
        super(VimbaCamera, self).__init__()
        if not VIMBA_ENABLED:
            raise Exception(
                "You don't seem to have the pymba library installed.  This will make it hard to use a AVT Vimba Camera.")

        self._vimba = None
        self._setupVimba()

        camlist = self.listAllCameras()
        self._camTable = {}
        self._frame = None
        self._buffer = None  # Buffer to store images
        self._buffersize = 10  # Number of images to keep in the rolling image buffer for threads
        self._lastimage = None  # Last image loaded into memory
        self._thread = None
        self._framerate = 0
        self.threaded = False
        self._properties = {}
        self._camera = None

        i = 0
        for cam in camlist:
            self._camTable[i] = {'id': cam.cameraIdString}
            i += 1

        if not len(camlist):
            raise Exception("Couldn't find any cameras with the Vimba driver.  "
                            "Use VimbaViewer to confirm you have one connected.")

        if camera_id < 9000:  # camera was passed as an index reference
            if camera_id == -1:  # accept -1 for "first camera"
                camera_id = 0

            if camera_id > len(camlist):
                raise Exception("Couldn't find camera at index %d." % camera_id)

            cam_guid = camlist[camera_id].cameraIdString
        else:
            raise Exception("Index %d is too large" % camera_id)

        self._camera = self._vimba.getCamera(cam_guid)
        self._camera.openCamera()

        self.uniqueid = cam_guid

        self.setProperty("AcquisitionMode", "SingleFrame")
        self.setProperty("TriggerSource", "Freerun")

        # TODO: FIX
        if properties.get("mode", "RGB") == 'gray':
            self.setProperty("PixelFormat", "Mono8")
        else:
            fmt = "RGB8Packed"  # alternatively use BayerRG8
            self.setProperty("PixelFormat", "BayerRG8")

        # give some compatablity with other cameras
        if properties.get("mode", ""):
            properties.pop("mode")

        if properties.get("height", ""):
            properties["Height"] = properties["height"]
            properties.pop("height")

        if properties.get("width", ""):
            properties["Width"] = properties["width"]
            properties.pop("width")

        for p in properties:
            self.setProperty(p, properties[p])

        if threaded:
            self._thread = VimbaCameraThread(self)
            self._thread.daemon = True
            self._buffer = deque(maxlen=self._buffersize)
            self._thread.start()
            self.threaded = True

        self._refreshFrameStats()

    def restart(self):
        """
        This tries to restart the camera thread
        """
        self._thread.stop()
        self._thread = VimbaCameraThread(self)
        self._thread.daemon = True
        self._buffer = deque(maxlen=self._buffersize)
        self._thread.start()

    def list_all_cameras(self):
        """
        List all cameras attached to the host

        :return: List of VimbaCamera objects, otherwise empty list
                  VimbaCamera objects are defined in the pymba module
        """
        cameraIds = self._vimba.getCameraIds()
        ar = []
        for cameraId in cameraIds:
            ar.append(self._vimba.getCamera(cameraId))
        return ar

    def run_command(self, command):
        """
        Runs a Vimba Command on the camera
        Valid Commands include:
        - AcquisitionAbort
        - AcquisitionStart
        - AcquisitionStop

        :return: 0 on success

        :Example:
        >>> c = VimbaCamera()
        >>> c.run_command("TimeStampReset")
        """
        return self._camera.runFeatureCommand(command)

    def get_property(self, p):
        """
        This retrieves the value of the Vimba Camera attribute
        There are around 140 properties for the Vimba Camera, so reference the
        Vimba Camera pdf that is provided with
        the SDK for detailed information
        Throws VimbaException if property is not found or not implemented yet.

        :Example:
        >>>c = VimbaCamera()
        >>>print(c.get_property("ExposureMode"))
        """
        return self._camera.__getattr__(p)

    # TODO, implement the PvAttrRange* functions
    # def getPropertyRange(self, name)

    def get_all_properties(self):
        """
        This returns a dict with the name and current value of the
        documented Vimba attributes
        CAVEAT: it addresses each of the properties individually, so
        this may take time to run if there's network latency

        :Example:
        >>>c = VimbaCamera(0)
        >>>props = c.get_all_properties()
        >>>print(props['ExposureMode'])
        """
        from pymba import VimbaException

        # TODO
        ar = {}
        c = self._camera
        cameraFeatureNames = c.getFeatureNames()
        for name in cameraFeatureNames:
            try:
                ar[name] = c.__getattr__(name)
            except VimbaException:
                # Ignore features not yet implemented
                pass
        return ar

    def set_property(self, name, value, skip_buffer_size_check=False):
        """
        This sets the value of the Vimba Camera attribute.
        There are around 140 properties for the Vimba Camera, so reference the
        Vimba Camera pdf that is provided with
        the SDK for detailed information
        Throws VimbaException if property not found or not yet implemented

        :Example:
        >>> c = VimbaCamera()
        >>> c.setProperty("ExposureAutoRate", 200)
        >>> c.getImage().show()
        """
        ret = self._camera.__setattr__(name, value)

        # just to be safe, re-cache the camera metadata
        if not skip_buffer_size_check:
            self._refreshFrameStats()

        return ret

    def get_image(self):
        """
        Extract an Image from the Camera, returning the value.  No matter
        what the image characteristics on the camera, the Image returned
        will be RGB 8 bit depth, if camera is in greyscale mode it will
        be 3 identical channels.

        :Example:
        >>>c = VimbaCamera()
        >>>c.get_image().show()
        """

        if self.threaded:
            self._thread.lock.acquire()
            try:
                img = self._buffer.pop()
                self._lastimage = img
            except IndexError:
                img = self._lastimage
            self._thread.lock.release()

        else:
            img = self._capture_frame()

        return img

    def setup_async_mode(self):
        self.set_property('AcquisitionMode', 'SingleFrame')
        self.set_property('TriggerSource', 'Software')

    def setup_sync_mode(self):
        self.set_property('AcquisitionMode', 'SingleFrame')
        self.set_property('TriggerSource', 'Freerun')

    def _refresh_frame_stats(self):
        self.width = self.getProperty("Width")
        self.height = self.getProperty("Height")
        self.pixelformat = self.getProperty("PixelFormat")
        self.imgformat = 'RGB'
        if self.pixelformat == 'Mono8':
            self.imgformat = 'L'

    def _get_frame(self):
        if not self._frame:
            self._frame = self._camera.getFrame()  # creates a frame
            self._frame.announceFrame()

        return self._frame

    def _capture_frame(self, timeout=5000):
        try:
            c = self._camera
            f = self._get_frame()

            colorSpace = ColorSpace.BGR
            if self.pixelformat == 'Mono8':
                colorSpace = ColorSpace.GRAY

            c.startCapture()
            f.queueFrameCapture()
            c.runFeatureCommand('AcquisitionStart')
            c.runFeatureCommand('AcquisitionStop')
            try:
                f.waitFrameCapture(timeout)
            except Exception as e:
                print("Exception waiting for frame: %s: %s" % (e, traceback.format_exc()))
                raise e

            imgData = f.getBufferByteData()
            moreUsefulImgData = npy.ndarray(buffer=imgData,
                                            dtype=npy.uint8,
                                            shape=(f.height, f.width, 1))

            rgb = cv2.cvtColor(moreUsefulImgData, cv2.COLOR_BAYER_RG2RGB)
            c.endCapture()

            return Image(rgb, colorSpace=colorSpace, cv2image=imgData)

        except Exception, e:
            print("Exception acquiring frame: "
                  "{}: {}".format(e, traceback.format_exc()))
            raise e


class VimbaCameraThread(threading.Thread):
    vimba_cam = None
    run = True
    verbose = False
    lock = None
    logger = None
    framerate = 0

    def __init__(self, camera):
        super(VimbaCameraThread, self).__init__()
        self._stop = threading.Event()
        self.vimba_cam = camera
        self.lock = threading.Lock()
        self.name = 'Thread-Camera-ID-' + str(self.camera.uniqueid)

    def run(self):
        counter = 0
        timestamp = time.time()

        while self.run:
            self.lock.acquire()

            img = self.vimba_cam.capture_frame(1000)
            self.vimba_cam.buffer.appendleft(img)

            self.lock.release()
            counter += 1
            time.sleep(0.01)

            if time.time() - timestamp >= 1:
                self.vimba_cam.framerate = counter
                counter = 0
                timestamp = time.time()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()
