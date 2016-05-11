#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import unicode_literals, absolute_import
from PhloxAR.base import *
from PhloxAR.color import *
from PhloxAR.linescan import *

try:
    import urllib2
except ImportError:
    pass
else:
    import urllib

import cv2

from PhloxAR.exif import *


if not init_options_handler.headless:
    import pygame as pg


import scipy.ndimage as ndimage
import scipy.stats.stats as sci_stats
import scipy.cluster.vq as sci_cvq
import scipy.linalg as linalg
import copy


# used for ENUMs
class ColorSpace(object):
    UNKNOWN = 0
    RGB = 1
    BGR = 2
    GRAY = 3
    HLS = 4
    HSV = 5
    XYZ = 6
    YCrCb = 7


class Image(object):
    """
    The Image class allows you to convert to and from a number of source types
    with ease. It also has intelligent buffer management, so that modified
    copies of the Image required for algorithms such as edge detection, etc can
    be cached an reused when appropriate.

    Image are converted into 8-bit, 3-channel images in RGB color space. It
    will automatically handle conversion from other representations into this
    standard format. If dimensions are passed, an empty image is created.
    """
    width = 0
    height = 0
    depth = 0
    filename = ''
    filehandle = ''
    camera = ''

    _layers = []
    _do_hue_palette = False
    _palette_bins = None
    _palette = None
    _palette_members = None
    _palette_percentages = None

    _barcode_reader = ''  # property for the ZXing barcode reader

    # these are buffer frames for various operations on the image
    _bitmap = ''  # the bitmap (iplimage) representation of the image
    _matrix = ''  # the matrix (cvmat) representation
    _pilimg = ''  # holds a PIL Image object in buffer
    _surface = ''  # pygame surface representation of the image
    _narray = ''  # numpy array representation of the image
    _cv2narray = None  # Numpy array which compatible with OpenCV >= 2.3

    _gray_matrix = ''  # the gray scale (cvmat) representation
    _gray_bitmap = ''  # the 8-bit gray scale bitmap
    _gray_narray = ''  # gray scale numpy array for key point stuff
    _gray_cv2narray = None  # grayscale numpy array for OpenCV >= 2.3

    _equalized_gray_bitmap = ''  # the normalized bitmap

    _blob_label = ''  # the label image for blobbing
    _edge_map = ''  # holding reference for edge map
    _canny_param = ''  # parameters that created _edge_map
    _color_space = ColorSpace.UNKNOWN
    _grid_layer = [None, [0, 0]]

    # for DFT caching
    _dft = []  # an array of 2 channel (real, imaginary) 64f images

    # keypoint caching values
    _key_points = None
    _kp_descriptors = None
    _kp_flavor = 'None'

    # temp files
    _tmp_files = []

    # when we empty the buffers, populate with this:
    _initialized_buffers = {
        '_bit_map': '',
        '_matrix': '',
        '_gray_matrix': '',
        '_equalized_gray_bitmap': '',
        '_blob_label': '',
        '_edge_map': '',
        '_canny_param': (0, 0),
        '_pilimage': '',
        '_narray': '',
        '_gray_numpy': '',
        '_pygame_surface': '',
        '_cv2gray_numpy': '',
        'cv2numpy': ''
    }

    # used to buffer the points when we crop the image.
    _uncropped_x = 0
    _uncropped_y = 0

    # initialize the frame
    # TODO: handle camera/capture from file cases (detect on file extension)
    def __int__(self, src=None, camera=None, color_space=ColorSpace.UNKNOWN,
                verbose=True, sample=False, cv2image=False, webp=False):
        """
        Takes a single polymorphic parameter, tests to see how it should convert
        to RGB image.
        :param src: the source of the image, could be anything, a file name,
                        a width and height tuple, a url. Certain strings such as
                        'lena', or 'logo' are loaded automatically for quick test.
        :param camera: a camera to pull a live image
        :param color_space: default camera color space
        :param verbose:
        :param sample: set True, if you want to load som of the included sample
                        images without having to specify complete path
        :param cv2image:
        :param webp:

        Note:
        OpenCV: iplImage and cvMat types
        Python Image Library: Image type
        Filename: All OpenCV supported types (jpg, png, bmp, gif, etc)
        URL: The source can be a url, but must include the http://
        :return:
        """
        self._layers = []
        self.camera = camera
        self._color_space = color_space
        # keypoint descriptors
        self._key_points = []
        self._kp_descriptors = []
        self._kp_flavor = 'None'
        # palette stuff
        self._do_hue_palette = False
        self._palette_bins = None
        self._palette = None
        self._palette_members = None
        self._palette_percentages = None
        # temp files
        self._tmp_files = []

        # check url
        if type(src) == str and (src[:7].lower() == 'http://' or src[:8].lower() == 'https://'):
            req = urllib2.Request(src, headers={
                'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_4) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.54 Safari/536.5"
            })
            img_file = urllib2.urlopen(req)

            img = StringIO(img_file.read())
            src = pilImage.open(img).convert("RGB")

        # check base64 url
        if type(src) == str and (src.lower().startswith('data:image/png;base64')):
            ims = src[22:].decode('base64')
            img = StringIO(ims)
            src = pilImage.open(img).convert("RGB")

        if type(src) == str:
            tmp_name = src.lower()
            if tmp_name == 'lena':
                imgpth = os.path.join(LAUNCH_PATH, 'sample_images', 'lena.jpg')
                src = imgpth
            elif sample:
                imgpth = os.path.join(LAUNCH_PATH, 'sample_images', src)
                src = imgpth

        if type(src) == tuple:
            w = int(src[0])
            h = int(src[1])
            src = cv.CreateImage((w, h), cv.IPL_DEPTH_8U, 3)
            cv.Zero(src)

        scls = src.__class__
        sclsbase = scls.__base__
        sclsname = scls.__name__

        if type(src) == cv.cvmat:
            self._matrix = cv.CreateMat(src.rows, src.cols, cv.CV_8UC3)
            if src.step // src.cols == 3:
                cv.Copy(src, self._matrix, None)
                self._color_space = ColorSpace.BGR
            elif src.step // src.cols == 1:
                cv.Merge(src, src, src, None, self._matrix)
                self._color_space = ColorSpace.GRAY
            else:
                self._color_space = ColorSpace.UNKNOWN
                warnings.warn("Unable to process the provided cvmat.")
        elif type(src) == npy.ndarray:  # handle a numpy array conversion
            if type(src[0, 0]) == npy.ndarray:  # a 3 channel array
                src = src.astype(npy.uint8)
                self._narray = src
                # if the numpy array is not from cv2, then it must be transposed
                if not cv2image:
                    inv_src = src[:, :, ::-1].transpose([1, 0, 2])
                else:
                    inv_src = src

                self._bitmap = cv.CreateImageHeader((inv_src.shape[1],
                                                     inv_src.shape[0]),
                                                    cv.IPL_DEPTH_8U, 3)
                cv.SetData(self._bitmap, inv_src.tostring(),
                           inv_src.dtype.itemsize * 3 * inv_src.shape[1])
                self._color_space = ColorSpace.BGR
            else:
                # we have a single channel array, convert to an RGB iplimage
                src = src.astype(npy.uint8)
                if not cv2image:
                    # we expect width/height but use col/row
                    src = src.transpose([1, 0])
                size = (src.shape[1], src.shape[0])
                self._bitmap = cv.CreateImage(size, cv.IPL_DEPTH_8U, 3)
                channel = cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, 1)
                # initialize an empty channel bitmap
                cv.SetData(channel, src.tostring(),
                           src.dtype.itemsize * src.shape[1])
                cv.Merge(channel, channel, channel, None, self._bitmap)
                self._color_space = ColorSpace.BGR
        elif type(src) == cv.iplimage:
            if src.nChannels == 1:
                self._bitmap = cv.CreateImage(cv.GetSize(src), src.depth, 3)
                cv.Merge(src, src, src, None, self._bitmap)
                self._color_space = ColorSpace.GRAY
            else:
                self._bitmap = cv.CreateImage(cv.GetSize(src), src.depth, 3)
                cv.Copy(src, self._bitmap, None)
                self._color_space = ColorSpace.BGR
        elif type(src) == str or sclsname == 'StringIO':
            if src == '':
                raise IOError("No filename provided to Image constructor")
            elif webp or src.split('.')[-1] == 'webp':
                try:
                    if sclsname == 'StringIO':
                        src.seek(0)  # set the stringIO to the beginning
                    self._pilimage = pilImage.open(src)
                    self._bitmap = cv.CreateImageHeader(self._pilimage.size,
                                                        cv.IPL_DEPTH_8U, 3)
                except:
                    try:
                        # WebM video
                        from webm import decode as wdecode
                    except ImportError:
                        logger.warning('The webm module or latest PIL/PILLOW '
                                       'module needs to be installed to load '
                                       'webp files')
                        return

                    WEBP_IMAGE_DATA = bytearray(file(src, "rb").read())
                    result = wdecode.DecodeRGB(WEBP_IMAGE_DATA)
                    webpimage = pilImage.frombuffer("RGB",
                                                    (result.width, result.height),
                                                    str(result.bitmap),
                                                    "raw", "RGB", 0, 1)
                    self._pilimage = webpimage.convert("RGB")
                    self._bitmap = cv.CreateImageHeader(self._pilimage.size,
                                                        cv.IPL_DEPTH_8U, 3)
                    self.filename = src
                cv.SetData(self._bitmap, self._pilimage.tostring())
                cv.CvtColor(self._bitmap, self._bitmap, cv.CV_RGB2BGR)
            else:
                self.filename = src
                try:
                    self._bitmap = cv.LoadImage(self.filename,
                                                iscolor=cv.CV_LOAD_IMAGE_COLOR)
                except:
                    self._pil = pilImage.open(self.filename).convert("RGB")
                    self._bitmap = cv.CreateImageHeader(self._pil.size,
                                                        cv.IPL_DEPTH_8U, 3)
                    cv.SetData(self._bitmap, self._pil.tostring())
                    cv.CvtColor(self._bitmap, self._bitmap, cv.CV_RGB2BGR)

                # TODO, on IOError fail back to PIL
                self._colorSpace = ColorSpace.BGR
        elif type(src) == pg.Surface:
            self._surface = src
            self._bitmap = cv.CreateImageHeader(self._surface.get_size(),
                                                cv.IPL_DEPTH_8U, 3)
            cv.SetData(self._bitmap, pg.image.tostring(self._surface, 'RGB'))
            cv.CvtColor(self._bitmap, self._bitmap, cv.CV_RGB2BGR)
            self._color_space = ColorSpace

        elif (PIL_ENABLED and ((len(sclsbase) and sclsbase[0].__name__ == "ImageFile") or sclsname == "JpegImageFile" or sclsname == "WebPPImageFile" or sclsname == "Image")):
            if src.mode != 'RGB':
                src = src.convert('RGB')
            self._pilimage = src
            # from OpenCV cookbook
            # http://opencv.willowgarage.com/documentation/python/cookbook.html
            self._bitmap = cv.CreateImageHeader(self._pilimage.size,
                                                cv.IPL_DEPTH_8U, 3)
            cv.SetData(self._bitmap, self._pilimage.tostring())
            self._color_space = ColorSpace.BGR
            cv.CvtColor(self._bitmap, self._bitmap, cv.CV_RGB2BGR)
        else:
            return None

        if color_space != ColorSpace.UNKNOWN:
            self._color_space = color_space

        bmp = self.get_bitmap()
        self.width = bmp.width
        self.height = bmp.height
        self.depth = bmp.depth

    def __del__(self):
        try:
            for i in self._tmp_files:
                if i[1]:
                    os.remove(i[0])
        except:
            pass

    def __repr__(self):
        if len(self.filename) == 0:
            f = 'None'
        else:
            f = self.filename
        return "<PhloxAR.Image Object size: {}, name: {}, " \
               "at memory location: {}>".format((self.width, self.height),
                                                f, hex(id(self)))

    @property
    def exif_data(self):
        """
        Extracts the exif data from an image file. The data is
        returned as a dict.
        :return: a dict of key value pairs.

        Note:
        see: http://en.wikipedia.org/wiki/Exchangeable_image_file_format
        """
        import os, string
        if len(self.filename) < 5 or self.filename is None:
            return {}

        fname, fext = os.path.splitext(self.filename)
        fext = string.lower(fext)

        if fext not in ('.jpeg', '.jpg', '.tiff', '.tif'):
            return {}

        raw = open(self.filename, 'rb')
        data = process_file(raw)
        return data

    # TODO: more functions, need a display first.
    def live(self):
        """
        A live view of the camera.
        Left click will show mouse coordinates and color.
        Right click will kill the live image.
        :return: None.
        """
        start_time = time.time()
        pass

    @property
    def color_space(self):
        """
        Returns Image's color space.
        :return: integer corresponding to the color space.
        """
        return self._color_space

    def is_rgb(self):
        """
        Returns true if the image uses the RGB color space.
        :return: Bool
        """
        return self._color_space == ColorSpace.RGB

    def is_bgr(self):
        """
        Returns true if the image uses the BGR color space.
        :return: Bool
        """
        return self._color_space == ColorSpace.BGR

    def is_hsv(self):
        """
        Returns true if the image uses the HSV color space.
        :return: Bool
        """
        return self._color_space == ColorSpace.HSV

    def is_hls(self):
        """
        Returns true if the image uses the HLS color space.
        :return: Bool
        """
        return self._color_space == ColorSpace.HLS

    def is_xyz(self):
        """
        Returns true if the image uses the XYZ color space.
        :return: Bool
        """
        return self._color_space == ColorSpace.XYZ

    def is_gray(self):
        """
        Returns true if the image uses the grayscale color space.
        :return: Bool
        """
        return self._color_space == ColorSpace.GRAY

    def is_ycrcb(self):
        """
        Returns true if the image uses the YCrCb color space.
        :return: Bool
        """
        return self._color_space == ColorSpace.YCrCb
        pass

    def to_rgb(self):
        """
        Convert the image to RGB color space.
        :return: image in RGB
        """
        img = self.zeros()

        if self.is_bgr() or self._color_space == ColorSpace.UNKNOWN:
            cv.CvtColor(self.bitmap, img, cv.CV_BGR2RGB)
        elif self.is_hsv():
            cv.CvtColor(self.bitmap, img, cv.CV_HSV2RGB)
        elif self.is_hls():
            cv.CvtColor(self.bitmap, img, cv.CV_HLS2RGB)
        elif self.is_xyz():
            cv.CvtColor(self.bitmap, img, cv.CV_XYZ2RGB)
        elif self.is_ycrcb():
            cv.CvtColor(self.bitmap, img, cv.CV_YCrCb2RGB)
        elif self.is_rgb():
            img = self.bitmap
        else:
            logger.warning("Image.to_rgb: conversion no supported.")

        return Image(img, color_space=ColorSpace.RGB)

    def to_bgr(self):
        """
        Convert the image to BGR color space.
        :return: image in BGR
        """
        img = self.zeros()

        if self.is_rgb() or self._color_space == ColorSpace.UNKNOWN:
            cv.CvtColor(self.bitmap, img, cv.CV_RGB2BGR)
        elif self.is_hsv():
            cv.CvtColor(self.bitmap, img, cv.CV_HSV2BGR)
        elif self.is_hls():
            cv.CvtColor(self.bitmap, img, cv.CV_HLS2BGR)
        elif self.is_xyz():
            cv.CvtColor(self.bitmap, img, cv.CV_XYZ2BGR)
        elif self.is_ycrcb():
            cv.CvtColor(self.bitmap, img, cv.CV_YCrCb2BGR)
        elif self.is_bgr():
            img = self.bitmap
        else:
            logger.warning("Image.to_bgr: conversion no supported.")

        return Image(img, color_space=ColorSpace.BGR)

    def to_hls(self):
        """
            Convert the image to HLS color space.
            :return: image in HLS
            """
        img = img_tmp = self.zeros()

        if self.is_rgb() or self._color_space == ColorSpace.UNKNOWN:
            cv.CvtColor(self.bitmap, img, cv.CV_RGB2HLS)
        elif self.is_bgr():
            cv.CvtColor(self.bitmap, img, cv.CV_BGR2HLS)
        elif self.is_hsv():
            cv.CvtColor(self.bitmap, img_tmp, cv.CV_HSV2RGB)
            cv.CvtColor(img_tmp, img, cv.CV_RGB2HLS)
        elif self.is_xyz():
            cv.CvtColor(self.bitmap, img_tmp, cv.CV_XYZ2RGB)
            cv.CvtColor(img_tmp, img, cv.CV_RGB2HLS)
        elif self.is_ycrcb():
            cv.CvtColor(self.bitmap, img_tmp, cv.CV_YCrCb2RGB)
            cv.CvtColor(img_tmp, img, cv.CV_RGB2HLS)
        elif self.is_hls():
            img = self.bitmap
        else:
            logger.warning("Image.to_rgb: conversion no supported.")

        return Image(img, color_space=ColorSpace.HLS)

    def to_hsv(self):
        """
            Convert the image to HSV color space.
            :return: image in HSV
            """
        img = img_tmp = self.zeros()

        if self.is_rgb() or self._color_space == ColorSpace.UNKNOWN:
            cv.CvtColor(self.bitmap, img, cv.CV_RGB2HSV)
        elif self.is_bgr():
            cv.CvtColor(self.bitmap, img, cv.CV_BGR2HSV)
        elif self.is_hls():
            cv.CvtColor(self.bitmap, img_tmp, cv.CV_HLS2RGB)
            cv.CvtColor(img_tmp, img, cv.CV_RGB2HSV)
        elif self.is_xyz():
            cv.CvtColor(self.bitmap, img_tmp, cv.CV_XYZ2RGB)
            cv.CvtColor(img_tmp, img, cv.CV_RGB2HSV)
        elif self.is_ycrcb():
            cv.CvtColor(self.bitmap, img_tmp, cv.CV_YCrCb2RGB)
            cv.CvtColor(img_tmp, img, cv.CV_RGB2HSV)
        elif self.is_hsv():
            img = self.bitmap
        else:
            logger.warning("Image.to_rgb: conversion no supported.")

        return Image(img, color_space=ColorSpace.HSV)

    def to_xyz(self):
        """
            Convert the image to XYZ color space.
            :return: image in XYZ
            """
        img = img_tmp = self.zeros()

        if self.is_rgb() or self._color_space == ColorSpace.UNKNOWN:
            cv.CvtColor(self.bitmap, img, cv.CV_RGB2XYZ)
        elif self.is_bgr():
            cv.CvtColor(self.bitmap, img, cv.CV_BGR2XYZ)
        elif self.is_hsv():
            cv.CvtColor(self.bitmap, img_tmp, cv.CV_HSV2RGB)
            cv.CvtColor(img_tmp, img, cv.CV_RGB2XYZ)
        elif self.is_hls():
            cv.CvtColor(self.bitmap, img_tmp, cv.CV_HLS2RGB)
            cv.CvtColor(img_tmp, img, cv.CV_RGB2XYZ)
        elif self.is_ycrcb():
            cv.CvtColor(self.bitmap, img_tmp, cv.CV_YCrCb2RGB)
            cv.CvtColor(img_tmp, img, cv.CV_RGB2XYZ)
        elif self.is_xyz():
            img = self.bitmap
        else:
            logger.warning("Image.to_rgb: conversion no supported.")

        return Image(img, color_space=ColorSpace.XYZ)

    def to_gray(self):
        """
        Convert the image to GRAY color space.
        :return: image in GRAY
        """
        img = img_tmp = self.zeros(1)

        if self.is_rgb() or self._color_space == ColorSpace.UNKNOWN:
            cv.CvtColor(self.bitmap, img, cv.CV_RGB2GRAY)
        elif self.is_bgr():
            cv.CvtColor(self.bitmap, img, cv.CV_BGR2GRAY)
        elif self.is_hls():
            cv.CvtColor(self.bitmap, img_tmp, cv.CV_HLS2RGB)
            cv.CvtColor(img_tmp, img, cv.CV_RGB2GRAY )
        elif self.is_hsv():
            cv.CvtColor(self.bitmap, img_tmp, cv.CV_HSV2RGB)
            cv.CvtColor(img_tmp, img, cv.CV_RGB2GRAY)
        elif self.is_xyz():
            cv.CvtColor(self.bitmap, img_tmp, cv.CV_XYZ2RGB)
            cv.CvtColor(img_tmp, img, cv.CV_RGB2GRAY)
        elif self.is_ycrcb():
            cv.CvtColor(self.bitmap, img_tmp, cv.CV_YCrCb2RGB)
            cv.CvtColor(img_tmp, img, cv.CV_RGB2GRAY)
        elif self.is_gray():
            img = self.bitmap
        else:
            logger.warning("Image.to_rgb: conversion no supported.")

        return Image(img, color_space=ColorSpace.GRAY)

    def to_ycrcb(self):
        """
        Convert the image to RGB color space.
        :return: image in RGB
        """
        img = img_tmp = self.zeros()

        if self.is_rgb() or self._color_space == ColorSpace.UNKNOWN:
            cv.CvtColor(self.bitmap, img, cv.CV_RGB2YCrCb)
        elif self.is_bgr():
            cv.CvtColor(self.bitmap, img, cv.CV_BGR2YCrCb)
        elif self.is_hsv():
            cv.CvtColor(self.bitmap, img_tmp, cv.CV_HSV2RGB)
            cv.CvtColor(img_tmp, img, cv.CV_RGB2YCrCb)
        elif self.is_xyz():
            cv.CvtColor(self.bitmap, img_tmp, cv.CV_XYZ2RGB)
            cv.CvtColor(img_tmp, img, cv.CV_RGB2YCrCb)
        elif self.is_hls():
            cv.CvtColor(self.bitmap, img_tmp, cv.CV_HLS2RGB)
            cv.CvtColor(img_tmp, img, cv.CV_RGB2YCrCb)
        elif self.is_ycrcb():
            img = self.bitmap
        else:
            logger.warning("Image.to_rgb: conversion no supported.")

        return Image(img, color_space=ColorSpace.YCrCb)

    def cvt_color(self, color_space=None):
        if color_space == ColorSpace.RGB:
            self.to_rgb()
        elif color_space == ColorSpace.BGR:
            self.to_bgr()
        elif color_space == ColorSpace.GRAY:
            self.to_gray()
        elif color_space == ColorSpace.HLS:
            self.to_hls()
        elif color_space == ColorSpace.HSV:
            self.to_hsv()
        elif color_space == ColorSpace.XYZ:
            self.to_xyz()
        elif color_space == ColorSpace.YCrCb:
            self.to_ycrcb()
        else:
            logger.warning("Image.cvt_color: conversion not supported.")

    def zeros(self, channels=3):
        """
        Create a new empty OpenCV bitmap with the specified number of channels.
        This method basically creates an empty copy of the image. This is handy for
        interfacing with OpenCV functions directly.
        :param channels: the number of channels in the returned OpenCV image.
        :return: a black OpenCV IplImage that matches the width, height, and
                  color depth of the source image.
        """
        bitmap = cv.CreateImage(self.size, cv.IPL_DEPTH_8U, channels)
        cv.SetZero(bitmap)

        return bitmap

    @property
    def bitmap(self):
        """
        Retrieve the bitmap (iplImage) of the Image. This is useful if you want
        to use functions from OpenCV with PhloxAR's image class
        :return: black OpenCV IPlImage from this image.
        Example:
        >>> img = Image('lena')
        >>> raw_img = Image.bitmap
        """
        if self._bitmap:
            return self._bitmap
        elif self._matrix:
            self._bitmap = cv.GetImage(self._matrix)

        return self._bitmap

    @property
    def matrix(self):
        """
        Get the matrix (cvMat) version of the image, required for some
        OpenCV algorithms.
        :return: the OpenCV CvMat version of this image.
        """
        if self._matrix:
            return self._matrix
        else:
            self._matrix = cv.GetMat(self._bitmap)
            return self._matrix

    @property
    def gray_matrix(self):
        """
        Get the grayscale matrix (CvMat) version of the image, required for
        some OpenCV algorithms.
        :return: the OpenCV CvMat version of image.
        """
        if self._gray_matrix:
            return self._gray_matrix
        else:
            self._gray_matrix = cv.GetMat(self._gray_bitmap_func())
            return self._gray_matrix

    @property
    def float_matrix(self):
        """
        Converts the standard int bitmap to a floating point bitmap.
        Handy for OpenCV function.
        :return: the floating point OpenCV CvMat version of this image.
        """
        img = cv.CreateImage((self.width, self.height), cv.IPL_DEPTH_32F, 3)
        cv.Convert(self.bitmap, img)

        return img

    @property
    def pilimg(self):
        """
        Get PIL Image object for use with the Python Image Library.
        Handy for PIL functions.
        :return: PIL Image
        """
        if not PIL_ENABLED:
            return None

        if not self._pilimg:
            pass
        pass

    @property
    def narray(self):
        """
        Get a Numpy array of the image in width x height x RGB dimensions
        :return: the image, converted first to grayscale and then converted
                  to a 3D Numpy array.
        """
        if self._narray != '':
            return self._narray

        self._narray = npy.array(self.matrix)[:, :, ::-1].transpose([1, 0, 2])

        return self._narray

    @property
    def gray_narray(self):
        """
        Return a grayscale numpy array of the image.
        :return: the image, converted first to grayscale and the converted to
                  a 2D numpy array.
        """
        if self._gray_narray != '':
            return self._gray_narray
        else:
            self._gray_narray = uint8(npy.array(cv.GetMat(
                    self._gray_bitmap_func()
            )).transpose())

    @property
    def cv2narray(self):
        """
        Get a Numpy array of the image, compatible with OpenCV >= 2.3
        :return: 3D Numpy array of the image.
        """
        if not isinstance(self._cv2narray, npy.ndarray):
            self._cv2narray = npy.array(self.matrix)

        return self._cv2narray

    @property
    def gray_cv2narray(self):
        """
        Get a grayscale Numpy array of the image, compatible with OpenCV >= 2.3
        :return: the 3D Numpy array of the image.
        """
        if not isinstance(self._gray_cv2narray, npy.ndarray):
            self._gray_cv2narray = npy.array(self.gray_matrix)

        return self._gray_cv2narray

    def _gray_bitmap_func(self):
        """
        Gray scaling the image.
        :return: gray scaled image.
        """
        if self._gray_bitmap:
            return self._gray_bitmap

        self._gray_bitmap = self.zeros(1)
        tmp = self.zeros(3)

        if (self._color_space == ColorSpace.BGR or
            self._color_space == ColorSpace.UNKNOWN):
            cv.CvtColor(self.bitmap, self._gray_bitmap, cv.CV_BGR2GRAY)
        elif self._color_space == ColorSpace.RGB:
            cv.CvtColor(self.bitmap, self._gray_bitmap, cv.CV_RGB2GRAY)
        elif self._color_space == ColorSpace.HLS:
            cv.CvtColor(self.bitmap, tmp, cv.CV_HLS2RGB)
            cv.CvtColor(tmp, self._gray_bitmap, cv.CV_RGB2GRAY)
        elif self._color_space == ColorSpace.HSV:
            cv.CvtColor(self.bitmap, tmp, cv.CV_HSV2RGB)
            cv.CvtColor(tmp, self._gray_bitmap, cv.CV_RGB2GRAY)
        elif self._color_space == ColorSpace.XYZ:
            cv.CvtColor(self.bitmap, tmp, cv.CV_XYZ2RGB)
            cv.CvtColor(tmp, self._gray_bitmap, cv.CV_RGB2GRAY)
        elif self._color_space == ColorSpace.GRAY:
            cv.Split(self.bitmap, self._gray_bitmap, self._gray_bitmap,
                     self._gray_bitmap, None)
        else:
            logger.warning("Image._gray_bitmap: There is no supported conversion"
                           "to gray colorspace.")

        return self._gray_bitmap

    def equalize(self):
        """
        Perform histogram equalization on the image.
        :return: return a grayscale Image
        """
        return Image(self._equalize_gray_bitmap())

    def _equalize_gray_bitmap(self):
        """
        Perform histogram equalization on gray scale bitmap
        :return: equalized gracyscale bitmap
        """
        if self._equalized_gray_bitmap:
            return self._equalized_gray_bitmap

        self._equalized_gray_bitmap = self.zeros(1)
        cv.EqualizeHist(self._gray_bitmap_func(), self._equalized_gray_bitmap)

        return self._equalized_gray_bitmap

    @property
    def surface(self):
        """
        Returns the image as a Pygame Surface. Used for rendering the display.
        :return: a pygame Surface object used for rendering.
        """
        if self._surface:
            return self._surface
        else:
            if self.is_gray():
                self._surface = pg.image.fromstring(self.bitmap.tostring(),
                                                    self.size, 'RGB')
            else:
                self._surface = pg.image.fromstring(self.to_rgb().bitmap.tostring(),
                                                    self.size, 'RGB')
            return self._surface

    def to_string(self):
        """
        Returns the image as a string, useful for moving data around.
        :return: the image, converted to rgb, the converted to a string.
        """
        return self.to_rgb().bitmap.tostring()

    def save(self, handle_or_name='', mode='', verbose=False, tmp=False,
             path=None, clean=False, **kwargs):
        pass

    def copy(self):
        pass


    def scale(self, width, height=-1, interpolation=cv2.INTER_LINEAR):
        pass

    def resize(self, width=None, height=None):
        pass

    def smooth(self, method='gaussian', aperture=(3, 3), sigma=0,
               spatial_sigma=0, grayscale=False, aperature=None):
        pass

    def median_filter(self, window='', grayscale=False):
        pass

    def bilateral_filter(self, diameter=5, sigma_color=10, sigma_space=10,
                         grayscale=False):
        pass

    def blur(self, window='', grayscale=False):
        pass

    def gaussian_blur(self, window='', sigmax=0, sigmay=0, grayscale=False):
        pass

    def invert(self):
        pass

    def grayscale(self):
        pass

    def flip_horizontal(self):
        pass

    def flip_vertical(self):
        pass

    def stretch(self, threshold_low=0, threshold_high=255):
        pass

    def gamma_correct(self, gamma=1):
        pass

    def binarize(self, threshold=-1, maxv=255, blocksize=0, p=5):
        pass

    def mean_color(self, color_space=None):
        pass

    def find_corners(self, maxnum=50, minquality=0.04, mindistance=1.0):
        pass

    def find_blobs(self, threshold=-1, minsize=10, maxsize=0,
                   threshold_blocksize=0,
                   threshold_constant=5, appx_level=3):
        pass

    def find_skintone_blobs(self, minsize=10, maxsiz=0, dilate_iter=1):
        pass

    def get_skintone_mask(self, dilate_iter=0):
        pass

    def find_haar_features(self, cascade, scale_factor=1.2, min_neighbors=2,
                           use_cammy=cv.CV_HAAR_DO_CANNY_PRUNING,
                           min_size=(20, 20), max_size=(1000, 1000)):
        pass

    def draw_circle(self, ctr, rad, color=(0, 0, 0), thickness=1):
        pass

    def draw_line(self, pt1, pt2, color=(0, 0, 0), thickness=1):
        pass

    @property
    def size(self):
        """
        Returns a tuple of image's width and height
        :return: tuple
        """
        if self.width and self.height:
            return cv.GetSize(self.bitmap)
        else:
            return 0, 0

    def is_empty(self):
        """
        Check if the image is empty.
        :return: Bool
        """
        return self.size == (0, 0)

    def split(self, cols, rows):
        """
        Break an image into a series of image chunks. Given number of cols
        ans row, splits the image into a cols x rows 2D array.
        :param cols: integer number of rows
        :param rows: integer number of cols
        :return: a list of Images
        """
        crops = []
        w_ratio = self.width / cols
        h_ratio = self.height / rows

        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(self.crop(j * w_ratio, i * h_ratio, w_ratio, h_ratio))
            crops.append(row)

        return crops

    def split_channels(self, grayscale=True):
        pass

    def merge_channels(self, r=None, g=None, b=None):
        pass

    def apply_hls_curve(self, hcurve, lcurve, scurve):
        pass

    def apply_rgb_curve(self, rcurve, gcurve, bcurve):
        pass

    def apply_intensity_curve(self, curve):
        pass

    def color_distance(self, color=Color.BLACK):
        pass

    def hue_distance(self, color=Color.BLACK, minsaturation=20, minvalue=20,
                     maxvalue=255):
        pass

    def erode(self, iterations=1, kernelsize=3):
        pass

    def dilate(self, iteration=1):
        pass

    def morph_open(self):
        pass

    def morph_close(self):
        pass

    def morph_gradient(self):
        pass

    def histogram(self, bins=50):
        """
        Return a numpy array of the 1D histogram of intensity for pixels in the
        image.
        :param bins: integer number of bins in a histogram
        :return: a list of histogram bin values
        """
        gray = self._gray_bitmap_func()

        hist, bin_edges = npy.histogram(npy.asarray(cv.GetMat(gray)), bins=bins)

        return hist.tolist()

    def hue_histogram(self, bins=179, dynamic_range=True):
        """
        Returns the histogram of the hue channel for the image.
        :param bins: integer number of bins in a histogram
        :param dynamic_range:
        :return:
        """
        if dynamic_range:
            return npy.histogram(self.to_hsv().narray[:, :, 2], bins=bins)[0]
        else:
            return npy.histogram(self.to_hsv().narray[:, :, 2], bins=bins,
                                 range=(0.0, 360.0))[0]

    def hue_peaks(self, bins=179):
        pass

    def __getitem__(self, coord):
        pass

    def __setitem__(self, key, value):
        pass

    def __sub__(self, other):
        pass

    def __add__(self, other):
        pass

    def __and__(self, other):
        pass

    def __or__(self, other):
        pass

    def __div__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __pow__(self, power, modulo=None):
        pass

    def __neg__(self):
        pass

    def __invert__(self):
        return self.invert()

    def max(self, other):
        pass

    def min(self, other):
        pass

    def _clear_buffers(self, clearexcept='_bitmap'):
        pass

    def find_barcode(self, dozlib=True, zxing_path=''):
        pass

    def find_lines(self, threshold=80, minlinlen=30, maxlinegap=10, cannyth1=50,
                   cannyth2=100, standard=False, nlines=-1, maxpixelgap=1):
        pass

    def find_chessboard(self, dimensions=(8, 5), subpixel=True):
        pass

    def edges(self, t1=50, t2=100):
        pass

    def _get_edge_map(self, t1=50, t2=100):
        pass

    def roate(self, angle, fixed=True, point=None, scale=1.0):
        if point is None:
            point = [-1, -1]

    def transpose(self):
        pass

    def shear(self, cornerpoints):
        pass

    def transform_affine(self, rot_matrix):
        pass

    def warp(self, cornerpoints):
        pass

    def transfrom_perspective(self, rot_matrix):
        pass

    def get_pixel(self, x, y):
        pass

    def get_gray_pixel(self, x, y):
        pass

    def get_vert_scanline(self, col):
        pass

    def get_horz_scanline(self, row):
        pass

    def get_vert_scaline_gray(self, col):
        pass

    def get_horz_scaline_gray(self, row):
        pass

    def crop(self, x=None, y=None, w=None, h=None, centered=False, smart=False):
        pass

    def region_select(self, x1, y1, x2, y2):
        pass

    def clear(self):
        pass

    def draw(self, features, color=Color.GRAEEN, width=1, autocolor=False):
        pass

    def draw_text(self, text='', x=None, y=None, color=Color.BLUE, fontsize=16):
        pass

    def draw_rect(self, x, y, w, h, color=Color.RED, width=1, alpha=255):
        pass

    def draw_rotated_rect(self, bbox, color=Color.RED, widht=1):
        pass

    def show(self, type='window'):
        pass

    def _surface2image(self, surface):
        pass

    def _image2surface(self, image):
        pass

    def to_pygame_surface(self):
        pass

    def add_drawing_layer(self, layer=None):
        pass

    def insert_drawing_layer(self, layer, index):
        pass

    def remove_drawing_layer(self, index=-1):
        pass

    def get_drawing_layer(self, index=-1):
        pass

    def dl(self, index=-1):
        pass

    def clear_layers(self):
        pass

    def layers(self):
        pass

    def _render_layers(self):
        pass

    def merged_layers(self):
        pass

    def apply_layers(self, indicies=-1):
        pass

    def adaptive_scale(self, resolution, fit=True):
        pass

    def blit(self, img, pos=None, alpha=None, mask=None, alpha_mask=None):
        pass

    def side_by_side(self, img, size='right', scale=True):
        pass

    def embiggen(self, size=None, color=Color.BLACK, pos=None):
        pass

    def _rect_overlap_ROIs(self, top, bottom, pos):
        pass

    def create_binary_mask(self, color1=(0, 0, 0), color2=(255, 255, 255)):
        pass

    def apply_binary_mask(self, mask, bgcolor=Color.BLACK):
        pass

    def create_alpha_mask(self, hue=60, hue_lb=None, hue_ub=None):
        pass

    def apply_pixel_function(self, func):
        pass

    def integral_image(self, titled=False):
        pass

    def convolve(self, kernel=npy.eye(3), center=None):
        pass

    def find_template(self, template=None, threshold=5, method='SQR_DIFF_NORM',
                      grayscale=True, rawmatches=False):
        pass

    def find_template_once(self, template=None, threshold=0.2, method='SQR_DIFF_NORM',
                           grayscale=True):
        pass

    def read_text(self):
        pass

    def find_circle(self, canny=100, threshold=350, distance=-1):
        pass

    def white_balance(self, method='simple'):
        pass

    def apply_lut(self, rlut=None, blut=None, glut=None):
        pass

    def _get_raw_keypoints(self, threshold=500.00, flavor='SURF', highQuality=1,
                           force_reset=False):
        pass

    def _get_FLANN_matches(self, sd, td):
        pass

    def draw_keypoints_matches(self, template, threshold=500.00, min_dist=0.15,
                               width=1):
        pass

    def find_keypoint_match(self, template, quality=500.00, min_dist=0.2,
                            min_match=0.4):
        pass

    def find_keypoints(self, min_quality=300.00, flavor='SURF', high_quality=False):
        pass

    def find_motion(self, previous_frame, window=11, method='BM', aggregate=True):
        pass

    def _generate_palette(self, bins, hue, centroids=None):
        pass

    def get_palette(self, bins=10, hue=False, centeroids=None):
        pass

    def re_palette(self, palette, hue=False):
        pass

    def draw_palette_colors(self, size=(-1, -1), horizontal=True, bins=10, hue=False):
        pass

    def palettize(self, bins=10, hue=False, centroids=None):
        pass

    def find_blobs_from_palette(self, palette_selection, dilate=0, minsize=5,
                                maxsize=0, appx_level=3):
        pass

    def binarize_from_palette(self, palette_selection):
        pass

    def skeletonize(self, radius=5):
        pass

    def smart_threshold(self, mask=None, rect=None):
        pass

    def smart_find_blobs(self, mask=None, rect=None, thresh_level=2, appx_level=3):
        pass

    def threshold(self, value):
        pass

    def flood_fill(self, points, tolerance=None, color=Color.WHITE, lower=None,
                   upper=None, fixed_range=True):
        pass

    def flood_fill_to_mask(self, points, tolerance=None, color=Color.WHITE,
                           lower=None, upper=None, fixed_range=True, mask=None):
        pass

    def find_blobs_from_mask(self, mask, threshold=128, minsize=10, maxsize=0,
                             appx_level=3):
        pass

    def find_flood_fill_blobs(self, points, tolerance=None, lower=None, upper=None,
                              fixed_range=True, minsize=30, maxsize=-1):
        pass

    def _do_DFT(self, grayscale=False):
        pass

    def _get_DFT_clone(self, grayscale=False):
        pass

    def raw_DFT_image(self, grayscale=False):
        pass

    def get_DFT_log_magnitude(self, grayscale=False):
        pass

    def _bounds_from_percentage(self, floatVal, bound):
        pass

    def apply_DFT_filter(self, flt, grayscale=False):
        pass

    def _bounds_from_percentage(self, floatVal, bound):
        pass

    def high_pass_filter(self, xCutoff, yCutoff=None, grayscale=False):
        pass

    def low_pass_filter(self, xCutoff, yCutoff=None, grayscale=False):
        pass

    def band_pass_filter(self, xCutoffLow, xCutoffHigh,
                         yCutoffLow=None, yCutoffHigh=None,
                         grayscale=False):
        pass

    def _inverse_DFT(self, input):
        pass

    def inverse_dft(self, raw_dft_image):
        pass

    def apply_butterworth_filter(self, dia=400, order=2, highpass=False, grayscale=False):
        pass

    def apply_gaussian_filter(self, dia=400, highpass=False, grayscale=False):
        pass

    def apply_unsharp_mask(self, boost=1, dia=400, grayscale=False):
        pass

    def list_haar_features(self):
        pass

    def _copy_avg(self, src, dst, roi, levels, levels_f, mode):
        pass

    def pixelize(self, block_size=10, region=None, levels=None, doHue=False):
        pass

    def anonymize(self, block_size=10, features=None, transform=None):
        pass

    def fill_holes(self):
        pass

    def edge_intersections(self, pt0, pt1, width=1, canny1=0, canny2=100):
        pass

    def fit_contour(self, initial_curve, window=(11, 11), params=(0.1, 0.1, 0.1),
                    doAppx=True, appx_level=1):
        pass

    def fit_edge(self, guess, window=10, threshold=128, measurements=5,
                 darktolight=True, lighttodark=True, departurethreshold=1):
        pass

    def get_threshold_crossing(self, pt1, pt2, threshold=128, darktolight=True,
                               lighttodark=True, departurethreshold=1):
        pass

    def get_diagonal_scanline_grey(self, pt1, pt2):
        pass

    def fit_lines(self, guesses, window=10, threshold=128):
        pass

    def fit_line_points(self, guesses, window=(11, 11), samples=20,
                        params=(0.1, 0.1, 0.1)):
        pass

    def draw_points(self, pts, color=Color.RED, sz=3, width=-1):
        pass

    def sobel(self, xorder=1, yorder=1, doGray=True, aperture=5,
              aperature=None):
        pass

    def track(self, method="CAMShift", ts=None, img=None, bb=None, **kwargs):
        pass

    def _to32f(self):
        pass

    def __getstate__(self):
        pass

    def __setstate__(self):
        pass

    def area(self):
        pass

    def _get_header_anim(self):
        pass

    def rotate270(self):
        pass

    def rotate90(self):
        pass

    def rotate180(self):
        pass

    def rotate_left(self):
        pass

    def rotate_right(self):
        pass

    def vertical_histogram(self, bins=10, threshold=128, normalize=False,
                           forPlot=False):
        pass

    def horizontal_histogram(self, bins=10, threshold=128, normalize=False,
                           forPlot=False):
        pass

    def getLineScan(self, x=None, y=None, pt1=None, pt2=None, channel=-1):
        pass

    def setLineScan(self, linescan, x=None, y=None, pt1=None, pt2=None,
                    channel=-1):
        pass

    def replaceLineScan(self, linescan, x=None, y=None, pt1=None, pt2=None,
                        channel=None):
        pass

    def getPixelsOnLine(self, pt1, pt2):
        pass

    def bresenham_line(self, (x, y), (x2, y2)):
        pass

    def uncrop(self, ListofPts):
        pass

    def grid(self, dimensions=(10, 10), color=(0, 0, 0), width=1,
             antialias=True, alpha=-1):
        pass

    def remove_grid(self):
        pass

    def findGridLines(self):
        pass

    def logicalAND(self, img, grayscale=True):
        pass

    def logicalNAND(self, img, grayscale=True):
        pass

    def logicalOR(self, img, grayscale=True):
        pass

    def logicalXOR(self, img, grayscale=True):
        pass

    def matchSIFTKeyPoints(self, template, quality=200):
        pass

    def drawSIFTKeyPointMatch(self, template, distance=200, num=-1, width=1):
        pass

    def stegaEncode(self, message):
        pass

    def stegaDecode(self):
        pass

    def findFeatures(self, method="szeliski", threshold=1000):
        pass

    def watershed(self, mask=None, erode=2, dilate=2, useMyMask=False):
        pass

    def findBlobsFromWatershed(self, mask=None, erode=2, dilate=2,
                               useMyMask=False, invert=False, minsize=20,
                               maxsize=None):
        pass

    def maxValue(self, locations=False):
        pass

    def minValue(self, locations=False):
        pass

    def findKeypointClusters(self, num_of_clusters=5, order='dsc',
                             flavor='surf'):
        pass

    def getFREAKDescriptor(self, flavor="SURF"):
        pass

    def getGrayHistogramCounts(self, bins=255, limit=-1):
        pass

    def grayPeaks(self, bins=255, delta=0, lookahead=15):
        pass

    def tvDenoising(self, gray=False, weight=50, eps=0.0002, max_iter=200,
                    resize=1):
        pass

    def motionBlur(self, intensity=15, direction='NW'):
        pass

    def recognizeFace(self, recognizer=None):
        pass

    def findAndRecognizeFaces(self, recognizer, cascade=None):
        pass

    def channelMixer(self, channel='r', weight=(100, 100, 100)):
        pass

    def prewitt(self):
        pass

    def edgeSnap(self, pointList, step=1):
        pass

    def _edgeSnap2(self, start, end, step):
        pass

    def motionBlur(self, intensity=15, angle=0):
        pass

    def getLightness(self):
        pass

    def getLuminosity(self):
        pass

    def getAverage(self):
        pass

    def smartRotate(self, bins=18, point=[-1, -1], auto=True, threshold=80,
                    minLength=30, maxGap=10, t1=150, t2=200, fixed=True):
        pass

    def normalize(self, newMin=0, newMax=255, minCut=2, maxCut=98):
        pass

    def getNormalizedHueHistogram(self, roi=None):
        pass

    def backProjectHueHistogram(self, model, smooth=True, fullColor=False,
                                threshold=None):
        pass

    def findBlobsFromHueHistogram(self, model, threshold=1, smooth=True,
                                  minsize=10, maxsize=None):
        pass

    def filter(self, flt, grayscale=False):
        pass


class ImageSet(list):
    """
    An abstract class for keeping a list of images.
    """
    file_list = None

    def __init__(self, directory=None):
        if not directory:
            return

        if isinstance(directory, list):
            if isinstance(directory[0], Image):
                super(ImageSet, self).__init__(directory)
            elif isinstance(directory[0], str):
                super(ImageSet, self).__init__(map(Image, directory))
        elif directory.lower() == 'samples' or directory.lower == 'sample':
            path = LAUNCH_PATH
            path = os.path.realpath(path)
            directory = os.path.join(path, 'sample_images')
            self.load(directory)
        else:
            self.load(directory)

    # TODO: implementation
    def load(self, directory):
        pass

# TODO: import