#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import unicode_literals, absolute_import
from PhloxAR.base import *
from PhloxAR.color import *
from PhloxAR.linescan import *
from PhloxAR.features import *

try:
    import urllib2
except ImportError:
    pass
else:
    import urllib

import cv2

from PhloxAR.exif import *


if not init_options_handler.headless:
    import pygame as sdl2


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
    _cvnarray = None  # Numpy array which compatible with OpenCV >= 2.3

    _gray_matrix = ''  # the gray scale (cvmat) representation
    _gray_bitmap = ''  # the 8-bit gray scale bitmap
    _gray_narray = ''  # gray scale numpy array for key point stuff
    _gray_cvnarray = None  # grayscale numpy array for OpenCV >= 2.3

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
            src = PILImage.open(img).convert("RGB")

        # check base64 url
        if type(src) == str and (src.lower().startswith('data:image/png;base64')):
            ims = src[22:].decode('base64')
            img = StringIO(ims)
            src = PILImage.open(img).convert("RGB")

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
                    self._pilimage = PILImage.open(src)
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
                    webpimage = PILImage.frombuffer("RGB",
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
                    self._pil = PILImage.open(self.filename).convert("RGB")
                    self._bitmap = cv.CreateImageHeader(self._pil.size,
                                                        cv.IPL_DEPTH_8U, 3)
                    cv.SetData(self._bitmap, self._pil.tostring())
                    cv.CvtColor(self._bitmap, self._bitmap, cv.CV_RGB2BGR)

                # TODO, on IOError fail back to PIL
                self._color_space = ColorSpace.BGR
        elif type(src) == sdl2.Surface:
            self._surface = src
            self._bitmap = cv.CreateImageHeader(self._surface.get_size(),
                                                cv.IPL_DEPTH_8U, 3)
            cv.SetData(self._bitmap, sdl2.image.tostring(self._surface, 'RGB'))
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

        bmp = self.bitmap
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

        :Example:
        >>> cam = Camera()
        >>> cam.live()
        """
        start_time = time.time()

        from PhloxAR.display import Display

        i = self
        d = Display(i.size)
        i.save(d)
        col = Color.RED

        while not d.is_done():
            i = self
            i.clear_layers()
            elapsed_time = time.time() - start_time

            if d.mouse_l:
                txt1 = 'Coord: ({}, {})'.format(d.mouse_x, d.mouse_r)
                i.dl().text(txt1, (10, i.height / 2), color=col)
                txt2 = 'Color: {}'.format((i.get_pixel(d.mouse_x, d.mouse_y)))
                i.dl().text(txt2, (10, i.height / 2 + 10), color=col)
                print(txt1 + '\n' + txt2)

            if 0 < elapsed_time < 5:
                i.dl().text('In live mode', (10, 10), color=col)
                i.dl().text("Left click will show mouse coordinates and color",
                            (10,20), color=col)
                i.dl().text("Right click will kill the live image", (10,30),
                            color=col)

            i.save(d)
            if d.mouse_r:
                print("Closing Window!")
                d.done = True

    @property
    def color_space(self):
        """
        Returns Image's color space.
        :return: integer corresponding to the color space.
        """
        return self._color_space

    get_color_space = color_space

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

    get_empty = zeros

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

    get_bitmap = bitmap.fget

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

    get_matrix = matrix.fget

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

    get_grayscale_matrix = gray_matrix.fget

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

    get_float_matrix = float_matrix

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

    get_pilimg = pilimg

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

    get_narray = narray.fget

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

    get_grayscale_narray = gray_narray.fget

    @property
    def cvnarray(self):
        """
        Get a Numpy array of the image, compatible with OpenCV >= 2.3
        :return: 3D Numpy array of the image.
        """
        if not isinstance(self._cvnarray, npy.ndarray):
            self._cvnarray = npy.array(self.matrix)

        return self._cvnarray

    get_cvnarray = cvnarray.fget

    @property
    def gray_cvnarray(self):
        """
        Get a grayscale Numpy array of the image, compatible with OpenCV >= 2.3
        :return: the 3D Numpy array of the image.
        """
        if not isinstance(self._gray_cvnarray, npy.ndarray):
            self._gray_cvnarray = npy.array(self.gray_matrix)

        return self._gray_cvnarray

    get_grayscale_cvnarray = gray_cvnarray.fget

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
            logger.warning("Image._gray_bitmap: There is no supported "
                           "conversion to gray color space.")

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
                self._surface = sdl2.image.fromstring(self.bitmap.tostring(),
                                                      self.size, 'RGB')
            else:
                self._surface = sdl2.image.fromstring(self.to_rgb().bitmap.tostring(),
                                                      self.size, 'RGB')
            return self._surface

    get_sdl_surface = surface.fget

    def to_string(self):
        """
        Returns the image as a string, useful for moving data around.
        :return: the image, converted to rgb, the converted to a string.
        """
        return self.to_rgb().bitmap.tostring()

    def save(self, handle_or_name='', mode='', verbose=False, tmp=False,
             path=None, filename=None, clean=False, **kwargs):
        """
        Save the image to the specified file name. If no file name is provided
        then it will use the file name from which the Image was loaded, or the
        last place it was saved to. You can save to lots of places, not just files.
        For example you can save to the Display, a JpegStream, VideoStream,
        temporary file, or Ipython Notebook.

        Save will implicitly render the image's layers before saving, but the
        layers are not applied to the Image itself.

        :param handle_or_name: the filename to which to store the file.
                                The method will infer the file type.
        :param mode: used for saving using pul
        :param verbose: if True return the path where we saved the file.
        :param tmp: if True save the image as a temporary file and return the path
        :param path: where temporary files to be stored
        :param filename: name(prefix) of the temporary file
        :param clean: True if temp files are tobe deleted once the object
                       is to be destroyed
        :param kwargs: used for overloading the PIL save methods. In particular
                        this method is useful for setting the jpeg compression
                        level. For JPG see this documentation:
                        http://www.pythonware.com/library/pil/handbook/format-jpeg.htm

        :return:

        :Example:
        >>> img = Image('phlox')
        >>> img.save(tmp=True)

        It will return the path that it saved to.
        Save also supports IPython Notebooks when passing it a Display object
        that has been instainted with the notebook flag.
        To do this just use:

        >>> disp = Display(displaytype='notebook')
        >>> img.save(disp)

        :Note:
        You must have IPython notebooks installed for this to work path
        and filename are valid if and only if temp is set to True.
        """
        # TODO, we use the term mode here when we mean format
        # TODO, if any params are passed, use PIL
        if tmp:
            import glob
            if filename is None:
                filename = 'Image'
            if path is None:
                path = tempfile.gettempdir()
            if glob.os.path.exists(path):
                path = glob.os.path.abspath(path)
                imfiles = glob.glob(glob.os.path.join(path, filename + '*.png'))
                num = [0]
                for img in imfiles:
                    num.append(int(glob.re.findall('[0-9]+$', img[:-4])[-1]))
                num.sort()
                fnum = num[-1] + 1
                filename = glob.os.path.join(path, filename + ('%07d' % fnum) +
                                             '.png')
                self._tmp_files.append(filename, clean)
                self.save(self._tmp_files[-1][0])
                return self._tmp_files[-1][0]
            else:
                print("Path does not exist!")
        else:
            if filename:
                handle_or_name = filename + '.png'

        if not handle_or_name:
            if self.filename:
                handle_or_name = self.filename
            else:
                handle_or_name = self.filehandle

        if len(self._layers):
            img_save = self.apply_layers()
        else:
            img_save = self

        if (self._color_space != ColorSpace.BGR and
            self._color_space != ColorSpace.GRAY):
            img_save = img_save.to_bgr()

        if not isinstance(handle_or_name, str):
            fh = handle_or_name

            if not PIL_ENABLED:
                logger.warning("You need the Pillow to save by file handle")
                return 0

            if isinstance(fh, 'JpegStreamer'):
                fh.jpgdata = StringIO()
                # save via PIL to a StringIO handle
                img_save.pilimg.save(fh.jpgdata, 'jpeg', **kwargs)
                fh.refreshtime = time.time()
                self.filename = ''
                self.filehandle = fh

            elif isinstance(fh, 'VideoStream'):
                self.filename = ''
                self.filehandle = fh
                fh.write_frame(img_save)

            elif isinstance(fh, 'Display'):
                if fh.display_type == 'notebook':
                    try:
                        from IPython.core import Image as IPYImage
                    except ImportError:
                        print("Need IPython Notebooks to use the display mode!")
                        return

                    from IPython.core import display as IPYDisplay
                    tf = tempfile.NamedTemporaryFile(suffix='.png')
                    loc = tf.name
                    tf.close()
                    self.save(loc)
                    IPYDisplay.display(IPYImage(filename=loc))
                else:
                    self.filehandle = fh
                    fh.write_frame(img_save)

            else:
                if not mode:
                    mode = 'jpeg'

                try:
                    # latest version of PIL/PILLOW supports webp,
                    # try this first, if not gracefully fallback
                    img_save.pilimg.save(fh, mode, **kwargs)
                    # set the file name for future save operations
                    self.filehandle = fh
                    self.filename = ''
                    return 1
                except Exception as e:
                    if mode.lower() != 'webp':
                        raise e

            if verbose:
                print(self.filename)

            if not mode.lower() == 'webp':
                return 1

        # make a temporary file location if there isn't one
        if not handle_or_name:
            filename = tempfile.mkstemp(suffix='.png')[-1]
        else:
            filename = handle_or_name

        # allow saving in webp format
        if mode == 'webp' or re.search('\.webp$', filename):
            try:
                # newer versions of PIL support webp format, try that first
                self.pilimg.save(filename, **kwargs)
            except:
                logger.warning("Can't save to webp format!")

        if kwargs:
            if not mode:
                mode = 'jpeg'
            img_save.pilimg.save(filename, mode, **kwargs)
            return 1

        if filename:
            cv.SaveImage(filename, img_save.bitmap)
            self.filename = filename
            self.filehandle = None
        elif self.filename:
            cv.SaveImage(self.filename, img_save.bitmap)
        else:
            return 0

        if verbose:
            print(self.filename)

        if tmp:
            return filename
        else:
            return 1

    def copy(self):
        """
        Return a full copy of the Image's bitmap.  Note that this is different
        from using python's implicit copy function in that only the bitmap itself
        is copied. This method essentially performs a deep copy.

        :return: a copy of this Image

        :Example:
        >>> img = Image('lena')
        >>> img2 = img.copy()
        """
        img = self.zeros()
        cv.Copy(self.bitmap, img)

        return Image(img, colorspace=self._color_space)

    def scale(self, scalar, interp=cv2.INTER_LINEAR):
        """
        Scale the image to a new width and height.
        If no height is provided, the width is considered a scaling value

        :param scalar: scalar to scale
        :param interp: how to generate new pixels that don't match the original
                       pixels. Argument goes direction to cv.Resize.
                       See http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html?highlight=resize#cv2.resize for more details

        :return: resized image.

        :Example:
        >>> img.scale(2.0)
        """
        if scalar is not None:
            w = int(self.width *  scalar)
            h = int(self.height * scalar)
            if w > MAX_DIMS or h > MAX_DIMS or h < 1 or w < 1:
                logger.warning("You tried to make an image really big or "
                               "impossibly small. I can't scale that")
                return self
        else:
            return self

        scaled_array = npy.zeros((w, h, 3), dtype='uint8')
        ret = cv2.resize(self.cvnarray, (w, h), interpolation=interp)
        return Image(ret, color_space=self._color_space, cv2image=True)

    def resize(self, width=None, height=None):
        """
        Resize an image based on a width, a height, or both.
        If either width or height is not provided the value is inferred by
        keeping the aspect ratio. If both values are provided then the image
        is resized accordingly.

        :param width: width of the output image in pixels.
        :param height: height of the output image in pixels.

        :return:
        Returns a resized image, if the size is invalid a warning is issued and
        None is returned.

        :Example:
        >>> img = Image("lenna")
        >>> img2 = img.resize(w=1024) # h is guessed from w
        >>> img3 = img.resize(h=1024) # w is guessed from h
        >>> img4 = img.resize(w=200,h=100)
        """
        ret = None
        if width is None and height is None:
            logger.warning("Image.resize has no parameters. No operation is "
                           "performed")
            return None
        elif width is not None and height is None:
            sfactor = float(width) / float(self.width)
            height = int(sfactor * float(self.height))
        elif width is None and height is not None:
            sfactor = float(height) / float(self.height)
            width = int(sfactor * float(self.width))

        if width > MAX_DIMS or height > MAX_DIMS:
            logger.warning("Image.resize! You tried to make an image really"
                           " big or impossibly small. I can't scale that")
            return ret

        scaled_bitmap = cv.CreateImage((width, height), 8, 3)
        cv.Resize(self.bitmap, scaled_bitmap)
        return Image(scaled_bitmap, color_space=self._color_space)

    def smooth(self, method='gaussian', aperture=(3, 3), sigma=0,
               spatial_sigma=0, grayscale=False):
        """
        Smooth the image, by default with the Gaussian blur. If desired,
        additional algorithms and apertures can be specified. Optional
        parameters are passed directly to OpenCV's cv.Smooth() function.
        If grayscale is true the smoothing operation is only performed
        on a single channel otherwise the operation is performed on
        each channel of the image. for OpenCV versions >= 2.3.0 it is
        advisible to take a look at
        - bilateralFilter
        - medianFilter
        - blur
        - gaussianBlur

        :param method: valid options are 'blur' or 'gaussian', 'bilateral',
                        and 'median'.
        :param aperture: a tuple for the window size of the gaussian blur as
                          an (x,y) tuple. should be odd
        :param sigma:
        :param spatial_sigma:
        :param grayscale: return grayscale image

        :return: the smoothed image.

        :Example:
        >>> img = Image('lena')
        >>> img2 = img.smooth()
        >>> img3 = img.smooth('median')
        """
        # TODO: deprecated function -istuple-
        if istuple(aperture):
            win_x, win_y = aperture
            if win_x <= 0 or win_y <= 0 or win_x % 2 == 0 or win_y % 2 == 0:
                logger.warning('The size must be odd number and greater than 0')
                return None
        else:
            raise ValueError('Please provide a tuple to aperture')

        if method == 'blur':
            m = cv.CV_BLUR
        elif method == 'bilateral':
            m = cv.CV_BILATERAL
            win_y = win_x  # aperture must be square
        elif method == 'median':
            m = cv.CV_MEDIAN
            win_y = win_x  # aperture must be square
        else:
            m = cv.CV_GAUSSIAN  # default method

        if grayscale:
            new_img = self.zeros(1)
            cv.Smooth(self._gray_bitmap_func(), new_img, method, win_x, win_y,
                      sigma, spatial_sigma)
        else:
            new_img = self.zeros(3)
            r = self.zeros(1)
            g = self.zeros(1)
            b = self.zeros(1)
            ro = self.zeros(1)
            go = self.zeros(1)
            bo = self.zeros(1)
            cv.Split(self.bitmap, b, g, r, None)
            cv.Smooth(r, ro, method, win_x, win_y, sigma, spatial_sigma)
            cv.Smooth(g, go, method, win_x, win_y, sigma, spatial_sigma)
            cv.Smooth(b, bo, method, win_x, win_y, sigma, spatial_sigma)
            cv.Merge(bo, go, ro, None, new_img)

        return Image(new_img, color_space=self._color_space)

    def median_filter(self, window=(3, 3), grayscale=False):
        """
        Smooths the image, with the median filter. Performs a median
        filtering operation to denoise/despeckle the image.
        The optional parameter is the window size.


        :param window: should be in the form a tuple (win_x,win_y).
                       Where win_x should be equal to win_y. By default
                       it is set to 3x3, i.e window = (3, 3).

        :Note:
        win_x and win_y should be greater than zero, a odd number and equal.
        """
        if istuple(window):
            win_x, win_y = window
            if win_x >= 0 and win_y >= 0 and win_x % 2 == 1 and win_y % 2 == 1:
                if win_x != win_y:
                    win_x = win_y
            else:
                logger.warning("The aperture (win_x,win_y) must be odd number"
                               " and greater than 0.")
                return None

        elif isnum(window):
            win_x = window
        else:
            win_x = 3  # set the default aperture window size (3x3)

        if grayscale:
            img_median_blur = cv2.medianBlur(self.gray_narray, win_x)
            return Image(img_median_blur, color_space=ColorSpace.GRAY)
        else:
            img_median_blur = cv2.medianBlur(
                    self.narray[:, :, ::-1].transpose([1, 0, 2]), win_x
            )
            img_median_blur = img_median_blur[:, :, ::-1].transpose([1, 0, 2])
            return Image(img_median_blur, color_space=self._color_space)

    def bilateral_filter(self, diameter=5, sigma_color=10, sigma_space=10,
                         grayscale=False):
        """
        Smooths the image, using bilateral filtering. Potential of bilateral
        filtering is for the removal of texture. The optional parameter are
        diameter, sigmaColor, sigmaSpace.

        Bilateral Filter
        see : http://en.wikipedia.org/wiki/Bilateral_filter
        see : http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html

        :param diameter: a tuple for the window of the form (diameter,diameter).
                         By default window = (3, 3). Diameter of each pixel
                         neighborhood that is used during filtering.
        :param sigma_color: filter the specified value in the color space.
                            A larger value of the parameter means that farther
                            colors within the pixel neighborhood will be mixed
                            together, resulting in larger areas of semi-equal
                            color.
        :param sigma_space: filter the specified value in the coordinate space.
                            A larger value of the parameter means that farther
                            pixels will influence each other as long as their
                            colors are close enough
        """
        if istuple(diameter):
            win_x, win_y = diameter
            if win_x >= 0 and win_y >= 0 and win_x % 2 == 1 and win_y % 2 == 1:
                if win_x != win_y:
                    diameter = (win_x, win_y)
            else :
                logger.warning("The aperture (win_x,win_y) must be odd number and greater than 0.")
                return None
        else :
            win_x = 3  # set the default aperture window size (3x3)
            diameter = (win_x,win_x)

        if grayscale:
            img_bilateral = cv2.bilateralFilter(self.gray_narray, diameter,
                                                sigma_color, sigma_space)
            return Image(img_bilateral, color_space=ColorSpace.GRAY)
        else:
            img_bilateral = cv2.bilateralFilter(
                    self.narray[:, :, ::-1].transpose([1, 0, 2]),
                    diameter, sigma_color, sigma_space
            )
            img_bilateral = img_bilateral[:, :, ::-1].transpose([1, 0, 2])
            return Image(img_bilateral, color_space=self._color_space)

    def blur(self, window=None, grayscale=False):
        """
        Smooth an image using the normalized box filter.
        The optional parameter is window.
        see : http://en.wikipedia.org/wiki/Blur

        :param window: should be in the form a tuple (win_x,win_y).
                       By default it is set to 3x3, i.e window = (3, 3).
        """
        if istuple(window):
            win_x, win_y = window
            if win_x <= 0 or win_y <= 0:
                logger.warning("win_x and win_y should be greater than 0.")
                return None
        elif isnum(window):
            window = (window, window)
        else:
            window = (3, 3)

        if grayscale:
            img_blur = cv2.blur(self.gray_narray, window)
            return Image(img_blur, color_space=ColorSpace.GRAY)
        else:
            img_blur = cv2.blur(self.narray[:, :, ::-1].transpose([1, 0, 2]),
                                window)
            img_blur = img_blur[:, :, ::-1].transpose([1, 0, 2])
            return Image(img_blur, color_space=self._color_space)

    def gaussian_blur(self, window='', sigmax=0, sigmay=0, grayscale=False):
        """
        Smoothes an image, typically used to reduce image noise and reduce detail.
        The optional parameter is window.
        see : http://en.wikipedia.org/wiki/Gaussian_blur

        :param window: should be in the form a tuple (win_x,win_y).
                        Where win_x and win_y should be positive and odd.
                        By default it is set to 3x3, i.e window = (3, 3).
        :param sigmax: Gaussian kernel standard deviation in X direction.
        :param sigmay: Gaussian kernel standard deviation in Y direction.
        :param grayscale: if true, the effect is applied on grayscale images.
        """
        if istuple(window):
            win_x, win_y = window
            if win_x >= 0 and win_y >= 0 and win_x % 2 == 1 and win_y % 2 == 1:
                pass
            else:
                logger.warning("The aperture (win_x,win_y) must be odd number "
                               "and greater than 0.")
                return None

        elif isnum(window):
            window = (window, window)

        else:
            window = (3, 3)  # set the default aperture window size (3x3)

        image_gauss = cv2.GaussianBlur(self.cvnarray, window,
                                       sigmaX=sigmax,
                                       sigmaY=sigmay)

        if grayscale:
            return Image(image_gauss, color_space=ColorSpace.GRAY,
                         cv2image=True)
        else:
            return Image(image_gauss, color_space=self._color_space,
                         cv2image=True)

    def invert(self):
        """
        Invert (negative) the image note that this can also be done with the
        unary minus (-) operator. For binary image this turns black into white and white into black (i.e. white is the new black).

        :return: opposite of the current image.

        :Example:
        >>> img  = Image("polar_bear_in_the_snow.png")
        >>> img.invert().save("black_bear_at_night.png")
        """
        return -self

    def grayscale(self):
        """
        This method returns a gray scale version of the image. It makes
        everything look like an old movie.

        :return: a grayscale image.

        :Example:
        >>> img = Image("lenna")
        >>> img.grayscale().binarize().show()
        """
        return Image(self._gray_bitmap_func(), color_space=ColorSpace.GRAY)

    def flip_horizontal(self):
        """
        Horizontally mirror an image.
        Note that flip does not mean rotate 180 degrees! The two are different.

        :return: flipped image.

        :Example:
        >>> img = Image("lena")
        >>> upsidedown = img.flip_horizontal()
        """
        new_img = self.zeros()
        cv.Flip(self.bitmap, new_img, 1)
        return Image(new_img, color_space=self._color_space)

    def flip_vertical(self):
        """
        Vertically mirror an image.
        Note that flip does not mean rotate 180 degrees! The two are different.

        :return: flipped image.

        :Example:
        >>> img = Image("lena")
        >>> upsidedown = img.flip_vertical()
        """
        new_img = self.zeros()
        cv.Flip(self.bitmap, new_img, 0)
        return Image(new_img, color_space=self._color_space)

    def stretch(self, thresh_low=0, thresh_high=255):
        """
        The stretch filter works on a greyscale image, if the image
        is color, it returns a greyscale image.  The filter works by
        taking in a lower and upper threshold.  Anything below the lower
        threshold is pushed to black (0) and anything above the upper
        threshold is pushed to white (255)

        :param thresh_low: the lower threshold for the stretch operation.
                             This should be a value between 0 and 255.
        :param thresh_high: the upper threshold for the stretch operation.
                             This should be a value between 0 and 255.

        :return: A gray scale version of the image with the appropriate
                  histogram stretching.

        :Example:
        >>> img = Image("orson_welles.jpg")
        >>> img2 = img.stretch(56.200)
        >>> img2.show()

        # TODO - make this work on RGB images with thresholds for each channel.
        """
        try:
            new_img = self.zeros(1)
            cv.Threshold(self._gray_bitmap_func(), new_img, thresh_low, 255,
                         cv.CV_THRESH_TOZERO)
            cv.Not(new_img, new_img)
            cv.Threshold(new_img, new_img, 255 - thresh_high, 255,
                         cv.CV_THRESH_TOZERO)
            cv.Not(new_img, new_img)
            return Image(new_img)
        except:
            return None

    def gamma_correct(self, gamma=1):
        """
        Transforms an image according to Gamma Correction also known as
        Power Law Transform.

        :param gamma: a non-negative real number.
        :return: a Gamma corrected image.

        :Example:
        >>> img = Image('family_watching_television_1958.jpg')
        >>> img.show()
        >>> img.gamma_correct(1.5).show()
        >>> img.gamma_correct(0.7).show()
        """
        if gamma < 0:
            return "Gamma should be a non-negative real number"
        scale = 255.0
        src = self.narray
        dst = (((1.0 / scale) * src) ** gamma) * scale
        return Image(dst)

    def binarize(self, thresh=-1, maxv=255, blocksize=0, p=5):
        """
        Do a binary threshold the image, changing all values below thresh to
        maxv and all above to black.  If a color tuple is provided, each color
        channel is thresholded separately. If threshold is -1 (default), an
        adaptive method (OTSU's method) is used. If then a blocksize is
        specified, a moving average over each region of block*block pixels a
        threshold is applied where threshold = local_mean - p.

        :param thresh: the threshold as an integer or an (r,g,b) tuple,
                        where pixels below (darker) than thresh are set
                        to to max value, and all values above this value
                        are set to black. If this parameter is -1 we use
                        Otsu's method.
        :param maxv: the maximum value for pixels below the threshold.
                     Ordinarily this should be 255 (white)
        :param blocksize: the size of the block used in the adaptive binarize
                           operation. This parameter must be an odd number.
        :param p: the difference from the local mean to use for thresholding
                   in Otsu's method.

        :return: a binary (two colors, usually black and white) image. This
                 works great for the find_blobs family of functions.

        :Example:
        Example of a vanila threshold versus an adaptive threshold:
        >>> img = Image("orson_welles.jpg")
        >>> b1 = img.binarize(128)
        >>> b2 = img.binarize(blocksize=11,p=7)
        >>> b3 = b1.sideBySide(b2)
        >>> b3.show()

        :Note:
        Otsu's Method Description<http://en.wikipedia.org/wiki/Otsu's_method>`
        """
        if istuple(thresh):
            r = self.zeros(1)
            g = self.zeros(1)
            b = self.zeros(1)
            cv.Split(self.bitmap, b, g, r, None)

            cv.Threshold(r, r, thresh[0], maxv, cv.CV_THRESH_BINARY_INV)
            cv.Threshold(g, g, thresh[1], maxv, cv.CV_THRESH_BINARY_INV)
            cv.Threshold(b, b, thresh[2], maxv, cv.CV_THRESH_BINARY_INV)

            cv.Add(r, g, r)
            cv.Add(r, b, r)

            return Image(r, color_space=self._color_space)
        elif thresh == -1:
            new_bitmap = self.zeros(1)
            if blocksize:
                cv.AdaptiveThreshold(self._gray_bitmap_func(), new_bitmap,
                                     maxv,
                                     cv.CV_ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv.CV_THRESH_BINARY_INV, blocksize, p)
            else:
                cv.Threshold(self._gray_bitmap_func(), new_bitmap, thresh,
                             float(maxv),
                             cv.CV_THRESH_BINARY_INV + cv.CV_THRESH_OTSU)
            return Image(new_bitmap, color_space=self._color_space)
        else:
            new_bitmap = self.zeros(1)
            # desaturate the image, and apply the new threshold
            cv.Threshold(self._gray_bitmap_func(), new_bitmap, thresh,
                         float(maxv), cv.CV_THRESH_BINARY_INV)
            return Image(new_bitmap, color_space=self._color_space)

    def mean_color(self, color_space=None):
        """
        This method finds the average color of all the pixels in the image and
        displays tuple in the color space specfied by the user.
        If no color space is specified , (B,G,R) color space is taken as default.

        :return: a tuple of the average image values. Tuples are in the
                 channel order. For most images this means the results
                 are (B,G,R).

        :Example:
        >>> img = Image('lenna')
        >>> colors = img.mean_color()        # returns tuple in Image's colorspace format.
        >>> colors = img.mean_color('BGR')   # returns tuple in (B,G,R) format.
        >>> colors = img.mean_color('RGB')   # returns tuple in (R,G,B) format.
        >>> colors = img.mean_color('HSV')   # returns tuple in (H,S,V) format.
        >>> colors = img.mean_color('XYZ')   # returns tuple in (X,Y,Z) format.
        >>> colors = img.mean_color('Gray')  # returns float of mean intensity.
        >>> colors = img.mean_color('YCrCb') # returns tuple in (Y,Cr,Cb) format.
        >>> colors = img.mean_color('HLS')   # returns tuple in (H,L,S) format.
        """

        if color_space is None:
            return tuple(cv.Avg(self.bitmap())[0:3])
        elif color_space == 'BGR':
            return tuple(cv.Avg(self.to_bgr().bitmap)[0:3])
        elif color_space == 'RGB':
            return tuple(cv.Avg(self.to_rgb().bitmap)[0:3])
        elif color_space == 'HSV':
            return tuple(cv.Avg(self.to_hsv().bitmap)[0:3])
        elif color_space == 'XYZ':
            return tuple(cv.Avg(self.to_xyz().bitmap)[0:3])
        elif color_space == 'Gray':
            return cv.Avg(self._gray_bitmap_func())[0]
        elif color_space == 'YCrCb':
            return tuple(cv.Avg(self.to_ycrcb().bitmap)[0:3])
        elif color_space == 'HLS':
            return tuple(cv.Avg(self.to_hls().bitmap)[0:3])
        else:
            logger.warning("Image.mean_color: There is no supported conversion"
                           " to the specified colorspace. Use one of these as "
                           "argument: 'BGR' , 'RGB' , 'HSV' , 'Gray' , 'XYZ' , "
                           "'YCrCb' , 'HLS' .")
            return None

    def find_corners(self, maxnum=50, minquality=0.04, mindistance=1.0):
        """
        This will find corner Feature objects and return them as a FeatureSet
        strongest corners first.  The parameters give the number of corners
        to look for, the minimum quality of the corner feature, and the minimum
        distance between corners.

        :param maxnum: the maximum number of corners to return.
        :param minquality: the minimum quality metric. This should be a number
                           between zero and one.
        :param mindistance: the minimum distance, in pixels, between successive
                             corners.
        :return: a featureset of Corner features or None if no corners are found.

        :Example:
        >>> img = Image("sampleimages/simplecv.png")
        >>> corners = img.find_corners()
        >>> if corners: True

        >>> img = Image("sampleimages/black.png")
        >>> corners = img.find_corners()
        >>> if not corners: True
        """
        # initialize buffer frames
        eig_image = cv.CreateImage(cv.GetSize(self.bitmap),
                                   cv.IPL_DEPTH_32F, 1)
        temp_image = cv.CreateImage(cv.GetSize(self.bitmap),
                                    cv.IPL_DEPTH_32F, 1)

        corner_coordinates = cv.GoodFeaturesToTrack(self._gray_bitmap_func(),
                                                    eig_image, temp_image,
                                                    maxnum, minquality,
                                                    mindistance, None)
        corner_features = []
        for (x, y) in corner_coordinates:
            corner_features.append(Corner(self, x, y))

        return FeatureSet(corner_features)

    def find_blobs(self, threshold=-1, minsize=10, maxsize=0,
                   threshold_blocksize=0,
                   threshold_constant=5, appx_level=3):
        """
        Find blobs  will look for continuous light regions and return them as
        Blob features in a FeatureSet.  Parameters specify the binarize filter
        threshold value, and minimum and maximum size for blobs. If a threshold
        value is -1, it will use an adaptive threshold.  See binarize() for
        more information about thresholding.  The threshblocksize and
        threshconstant parameters are only used for adaptive threshold.

        :param threshold: the threshold as an integer or an (r,g,b) tuple,
                           where pixels below (darker) than thresh are set
                           to to max value, and all values above this value
                           are set to black. If this parameter is -1 we use
                           Otsu's method.
        :param minsize: the minimum size of the blobs, in pixels, of the
                         returned blobs. This helps to filter out noise.
        :param maxsize: the maximim size of the blobs, in pixels, of the
                        returned blobs.
        :param threshold_blocksize: the size of the block used in the adaptive
                                    binarize operation.
                                    # TODO - make this match binarize
        :param threshold_constant: the difference from the local mean to use
                                    for thresholding in Otsu's method.
                                    # TODO - make this match binarize
        :param appx_level: the blob approximation level - an integer for
                            the maximum distance between the true edge and
                            the approximation edge - lower numbers yield
                            better approximation.

        :return: a featureset (basically a list) of blob features. If no
                  blobs are found this method returns None.

        :Example:
        >>> img = Image("lena")
        >>> fs = img.find_blobs()
        >>> if fs is not None:
            ... fs.draw()

        :Warning:
        For blobs that live right on the edge of the image OpenCV reports
        the position and width height as being one over for the true position.
        E.g. if a blob is at (0,0) OpenCV reports its position as (1,1).
        Likewise the width and height for the other corners is reported as
        being one less than the width and height. This is a known bug.
        """
        if maxsize == 0:
            maxsize = self.width * self.height
        # create a single channel image, thresholded to parameters

        blobmaker = BlobMaker()
        blobs = blobmaker.extractFromBinary(
            self.binarize(threshold, 255, threshold_blocksize,
                          threshold_constant).invert(),
            self, minsize=minsize, maxsize=maxsize, appx_level=appx_level)

        if not len(blobs):
            return None

        return FeatureSet(blobs).sort_area()

    def find_skintone_blobs(self, minsize=10, maxsize=0, dilate_iter=1):
        """
        Find Skintone blobs will look for continuous regions of Skintone in a
        color image and return them as Blob features in a FeatureSet. Parameters
        specify the binarize filter threshold value, and minimum and maximum
        size for blobs. If a threshold value is -1, it will use an adaptive
        threshold.  See binarize() for more information about thresholding.
        The threshblocksize and threshconstant parameters are only used for
        adaptive threshold.

        :param minsize: the minimum size of the blobs, in pixels, of the
                         returned blobs. This helps to filter out noise.n
        :param maxsize: the maximim size of the blobs, in pixels, of the
                         returned blobs.
        :param dilate_iter: the number of times to run the dilation operation.

        :return: a featureset (basically a list) of blob features. If no blobs
                 are found this method returns None.
        :Example:
        >>> img = Image("lenna")
        >>> fs = img.find_skintone_blobs()
        >>> if fs is not None:
            ... fs.draw()

        :Note:
        It will be really awesome for making UI type stuff, where you want
        to track a hand or a face.
        """
        if maxsize == 0:
            maxsize = self.width * self.height
        mask = self.get_skintone_mask(dilate_iter)
        blobmaker = BlobMaker()
        blobs = blobmaker.extract_from_binary(mask, self, minsize=minsize,
                                              maxsize=maxsize)
        if not len(blobs):
            return None

        return FeatureSet(blobs).sort_area()

    def get_skintone_mask(self, dilate_iter=0):
        """
        Find Skintone mask will look for continuous regions of Skintone in a
        color image and return a binary mask where the white pixels denote
        Skintone region.

        :param dilate_iter: the number of times to run the dilation operation.

        :return: a binary mask.

        :Example:
        >>> img = Image("lenna")
        >>> mask = img.get_skintone_mask()
        >>> mask.show()
        """
        if self._color_space != ColorSpace.YCrCb:
            YCrCb = self.to_ycrcb()
        else:
            YCrCb = self

        Y = npy.ones((256, 1), dtype=uint8) * 0
        Y[5:] = 255
        Cr = npy.ones((256, 1), dtype=uint8) * 0
        Cr[140:180] = 255
        Cb = npy.ones((256, 1), dtype=uint8) * 0
        Cb[77:135] = 255
        Y_img = YCrCb.zeros(1)
        Cr_img = YCrCb.zeros(1)
        Cb_img = YCrCb.zeros(1)
        cv.Split(YCrCb.bitmap, Y_img, Cr_img, Cb_img, None)
        cv.LUT(Y_img, Y_img, cv.fromarray(Y))
        cv.LUT(Cr_img, Cr_img, cv.fromarray(Cr))
        cv.LUT(Cb_img, Cb_img, cv.fromarray(Cb))
        temp = self.zeros()
        cv.Merge(Y_img, Cr_img, Cb_img, None, temp)
        mask = Image(temp, colorSpace=ColorSpace.YCrCb)
        mask = mask.binarize((128, 128, 128))
        mask = mask.to_rgb().binarize()
        mask.dilate(dilate_iter)
        return mask

    # this code is based on code that's based on code from
    # http://blog.jozilla.net/2008/06/27/fun-with-python-opencv-and-face-detection/
    def find_haar_features(self, cascade, scale_factor=1.2, min_neighbors=2,
                           use_canny=cv.CV_HAAR_DO_CANNY_PRUNING,
                           min_size=(20, 20), max_size=(1000, 1000)):
        """
        A Haar like feature cascase is a really robust way of finding the
        location of a known object. This technique works really well for a few
        specific applications like face, pedestrian, and vehicle detection.
        It is worth noting that this approach **IS NOT A MAGIC BULLET** .
        Creating a cascade file requires a large number of images that have
        been sorted by a human. If you want to find Haar Features (useful for
        face detection among other purposes) this will return Haar feature
        objects in a FeatureSet.
        For more information, consult the cv.HaarDetectObjects documentation.
        To see what features are available run img.listHaarFeatures() or you can
        provide your own haarcascade file if you have one available.
        Note that the cascade parameter can be either a filename, or a HaarCascade
        loaded with cv.Load(), or a HaarCascade object.

        :param cascade: the Haar Cascade file, this can be either the path to
                        a cascade file or a HaarCascased SimpleCV object that
                        has already been loaded.
        :param scale_factor: the scaling factor for subsequent rounds of the
                             Haar cascade (default 1.2) in terms of a
                             percentage (i.e. 1.2 = 20% increase in size)
        :param min_neighbors: the minimum number of rectangles that makes up
                              an object. Usually detected faces are clustered
                              around the face, this is the number of detections
                              in a cluster that we need for detection. Higher
                              values here should reduce false positives and
                              decrease false negatives.
        :param use_canny: whether or not to use Canny pruning to reject areas
                          with too many edges (default yes, set to 0 to disable)
        :param min_size: minimum window size. By default, it is set to the size
                         of samples the classifier has been trained on ((20,20)
                         for face detection)
        :param max_size: maximum window size. By default, it is set to the size
                         of samples the classifier has been trained on
                         ((1000,1000) for face detection)
        :return: a feature set of HaarFeatures

        :Example:
        >>> faces = HaarCascade("face.xml","myFaces")
        >>> cam = Camera()
        >>> while True:
        >>>     f = cam.get_image().find_haar_features(faces)
        >>>     if f is not None:
        >>>         f.show()

        :Note:
        OpenCV Docs:
        - http://opencv.willowgarage.com/documentation/python/objdetect_cascade_classification.html
        Wikipedia:
        - http://en.wikipedia.org/wiki/Viola-Jones_object_detection_framework
        - http://en.wikipedia.org/wiki/Haar-like_features
        The video on this pages shows how Haar features and cascades work to located faces:
        - http://dismagazine.com/dystopia/evolved-lifestyles/8115/anti-surveillance-how-to-hide-from-machines/
        """
        storage = cv.CreateMemStorage(0)

        # lovely.  This segfaults if not present
        from PhloxAR.features.HaarCascade import HaarCascade
        if isinstance(cascade, str):
            cascade = HaarCascade(cascade)
            if not cascade.get_cascade():
                return None
        elif isinstance(cascade, HaarCascade):
            pass
        else:
            logger.warning('Could not initialize HaarCascade. Enter Valid'
                           'cascade value.')

        # added all of the arguments from the opencv docs arglist
        import cv2
        haar_classify = cv2.CascadeClassifier(cascade.get_file_handle())
        objects = haar_classify.detectMultiScale(self.gray_narray,
                                                 scale_factor=scale_factor,
                                                 min_neighbors=min_neighbors,
                                                 min_size=min_size,
                                                 flags=use_canny)
        cv2flag = True

        if objects is not None:
            return FeatureSet([HaarFeature(self, o, cascade, cv2flag) for o in objects])

        return None

    def draw_circle(self, ctr, rad, color=(0, 0, 0), thickness=1):
        """
        Draw a circle on the image.

        :param ctr: the center of the circle as an (x,y) tuple.
        :param rad: the radius of the circle in pixels
        :param color: a color tuple (default black)
        :param thickness: the thickness of the circle, -1 means filled in.

        :Example:
        >>> img = Image("lena")
        >>> img.draw_circle((img.width/2,img.height/2),r=50,color=Color.RED,width=3)
        >>> img.show()

        :Note:
        Note that this function is deprecated, try to use DrawingLayer.circle() instead.
        """
        if thickness < 0:
            self.get_drawing_layer().circle((int(ctr[0]), int(ctr[1])),
                                            int(rad),
                                            color, int(thickness), filled=True)
        else:
            self.get_drawing_layer().circle((int(ctr[0]), int(ctr[1])),
                                            int(rad), color, int(thickness))

    def draw_line(self, pt1, pt2, color=(0, 0, 0), thickness=1):
        """
        Draw a line on the image.

        :param pt1: the first point for the line (tuple).
        :param pt2: the second point on the line (tuple).
        :param color: a color tuple (default black).
        :param thickness: the thickness of the line in pixels.

        :return: None


        :Example:
        >>> img = Image("lena")
        >>> img.draw_line((0,0),(img.width,img.height),color=Color.RED,thickness=3)
        >>> img.show()

        :Note:
        This function is deprecated, try to use DrawingLayer.line() instead.
        """
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0]), int(pt2[1]))
        self.get_drawing_layer().line(pt1, pt2, color, thickness)

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
        """
        Split the channels of an image into RGB (not the default BGR)
        single parameter is whether to return the channels as grey images (default)
        or to return them as tinted color image
        :param: grayscale: if this is true we return three grayscale images,
        one per channel. if it is False return tinted images.

        :return: a tuple of of 3 image objects.

        :Example:
        >>> img = Image("lena")
        >>> data = img.split_channels()
        >>> for d in data:
        >>>    d.show()
        >>>    time.sleep(1)
        """
        r = self.zeros(1)
        g = self.zeros(1)
        b = self.zeros(1)
        cv.Split(self.bitmap, b, g, r, None)

        red = self.zeros()
        green = self.zeros()
        blue = self.zeros()

        if grayscale:
            cv.Merge(r, r, r, None, red)
            cv.Merge(g, g, g, None, green)
            cv.Merge(b, b, b, None, blue)
        else:
            cv.Merge(None, None, r, None, red)
            cv.Merge(None, g, None, None, green)
            cv.Merge(b, None, None, None, blue)

        return Image(red), Image(green), Image(blue)

    def merge_channels(self, r=None, g=None, b=None):
        """
        Merge channels is the opposite of splitChannels. The image takes one
        image for each of the R,G,B channels and then recombines them into a
        single image. Optionally any of these channels can be None.

        :param r: the r or last channel  of the result SimpleCV Image.
        :param g: the g or center channel of the result SimpleCV Image.
        :param b: the b or first channel of the result SimpleCV Image.
        :return: Image

        :Example:
        >>> img = Image("lenna")
        >>> [r,g,b] = img.split_channels()
        >>> r = r.binarize()
        >>> g = g.binarize()
        >>> b = b.binarize()
        >>> result = img.merge_channels(r,g,b)
        >>> result.show()
        """
        if r is None and g is None and b is None:
            logger.warning("Image.merge_channels - we need at least one "
                           "valid channel")
            return None
        if r is None:
            r = self.zeros(1)
            cv.Zero(r)
        else:
            rt = r.zeros(1)
            cv.Split(r.bitmap, rt, rt, rt, None)
            r = rt
        if g is None:
            g = self.zeros(1)
            cv.Zero(g)
        else:
            gt = g.zeros(1)
            cv.Split(g.bitmap, gt, gt, gt, None)
            g = gt
        if b is None:
            b = self.zeros(1)
            cv.Zero(b)
        else:
            bt = b.zeros(1)
            cv.Split(b.bitmap, bt, bt, bt, None)
            b = bt

        ret = self.zeros()
        cv.Merge(b, g, r, None, ret)
        return Image(ret)

    def apply_hls_curve(self, hcurve, lcurve, scurve):
        """
        Apply a color correction curve in HSL space. This method can be used
        to change values for each channel. The curves are :py:class:`ColorCurve`
        class objects.

        :param hcurve: the hue ColorCurve object.
        :param lcurve: the lightness / value ColorCurve object.
        :param scurve: the saturation ColorCurve object

        :return: Image

        :Example:
        >>> img = Image("lena")
        >>> hc = ColorCurve([[0,0], [100, 120], [180, 230], [255, 255]])
        >>> lc = ColorCurve([[0,0], [90, 120], [180, 230], [255, 255]])
        >>> sc = ColorCurve([[0,0], [70, 110], [180, 230], [240, 255]])
        >>> img2 = img.apply_hls_curve(hc, lc, sc)
        """
        #TODO CHECK ROI
        #TODO CHECK CURVE SIZE
        #TODO CHECK COLORSPACE
        #TODO CHECK CURVE SIZE
        temp = cv.CreateImage(self.size(), 8, 3)
        # Move to HLS spacecol
        cv.CvtColor(self._bitmap, temp, cv.CV_RGB2HLS)
        tmp_mat = cv.GetMat(temp)  # convert the bitmap to a matrix
        # now apply the color curve correction
        tmp_mat = npy.array(self.matrix).copy()
        tmp_mat[:, :, 0] = npy.take(hcurve.curve, tmp_mat[:, :, 0])
        tmp_mat[:, :, 1] = npy.take(scurve.curve, tmp_mat[:, :, 1])
        tmp_mat[:, :, 2] = npy.take(lcurve.curve, tmp_mat[:, :, 2])
        # Now we jimmy the np array into a cvMat
        image = cv.CreateImageHeader((tmp_mat.shape[1], tmp_mat.shape[0]),
                                     cv.IPL_DEPTH_8U, 3)
        cv.SetData(image, tmp_mat.tostring(),
                   tmp_mat.dtype.itemsize * 3 * tmp_mat.shape[1])
        cv.CvtColor(image, image, cv.CV_HLS2RGB)
        return Image(image, color_space=self._color_space)

    def apply_rgb_curve(self, rcurve, gcurve, bcurve):
        """
        Apply a color correction curve in RGB space. This method can be used
        to change values for each channel. The curves are :py:class:`ColorCurve`
        class objects.

        :param rcurve: the red ColorCurve object, or appropriately formatted list
        :param gcurve: the green ColorCurve object, or appropriately formatted list
        :param bcurve: the blue ColorCurve object, or appropriately formatted list

        :return: Image

        :Example:
        >>> img = Image("lena")
        >>> rc = ColorCurve([[0,0], [100, 120], [180, 230], [255, 255]])
        >>> gc = ColorCurve([[0,0], [90, 120], [180, 230], [255, 255]])
        >>> bc = ColorCurve([[0,0], [70, 110], [180, 230], [240, 255]])
        >>> img2 = img.apply_rgb_curve(rc, gc, bc)
        """
        if isinstance(bcurve, list):
            bcurve = ColorCurve(bcurve)
        if isinstance(gcurve, list):
            gcurve = ColorCurve(gcurve)
        if isinstance(rcurve, list):
            rcurve = ColorCurve(rcurve)

        tmp_mat = npy.array(self.matrix).copy()
        tmp_mat[:, :, 0] = npy.take(bcurve.curve, tmp_mat[:, :, 0])
        tmp_mat[:, :, 1] = npy.take(gcurve.curve, tmp_mat[:, :, 1])
        tmp_mat[:, :, 2] = npy.take(rcurve.curve, tmp_mat[:, :, 2])
        # Now we jimmy the np array into a cvMat
        image = cv.CreateImageHeader((tmp_mat.shape[1], tmp_mat.shape[0]),
                                     cv.IPL_DEPTH_8U, 3)
        cv.SetData(image, tmp_mat.tostring(),
                   tmp_mat.dtype.itemsize * 3 * tmp_mat.shape[1])
        return Image(image, color_space=self._color_space)

    def apply_intensity_curve(self, curve):
        """
        Intensity applied to all three color channels

        :param curve: a ColorCurve object, or 2d list that can be conditioned
                      into one
        :return:
        Image

        >>> img = Image("lenna")
        >>> rc = ColorCurve([[0,0], [100, 120], [180, 230], [255, 255]])
        >>> gc = ColorCurve([[0,0], [90, 120], [180, 230], [255, 255]])
        >>> bc = ColorCurve([[0,0], [70, 110], [180, 230], [240, 255]])
        >>> img2 = img.apply_rgb_curve(rc,gc,bc)
        """
        return self.apply_rgb_curve(curve, curve, curve)

    def color_distance(self, color=Color.BLACK):
        # reshape our matrix to 1xN
        pixels = npy.array(self.narray).reshape(-1, 3)
        # calculate the distance each pixel is
        distances = spsd.cdist(pixels, [color])
        distances *= (255.0 / distances.max())  # normalize to 0 - 255
        # return an Image
        return Image(distances.reshape(self.width, self.height))

    def hue_distance(self, color=Color.BLACK, minsaturation=20, minvalue=20,
                     maxvalue=255):
        if isinstance(color, (float, int, long, complex)):
            color_hue = color
        else:
            color_hue = Color.rgb_to_hsv(color)[0]

        # again, gets transposed to vsh
        vsh_matrix = self.to_bgr().narray.reshape(-1, 3)
        hue_channel = npy.cast['int'](vsh_matrix[:, 2])

        if color_hue < 90:
            hue_loop = 180
        else:
            hue_loop = -180
        # set whether we need to move back or forward on the hue circle

        distances = npy.minimum(npy.abs(hue_channel - color_hue),
                               npy.abs(hue_channel - (color_hue + hue_loop)))
        # take the minimum distance for each pixel


        distances = npy.where(npy.logical_and(vsh_matrix[:, 0] > minvalue,
                                              vsh_matrix[:, 1] > minsaturation),
                              distances * (255.0 / 90.0),  # normalize 0 - 90 -> 0 - 255
                              255.0)  # use the maxvalue if it false outside of our value/saturation tolerances

        return Image(distances.reshape(self.width, self.height))

    def erode(self, iterations=1, kernelsize=3):
        ret = self.zeros()
        kern = cv.CreateStructuringElementEx(kernelsize, kernelsize, 1, 1,
                                             cv.CV_SHAPE_RECT)
        cv.Erode(self.bitmap, ret, kern, iterations)
        return Image(ret, color_space=self._color_space)

    def dilate(self, iteration=1):
        ret = self.zeros()
        kern = cv.CreateStructuringElementEx(3, 3, 1, 1, cv.CV_SHAPE_RECT)
        cv.Dilate(self.bitmap, ret, kern, iteration)
        return Image(ret, color_space=self._color_space)

    def morph_open(self):
        ret = self.zeros()
        temp = self.zeros()
        kern = cv.CreateStructuringElementEx(3, 3, 1, 1, cv.CV_SHAPE_RECT)
        try:
            cv.MorphologyEx(self.bitmap, ret, temp, kern, cv.MORPH_OPEN, 1)
        except:
            cv.MorphologyEx(self.bitmap, ret, temp, kern, cv.CV_MOP_OPEN, 1)
            # OPENCV 2.2 vs 2.3 compatability

        return Image(ret)

    def morph_close(self):
        ret = self.zeros()
        temp = self.zeros()
        kern = cv.CreateStructuringElementEx(3, 3, 1, 1, cv.CV_SHAPE_RECT)
        try:
            cv.MorphologyEx(self.bitmap, ret, temp, kern, cv.MORPH_CLOSE, 1)
        except:
            cv.MorphologyEx(self.bitmap, ret, temp, kern, cv.CV_MOP_CLOSE, 1)
            # OPENCV 2.2 vs 2.3 compatability

        return Image(ret, color_space=self._color_space)

    def morph_gradient(self):
        ret = self.zeros()
        temp = self.zeros()
        kern = cv.CreateStructuringElementEx(3, 3, 1, 1, cv.CV_SHAPE_RECT)
        try:
            cv.MorphologyEx(self.bitmap, ret, temp, kern, cv.MORPH_GRADIENT, 1)
        except:
            cv.MorphologyEx(self.bitmap, ret, temp, kern, cv.CV_MOP_GRADIENT, 1)
        return Image(ret, color_space=self._color_space)

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

    def rotate(self, angle, fixed=True, point=None, scale=1.0):
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