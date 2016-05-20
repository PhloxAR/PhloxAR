#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import unicode_literals, absolute_import

from PhloxAR.core.color import *
from PhloxAR.core.dft import *
from PhloxAR.core.drawing_layer import *
from PhloxAR.core.linescan import *
from PhloxAR.core.stream import *
from PhloxAR.features import *
from PhloxAR.tracking import *

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
import multipledispatch


class Image(object):
    """
    The Image class allows you to convert to and from a number of source types
    with ease. It also has intelligent buffer management, so that modified
    copies of the Image required for algorithms such as edge detection, etc can
    be cached an reused when appropriate.

    Image are converted into 8-bit, 3-channel images in RGB _color space. It
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
    # deprecated
    _matrix = ''  # the matrix (cvmat) representation

    _pilimg = ''  # holds a PIL Image object in buffer
    _surface = ''  # pygame surface representation of the image
    _narray = ''  # numpy array representation of the image

    # TODO: pyglet patch
    _patch = ''

    _gray_matrix = ''  # the gray scale (cvmat) representation
    _gray_narray = ''  # gray scale numpy array for key point stuff

    _equalized_gray_bitmap = ''  # the normalized bitmap

    _blob_label = ''  # the label image for blobbing
    _edge_map = ''  # holding reference for edge map
    _canny_param = ''  # parameters that created _edge_map
    _color_space = ColorSpace.UNKNOWN
    _grid_layer = [None, [0, 0]]

    # for DFT caching
    _dft = []  # an array of 2 channel (real, imaginary) 64f images

    # keypoint caching values
    _keypoints = None
    _kp_descriptors = None
    _kp_flavor = 'None'

    # temp files
    _tmp_files = []

    # when we empty the buffers, populate with this:
    _initialized_buffers = {
        '_matrix': '',
        '_gray_matrix': '',
        '_equalized_gray_bitmap': '',
        '_blob_label': '',
        '_edge_map': '',
        '_canny_param': (0, 0),
        '_pilimg': '',
        '_narray': '',
        '_gray_numpy': '',
        '_pygame_surface': '',
    }

    # used to buffer the points when we crop the image.
    _uncropped_x = 0
    _uncropped_y = 0

    # initialize the frame
    # TODO: handle camera/capture from file cases (detect on file extension)
    def __int__(self, src, camera=None, color_space=ColorSpace.UNKNOWN,
                 verbose=True, sample=False, cv2image=False, webp=False):
        """
        Takes a single polymorphic parameter, tests to see how it should convert
        to RGB image.

        :param src: the source of the image, could be anything, a file name,
                        a width and height tuple, a url. Certain strings such as
                        'lena', or 'logo' are loaded automatically for quick test.
        :param camera: a camera to pull a live image
        :param color_space: default camera _color space
        :param verbose:
        :param sample: set True, if you want to load som of the included sample
                        images without having to specify complete path
        :param cv2image:
        :param webp:

        Note:
        Python Image Library: Image type
        Filename: All OpenCV supported types (jpg, png, bmp, gif, etc)
        URL: The source can be a url, but must include the http://
        """
        self._layers = []
        self.camera = camera
        self._color_space = color_space
        # keypoint descriptors
        self._keypoints = []
        self._kp_descriptors = []
        self._kp_flavor = 'None'
        # palette stuff
        self._do_hue_palette = False
        self._palette_bins = None
        self._palette = None
        self._palette_members = None
        self._palette_percentages = None
        # tmp files
        self._tmp_files = []

        # check url
        if isinstance(src, str) and (src[:7].lower() == 'http://' or src[:8].lower() == 'https://'):
            req = urllib2.Request(src, headers={
                'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_4) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.54 Safari/536.5"
            })

            img_file = urllib2.urlopen(req)

            img = StringIO(img_file.read())

            src = PILImage.open(img).convert("RGB")

        # check base64 url
        if isinstance(src, str) and (src.lower().startswith('data:image/png;base64')):
            ims = src[22:].decode('base64')
            img = StringIO(ims)
            src = PILImage.open(img).convert("RGB")

        if isinstance(src, str):
            tmp_name = src.lower()
            if tmp_name == 'lena':
                imgpth = os.path.join(LAUNCH_PATH, 'sample_images', 'lena.jpg')
                src = imgpth
            elif sample:
                imgpth = os.path.join(LAUNCH_PATH, 'sample_images', src)
                src = imgpth

        scls = src.__class__
        sclsbase = scls.__base__
        sclsname = scls.__name__

        if isinstance(src, tuple):
            w = int(src[0])
            h = int(src[1])
            src = np.zeros((w, h, 3), np.uint8)

        elif isinstance(src, np.ndarray):  # handle a numpy array conversion
            if isinstance(src[0, 0], np.ndarray):  # a 3 channel array
                src = src.astype(np.uint8)
                self._narray = src
                self._color_space = ColorSpace.BGR
            else:
                # we have a single channel array, convert to an RGB iplimage
                src = src.astype(np.uint8)
                self._narray = src
                size = (src.shape[1], src.shape[0])
                self._color_space = ColorSpace.BGR
        elif isinstance(src, str) or sclsname == 'StringIO':
            if src == '':
                raise IOError("No filename provided to Image constructor")
            elif webp or src.split('.')[-1] == 'webp':
                try:
                    if sclsname == 'StringIO':
                        src.seek(0)  # set the stringIO to the beginning
                    self._pilimg = PILImage.open(src)
                    self._narray = np.asarray(self._pilimg)
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
                                                    (result.width,
                                                     result.height),
                                                    str(result.bitmap),
                                                    "raw", "RGB", 0, 1)
                    self._pilimg = webpimage.convert("RGB")
                    self._narray = np.asarray(self._pilimg)
                    self.filename = src
            else:
                self.filename = src

                self._pilimg = PILImage.open(self.filename).convert("RGB")
                self._narray = np.asarray(self._pilimg)
                self._color_space = ColorSpace.RGB
        elif isinstance(src, sdl2.Surface):
            self._surface = src
            self._pilimg = sdl2.image.tostring(src, 'RGB')
            self._narray = np.asarray(self._pilimg)
            cv2.cvtColor(self._narray, self._narray, cv2.COLOR_RGB2BGR)
            self._color_space = ColorSpace.BGR
        elif (PIL_ENABLED and ((len(sclsbase) and sclsbase[0].__name__ == "ImageFile") or sclsname == "JpegImageFile" or sclsname == "WebPPImageFile" or sclsname == "Image")):
            if src.mode != 'RGB':
                src = src.convert('RGB')
            self._pilimg = src
            # from OpenCV cookbook
            # http://opencv2.willowgarage.com/documentation/python/cookbook.html
            self._bitmap = cv2.CreateImageHeader(self._pilimg.size,
                                                cv2.IPL_DEPTH_8U, 3)
            cv2.SetData(self._bitmap, self._pilimg.tostring())
            self._color_space = ColorSpace.BGR
            cv2.cvtColor(self._bitmap, self._bitmap, cv2.CV_RGB2BGR)
        else:
            return None

        if color_space != ColorSpace.UNKNOWN:
            self._color_space = color_space

        self.width, self.height, self.depth = self._narray.shape

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
        Left click will show mouse coordinates and _color.
        Right click will kill the live image.

        :return: None.

        :Example:
        >>> cam = Camera()
        >>> cam.live()
        """
        start_time = time.time()

        from PhloxAR.core.display import Display

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
                i.dl().text("Left click will show mouse coordinates and _color",
                            (10, 20), color=col)
                i.dl().text("Right click will kill the live image", (10, 30),
                            color=col)

            i.save(d)
            if d.mouse_r:
                print("Closing Window!")
                d.done = True

    @property
    def color_space(self):
        """
        Returns Image's _color space.
        :return: integer corresponding to the _color space.
        """
        return self._color_space

    get_color_space = color_space

    def is_rgb(self):
        """
        Returns true if the image uses the RGB _color space.
        :return: Bool
        """
        return self._color_space == ColorSpace.RGB

    def is_bgr(self):
        """
        Returns true if the image uses the BGR _color space.
        :return: Bool
        """
        return self._color_space == ColorSpace.BGR

    def is_hsv(self):
        """
        Returns true if the image uses the HSV _color space.
        :return: Bool
        """
        return self._color_space == ColorSpace.HSV

    def is_hls(self):
        """
        Returns true if the image uses the HLS _color space.
        :return: Bool
        """
        return self._color_space == ColorSpace.HLS

    def is_xyz(self):
        """
        Returns true if the image uses the XYZ _color space.
        :return: Bool
        """
        return self._color_space == ColorSpace.XYZ

    def is_gray(self):
        """
        Returns true if the image uses the grayscale _color space.
        :return: Bool
        """
        return self._color_space == ColorSpace.GRAY

    def is_ycrcb(self):
        """
        Returns true if the image uses the YCrCb _color space.
        :return: Bool
        """
        return self._color_space == ColorSpace.YCrCb

    def to_rgb(self):
        """
        Convert the image to RGB _color space.
        :return: image in RGB
        """
        img = self.zeros()

        if self.is_bgr() or self._color_space == ColorSpace.UNKNOWN:
            cv2.cvtColor(self._narray, img, cv2.COLOR_BGR2RGB)
        elif self.is_hsv():
            cv2.cvtColor(self._narray, img, cv2.COLOR_HSV2RGB)
        elif self.is_hls():
            cv2.cvtColor(self._narray, img, cv2.COLOR_HLS2RGB)
        elif self.is_xyz():
            cv2.cvtColor(self._narray, img, cv2.COLOR_XYZ2RGB)
        elif self.is_ycrcb():
            cv2.cvtColor(self._narray, img, cv2.COLOR_YCrCb2RGB)
        elif self.is_rgb():
            img = self._narray
        else:
            logger.warning("Image.to_rgb: conversion no supported.")

        return Image(img, color_space=ColorSpace.RGB)

    def to_bgr(self):
        """
        Convert the image to BGR _color space.
        :return: image in BGR
        """
        img = self.zeros()

        if self.is_rgb() or self._color_space == ColorSpace.UNKNOWN:
            cv2.cvtColor(self._narray, img, cv2.COLOR_RGB2BGR)
        elif self.is_hsv():
            cv2.cvtColor(self._narray, img, cv2.COLOR_HSV2BGR)
        elif self.is_hls():
            cv2.cvtColor(self._narray, img, cv2.COLOR_HLS2BGR)
        elif self.is_xyz():
            cv2.cvtColor(self._narray, img, cv2.COLOR_XYZ2BGR)
        elif self.is_ycrcb():
            cv2.cvtColor(self._narray, img, cv2.COLOR_YCrCb2BGR)
        elif self.is_bgr():
            img = self.bitmap
        else:
            logger.warning("Image.to_bgr: conversion no supported.")

        return Image(img, color_space=ColorSpace.BGR)

    def to_hls(self):
        """
            Convert the image to HLS _color space.
            :return: image in HLS
            """
        img = img_tmp = self.zeros()

        if self.is_rgb() or self._color_space == ColorSpace.UNKNOWN:
            cv2.cvtColor(self._narray, img, cv2.COLOR_RGB2HLS)
        elif self.is_bgr():
            cv2.cvtColor(self._narray, img, cv2.COLOR_BGR2HLS)
        elif self.is_hsv():
            cv2.cvtColor(self._narray, img_tmp, cv2.COLOR_HSV2RGB)
            cv2.cvtColor(img_tmp, img, cv2.COLOR_RGB2HLS)
        elif self.is_xyz():
            cv2.cvtColor(self._narray, img_tmp, cv2.COLOR_XYZ2RGB)
            cv2.cvtColor(img_tmp, img, cv2.COLOR_RGB2HLS)
        elif self.is_ycrcb():
            cv2.cvtColor(self._narray, img_tmp, cv2.COLOR_YCrCb2RGB)
            cv2.cvtColor(img_tmp, img, cv2.COLOR_RGB2HLS)
        elif self.is_hls():
            img = self._narray
        else:
            logger.warning("Image.to_rgb: conversion no supported.")

        return Image(img, color_space=ColorSpace.HLS)

    def to_hsv(self):
        """
            Convert the image to HSV _color space.
            :return: image in HSV
            """
        img = img_tmp = self.zeros()

        if self.is_rgb() or self._color_space == ColorSpace.UNKNOWN:
            cv2.cvtColor(self._narray, img, cv2.COLOR_RGB2HSV)
        elif self.is_bgr():
            cv2.cvtColor(self._narray, img, cv2.COLOR_BGR2HSV)
        elif self.is_hls():
            cv2.cvtColor(self._narray, img_tmp, cv2.COLOR_HLS2RGB)
            cv2.cvtColor(img_tmp, img, cv2.COLOR_RGB2HSV)
        elif self.is_xyz():
            cv2.cvtColor(self._narray, img_tmp, cv2.COLOR_XYZ2RGB)
            cv2.cvtColor(img_tmp, img, cv2.COLOR_RGB2HSV)
        elif self.is_ycrcb():
            cv2.cvtColor(self._narray, img_tmp, cv2.COLOR_YCrCb2RGB)
            cv2.cvtColor(img_tmp, img, cv2.COLOR_RGB2HSV)
        elif self.is_hsv():
            img = self._narray
        else:
            logger.warning("Image.to_rgb: conversion no supported.")

        return Image(img, color_space=ColorSpace.HSV)

    def to_xyz(self):
        """
            Convert the image to XYZ _color space.
            :return: image in XYZ
            """
        img = img_tmp = self.zeros()

        if self.is_rgb() or self._color_space == ColorSpace.UNKNOWN:
            cv2.cvtColor(self._narray, img, cv2.COLOR_RGB2XYZ)
        elif self.is_bgr():
            cv2.cvtColor(self._narray, img, cv2.COLOR_BGR2XYZ)
        elif self.is_hsv():
            cv2.cvtColor(self._narray, img_tmp, cv2.COLOR_HSV2RGB)
            cv2.cvtColor(img_tmp, img, cv2.COLOR_RGB2XYZ)
        elif self.is_hls():
            cv2.cvtColor(self._narray, img_tmp, cv2.COLOR_HLS2RGB)
            cv2.cvtColor(img_tmp, img, cv2.COLOR_RGB2XYZ)
        elif self.is_ycrcb():
            cv2.cvtColor(self._narray, img_tmp, cv2.COLOR_YCrCb2RGB)
            cv2.cvtColor(img_tmp, img, cv2.COLOR_RGB2XYZ)
        elif self.is_xyz():
            img = self._narray
        else:
            logger.warning("Image.to_rgb: conversion no supported.")

        return Image(img, color_space=ColorSpace.XYZ)

    def to_gray(self):
        """
        Convert the image to GRAY _color space.
        :return: image in GRAY
        """
        img = img_tmp = self.zeros(1)

        if self.is_rgb() or self._color_space == ColorSpace.UNKNOWN:
            cv2.cvtColor(self._narray, img, cv2.COLOR_RGB2GRAY)
        elif self.is_bgr():
            cv2.cvtColor(self._narray, img, cv2.COLOR_BGR2GRAY)
        elif self.is_hls():
            cv2.cvtColor(self._narray, img_tmp, cv2.COLOR_HLS2RGB)
            cv2.cvtColor(img_tmp, img, cv2.COLOR_RGB2GRAY)
        elif self.is_hsv():
            cv2.cvtColor(self._narray, img_tmp, cv2.COLOR_HSV2RGB)
            cv2.cvtColor(img_tmp, img, cv2.COLOR_RGB2GRAY)
        elif self.is_xyz():
            cv2.cvtColor(self._narray, img_tmp, cv2.COLOR_XYZ2RGB)
            cv2.cvtColor(img_tmp, img, cv2.COLOR_RGB2GRAY)
        elif self.is_ycrcb():
            cv2.cvtColor(self._narray, img_tmp, cv2.COLOR_YCrCb2RGB)
            cv2.cvtColor(img_tmp, img, cv2.COLOR_RGB2GRAY)
        elif self.is_gray():
            img = self._narray
        else:
            logger.warning("Image.to_rgb: conversion no supported.")

        return Image(img, color_space=ColorSpace.GRAY)

    def to_ycrcb(self):
        """
        Convert the image to RGB _color space.
        :return: image in RGB
        """
        img = img_tmp = self.zeros()

        if self.is_rgb() or self._color_space == ColorSpace.UNKNOWN:
            cv2.cvtColor(self._narray, img, cv2.COLOR_RGB2YCrCb)
        elif self.is_bgr():
            cv2.cvtColor(self._narray, img, cv2.COLOR_BGR2YCrCb)
        elif self.is_hsv():
            cv2.cvtColor(self._narray, img_tmp, cv2.COLOR_HSV2RGB)
            cv2.cvtColor(img_tmp, img, cv2.COLOR_RGB2YCrCb)
        elif self.is_xyz():
            cv2.cvtColor(self._narray, img_tmp, cv2.COLOR_XYZ2RGB)
            cv2.cvtColor(img_tmp, img, cv2.COLOR_RGB2YCrCb)
        elif self.is_hls():
            cv2.cvtColor(self._narray, img_tmp, cv2.COLOR_HLS2RGB)
            cv2.cvtColor(img_tmp, img, cv2.COLOR_RGB2YCrCb)
        elif self.is_ycrcb():
            img = self._narray
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
        :return: a numpy array that matches the width, height, and
                  _color depth of the source image.
        """
        img = np.zeros((self.width, self.height, channels), np.uint8)

        return img

    get_empty = zeros

    # deprecated
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
            self._matrix = cv2.GetMat(self._bitmap)
            return self._matrix

    get_matrix = matrix.fget

    # deprecated
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
            self._gray_matrix = cv2.GetMat(self._get_gray_narray())
            return self._gray_matrix

    get_grayscale_matrix = gray_matrix.fget

    # deprecated
    @property
    def float_matrix(self):
        """
        Converts the standard int bitmap to a floating point bitmap.
        Handy for OpenCV function.
        :return: the floating point OpenCV CvMat version of this image.
        """
        img = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_32F, 3)
        cv2.Convert(self.bitmap, img)

        return img

    get_float_matrix = float_matrix

    @property
    def pilimg(self):
        """
        Get PIL Image object for use with the Python Image Library.
        Handy for PIL functions.
        :return: PIL Image
        """
        if not self._pilimg:
            self._pilimg = PILImage.fromarray(self._narray)

        return self._pilimg


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
            self._gray_narray = uint8(np.array(cv2.GetMat(
                    self._get_gray_narray()
            )).transpose())

    get_grayscale_narray = gray_narray.fget

    def _get_gray_narray(self):
        """
        Gray scaling the image.
        :return: gray scaled image.
        """
        if self._gray_narray:
            return self._gray_narray

        self._gray_narray = self.zeros(1)
        tmp = self.zeros(3)

        if (self._color_space == ColorSpace.BGR or
                    self._color_space == ColorSpace.UNKNOWN):
            cv2.cvtColor(self._narray, self._gray_bitmap, cv2.CV_BGR2GRAY)
        elif self._color_space == ColorSpace.RGB:
            cv2.cvtColor(self._narray, self._gray_bitmap, cv2.CV_RGB2GRAY)
        elif self._color_space == ColorSpace.HLS:
            cv2.cvtColor(self._narray, tmp, cv2.CV_HLS2RGB)
            cv2.cvtColor(tmp, self._gray_bitmap, cv2.CV_RGB2GRAY)
        elif self._color_space == ColorSpace.HSV:
            cv2.cvtColor(self._narray, tmp, cv2.CV_HSV2RGB)
            cv2.cvtColor(tmp, self._gray_bitmap, cv2.CV_RGB2GRAY)
        elif self._color_space == ColorSpace.XYZ:
            cv2.cvtColor(self._narray, tmp, cv2.CV_XYZ2RGB)
            cv2.cvtColor(tmp, self._gray_bitmap, cv2.CV_RGB2GRAY)
        elif self._color_space == ColorSpace.GRAY:
            cv2.Split(self.bitmap, self._gray_bitmap, self._gray_bitmap,
                     self._gray_bitmap, None)
        else:
            logger.warning("Image._gray_bitmap: There is no supported "
                           "conversion to gray _color space.")

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
        cv2.EqualizeHist(self._get_gray_narray(), self._equalized_gray_bitmap)

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
                self._surface = sdl2.image.fromstring(
                    self.to_rgb().bitmap.tostring(),
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
            cv2.SaveImage(filename, img_save.bitmap)
            self.filename = filename
            self.filehandle = None
        elif self.filename:
            cv2.SaveImage(self.filename, img_save.bitmap)
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
        cv2.Copy(self.bitmap, img)

        return Image(img, colorspace=self._color_space)

    def scale(self, scalar, interp=cv2.INTER_LINEAR):
        """
        Scale the image to a new width and height.
        If no height is provided, the width is considered a scaling value

        :param scalar: scalar to scale
        :param interp: how to generate new pixels that don't match the original
                       pixels. Argument goes direction to cv2.Resize.
                       See http://docs.opencv2.org/modules/imgproc/doc/geometric_transformations.html?highlight=resize#cv2.resize for more details

        :return: resized image.

        :Example:
        >>> img.scale(2.0)
        """
        if scalar is not None:
            w = int(self.width * scalar)
            h = int(self.height * scalar)
            if w > MAX_DIMS or h > MAX_DIMS or h < 1 or w < 1:
                logger.warning("You tried to make an image really big or "
                               "impossibly small. I can't scale that")
                return self
        else:
            return self

        scaled_array = np.zeros((w, h, 3), dtype='uint8')
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

        scaled_bitmap = cv2.CreateImage((width, height), 8, 3)
        cv2.Resize(self.bitmap, scaled_bitmap)
        return Image(scaled_bitmap, color_space=self._color_space)

    def smooth(self, method='gaussian', aperture=(3, 3), sigma=0,
               spatial_sigma=0, grayscale=False):
        """
        Smooth the image, by default with the Gaussian blur. If desired,
        additional algorithms and apertures can be specified. Optional
        parameters are passed directly to OpenCV's cv2.Smooth() function.
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
            m = cv2.CV_BLUR
        elif method == 'bilateral':
            m = cv2.CV_BILATERAL
            win_y = win_x  # aperture must be square
        elif method == 'median':
            m = cv2.CV_MEDIAN
            win_y = win_x  # aperture must be square
        else:
            m = cv2.CV_GAUSSIAN  # default method

        if grayscale:
            new_img = self.zeros(1)
            cv2.Smooth(self._get_gray_narray(), new_img, method, win_x, win_y,
                       sigma, spatial_sigma)
        else:
            new_img = self.zeros(3)
            r = self.zeros(1)
            g = self.zeros(1)
            b = self.zeros(1)
            ro = self.zeros(1)
            go = self.zeros(1)
            bo = self.zeros(1)
            cv2.Split(self.bitmap, b, g, r, None)
            cv2.Smooth(r, ro, method, win_x, win_y, sigma, spatial_sigma)
            cv2.Smooth(g, go, method, win_x, win_y, sigma, spatial_sigma)
            cv2.Smooth(b, bo, method, win_x, win_y, sigma, spatial_sigma)
            cv2.Merge(bo, go, ro, None, new_img)

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
        :param sigma_color: filter the specified value in the _color space.
                            A larger value of the parameter means that farther
                            colors within the pixel neighborhood will be mixed
                            together, resulting in larger areas of semi-equal
                            _color.
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
            else:
                logger.warning(
                    "The aperture (win_x,win_y) must be odd number and greater than 0.")
                return None
        else:
            win_x = 3  # set the default aperture window size (3x3)
            diameter = (win_x, win_x)

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
        return Image(self._get_gray_narray(), color_space=ColorSpace.GRAY)

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
        cv2.Flip(self.bitmap, new_img, 1)
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
        cv2.Flip(self.bitmap, new_img, 0)
        return Image(new_img, color_space=self._color_space)

    def stretch(self, thresh_low=0, thresh_high=255):
        """
        The stretch filter works on a greyscale image, if the image
        is _color, it returns a greyscale image.  The filter works by
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
            cv2.Threshold(self._get_gray_narray(), new_img, thresh_low, 255,
                          cv2.CV_THRESH_TOZERO)
            cv2.Not(new_img, new_img)
            cv2.Threshold(new_img, new_img, 255 - thresh_high, 255,
                         cv2.CV_THRESH_TOZERO)
            cv2.Not(new_img, new_img)
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
        maxv and all above to black.  If a _color tuple is provided, each _color
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
        >>> b3 = b1. side_by_side(b2)
        >>> b3.show()

        :Note:
        Otsu's Method Description<http://en.wikipedia.org/wiki/Otsu's_method>`
        """
        if istuple(thresh):
            r = self.zeros(1)
            g = self.zeros(1)
            b = self.zeros(1)
            cv2.Split(self.bitmap, b, g, r, None)

            cv2.Threshold(r, r, thresh[0], maxv, cv2.CV_THRESH_BINARY_INV)
            cv2.Threshold(g, g, thresh[1], maxv, cv2.CV_THRESH_BINARY_INV)
            cv2.Threshold(b, b, thresh[2], maxv, cv2.CV_THRESH_BINARY_INV)

            cv2.Add(r, g, r)
            cv2.Add(r, b, r)

            return Image(r, color_space=self._color_space)
        elif thresh == -1:
            new_bitmap = self.zeros(1)
            if blocksize:
                cv2.AdaptiveThreshold(self._get_gray_narray(), new_bitmap,
                                      maxv,
                                      cv2.CV_ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.CV_THRESH_BINARY_INV, blocksize, p)
            else:
                cv2.Threshold(self._get_gray_narray(), new_bitmap, thresh,
                              float(maxv),
                              cv2.CV_THRESH_BINARY_INV + cv2.CV_THRESH_OTSU)
            return Image(new_bitmap, color_space=self._color_space)
        else:
            new_bitmap = self.zeros(1)
            # desaturate the image, and apply the new threshold
            cv2.Threshold(self._get_gray_narray(), new_bitmap, thresh,
                          float(maxv), cv2.CV_THRESH_BINARY_INV)
            return Image(new_bitmap, color_space=self._color_space)

    def mean_color(self, color_space=None):
        """
        This method finds the average _color of all the pixels in the image and
        displays tuple in the _color space specfied by the user.
        If no _color space is specified , (B,G,R) _color space is taken as default.

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
            return tuple(cv2.Avg(self.bitmap())[0:3])
        elif color_space == 'BGR':
            return tuple(cv2.Avg(self.to_bgr().bitmap)[0:3])
        elif color_space == 'RGB':
            return tuple(cv2.Avg(self.to_rgb().bitmap)[0:3])
        elif color_space == 'HSV':
            return tuple(cv2.Avg(self.to_hsv().bitmap)[0:3])
        elif color_space == 'XYZ':
            return tuple(cv2.Avg(self.to_xyz().bitmap)[0:3])
        elif color_space == 'Gray':
            return cv2.Avg(self._get_gray_narray())[0]
        elif color_space == 'YCrCb':
            return tuple(cv2.Avg(self.to_ycrcb().bitmap)[0:3])
        elif color_space == 'HLS':
            return tuple(cv2.Avg(self.to_hls().bitmap)[0:3])
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
        >>> img = Image("sampleimages/simplecv2.png")
        >>> corners = img.find_corners()
        >>> if corners: True

        >>> img = Image("sampleimages/black.png")
        >>> corners = img.find_corners()
        >>> if not corners: True
        """
        # initialize buffer frames
        eig_image = cv2.CreateImage(cv2.GetSize(self.bitmap),
                                   cv2.IPL_DEPTH_32F, 1)
        temp_image = cv2.CreateImage(cv2.GetSize(self.bitmap),
                                    cv2.IPL_DEPTH_32F, 1)

        corner_coordinates = cv2.GoodFeaturesToTrack(self._get_gray_narray(),
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
        blobs = blobmaker.extract_from_binary(
                self.binarize(threshold, 255, threshold_blocksize,
                              threshold_constant).invert(),
                self, minsize=minsize, maxsize=maxsize, appx_level=appx_level)

        if not len(blobs):
            return None

        return FeatureSet(blobs).sort_area()

    def find_skintone_blobs(self, minsize=10, maxsize=0, dilate_iter=1):
        """
        Find Skintone blobs will look for continuous regions of Skintone in a
        _color image and return them as Blob features in a FeatureSet. Parameters
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
        _color image and return a binary mask where the white pixels denote
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

        Y = np.ones((256, 1), dtype=uint8) * 0
        Y[5:] = 255
        Cr = np.ones((256, 1), dtype=uint8) * 0
        Cr[140:180] = 255
        Cb = np.ones((256, 1), dtype=uint8) * 0
        Cb[77:135] = 255
        Y_img = YCrCb.zeros(1)
        Cr_img = YCrCb.zeros(1)
        Cb_img = YCrCb.zeros(1)
        cv2.Split(YCrCb.bitmap, Y_img, Cr_img, Cb_img, None)
        cv2.LUT(Y_img, Y_img, cv2.fromarray(Y))
        cv2.LUT(Cr_img, Cr_img, cv2.fromarray(Cr))
        cv2.LUT(Cb_img, Cb_img, cv2.fromarray(Cb))
        temp = self.zeros()
        cv2.Merge(Y_img, Cr_img, Cb_img, None, temp)
        mask = Image(temp, color_space=ColorSpace.YCrCb)
        mask = mask.binarize((128, 128, 128))
        mask = mask.to_rgb().binarize()
        mask.dilate(dilate_iter)
        return mask

    # this code is based on code that's based on code from
    # http://blog.jozilla.net/2008/06/27/fun-with-python-opencv-and-face-detection/
    def find_haar_features(self, cascade, scale_factor=1.2, min_neighbors=2,
                           use_canny=cv2.CV_HAAR_DO_CANNY_PRUNING,
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
        For more information, consult the cv2.HaarDetectObjects documentation.
        To see what features are available run img.listHaarFeatures() or you can
        provide your own haarcascade file if you have one available.
        Note that the cascade parameter can be either a filename, or a HaarCascade
        loaded with cv2.Load(), or a HaarCascade object.

        :param cascade: the Haar Cascade file, this can be either the path to
                        a cascade file or a HaarCascased PhloxAR object that
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
        - http://opencv2.willowgarage.com/documentation/python/objdetect_cascade_classification.html
        Wikipedia:
        - http://en.wikipedia.org/wiki/Viola-Jones_object_detection_framework
        - http://en.wikipedia.org/wiki/Haar-like_features
        The video on this pages shows how Haar features and cascades work to located faces:
        - http://dismagazine.com/dystopia/evolved-lifestyles/8115/anti-surveillance-how-to-hide-from-machines/
        """
        storage = cv2.CreateMemStorage(0)

        # lovely.  This segfaults if not present
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
            return FeatureSet(
                    [HaarFeature(self, o, cascade, cv2flag) for o in objects])

        return None

    def draw_circle(self, ctr, rad, color=(0, 0, 0), thickness=1):
        """
        Draw a circle on the image.

        :param ctr: the center of the circle as an (x,y) tuple.
        :param rad: the radius of the circle in pixels
        :param color: a _color tuple (default black)
        :param thickness: the thickness of the circle, -1 means filled in.

        :Example:
        >>> img = Image("lena")
        >>> img.draw_circle((img.width/2,img.height/2),r=50,_color=Color.RED,width=3)
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
        :param color: a _color tuple (default black).
        :param thickness: the thickness of the line in pixels.

        :return: None


        :Example:
        >>> img = Image("lena")
        >>> img.draw_line((0,0),(img.width,img.height),_color=Color.RED,thickness=3)
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
            return cv2.GetSize(self.bitmap)
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
                row.append(
                    self.crop(j * w_ratio, i * h_ratio, w_ratio, h_ratio))
            crops.append(row)

        return crops

    def split_channels(self, grayscale=True):
        """
        Split the channels of an image into RGB (not the default BGR)
        single parameter is whether to return the channels as grey images (default)
        or to return them as tinted _color image
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
        cv2.Split(self.bitmap, b, g, r, None)

        red = self.zeros()
        green = self.zeros()
        blue = self.zeros()

        if grayscale:
            cv2.Merge(r, r, r, None, red)
            cv2.Merge(g, g, g, None, green)
            cv2.Merge(b, b, b, None, blue)
        else:
            cv2.Merge(None, None, r, None, red)
            cv2.Merge(None, g, None, None, green)
            cv2.Merge(b, None, None, None, blue)

        return Image(red), Image(green), Image(blue)

    def merge_channels(self, r=None, g=None, b=None):
        """
        Merge channels is the opposite of split_channels. The image takes one
        image for each of the R,G,B channels and then recombines them into a
        single image. Optionally any of these channels can be None.

        :param r: the r or last channel  of the result PhloxAR Image.
        :param g: the g or center channel of the result PhloxAR Image.
        :param b: the b or first channel of the result PhloxAR Image.
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
            cv2.Zero(r)
        else:
            rt = r.zeros(1)
            cv2.Split(r.bitmap, rt, rt, rt, None)
            r = rt
        if g is None:
            g = self.zeros(1)
            cv2.Zero(g)
        else:
            gt = g.zeros(1)
            cv2.Split(g.bitmap, gt, gt, gt, None)
            g = gt
        if b is None:
            b = self.zeros(1)
            cv2.Zero(b)
        else:
            bt = b.zeros(1)
            cv2.Split(b.bitmap, bt, bt, bt, None)
            b = bt

        ret = self.zeros()
        cv2.Merge(b, g, r, None, ret)
        return Image(ret)

    def apply_hls_curve(self, hcurve, lcurve, scurve):
        """
        Apply a _color correction _curve in HSL space. This method can be used
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
        # TODO CHECK ROI
        # TODO CHECK CURVE SIZE
        # TODO CHECK COLORSPACE
        # TODO CHECK CURVE SIZE
        temp = cv2.CreateImage(self.size(), 8, 3)
        # Move to HLS spacecol
        cv2.cvtColor(self._bitmap, temp, cv2.CV_RGB2HLS)
        tmp_mat = cv2.GetMat(temp)  # convert the bitmap to a matrix
        # now apply the _color _curve correction
        tmp_mat = np.array(self.matrix).copy()
        tmp_mat[:, :, 0] = np.take(hcurve.curve, tmp_mat[:, :, 0])
        tmp_mat[:, :, 1] = np.take(scurve.curve, tmp_mat[:, :, 1])
        tmp_mat[:, :, 2] = np.take(lcurve.curve, tmp_mat[:, :, 2])
        # Now we jimmy the np array into a cvMat
        image = cv2.CreateImageHeader((tmp_mat.shape[1], tmp_mat.shape[0]),
                                     cv2.IPL_DEPTH_8U, 3)
        cv2.SetData(image, tmp_mat.tostring(),
                   tmp_mat.dtype.itemsize * 3 * tmp_mat.shape[1])
        cv2.cvtColor(image, image, cv2.CV_HLS2RGB)
        return Image(image, color_space=self._color_space)

    def apply_rgb_curve(self, rcurve, gcurve, bcurve):
        """
        Apply a _color correction _curve in RGB space. This method can be used
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

        tmp_mat = np.array(self.matrix).copy()
        tmp_mat[:, :, 0] = np.take(bcurve._curve, tmp_mat[:, :, 0])
        tmp_mat[:, :, 1] = np.take(gcurve._curve, tmp_mat[:, :, 1])
        tmp_mat[:, :, 2] = np.take(rcurve._curve, tmp_mat[:, :, 2])
        # Now we jimmy the np array into a cvMat
        image = cv2.CreateImageHeader((tmp_mat.shape[1], tmp_mat.shape[0]),
                                     cv2.IPL_DEPTH_8U, 3)
        cv2.SetData(image, tmp_mat.tostring(),
                   tmp_mat.dtype.itemsize * 3 * tmp_mat.shape[1])
        return Image(image, color_space=self._color_space)

    def apply_intensity_curve(self, curve):
        """
        Intensity applied to all three _color channels

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
        pixels = np.array(self.narray).reshape(-1, 3)
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
        hue_channel = np.cast['int'](vsh_matrix[:, 2])

        if color_hue < 90:
            hue_loop = 180
        else:
            hue_loop = -180
        # set whether we need to move back or forward on the hue circle

        distances = np.minimum(np.abs(hue_channel - color_hue),
                                np.abs(hue_channel - (color_hue + hue_loop)))
        # take the minimum distance for each pixel


        distances = np.where(np.logical_and(vsh_matrix[:, 0] > minvalue,
                                              vsh_matrix[:, 1] > minsaturation),
                              distances * (255.0 / 90.0),
                              # normalize 0 - 90 -> 0 - 255
                              255.0)  # use the maxvalue if it false outside of our value/saturation tolerances

        return Image(distances.reshape(self.width, self.height))

    def erode(self, iterations=1, kernelsize=3):
        ret = self.zeros()
        kern = cv2.CreateStructuringElementEx(kernelsize, kernelsize, 1, 1,
                                             cv2.CV_SHAPE_RECT)
        cv2.Erode(self.bitmap, ret, kern, iterations)
        return Image(ret, color_space=self._color_space)

    def dilate(self, iteration=1):
        ret = self.zeros()
        kern = cv2.CreateStructuringElementEx(3, 3, 1, 1, cv2.CV_SHAPE_RECT)
        cv2.Dilate(self.bitmap, ret, kern, iteration)
        return Image(ret, color_space=self._color_space)

    def morph_open(self):
        ret = self.zeros()
        temp = self.zeros()
        kern = cv2.CreateStructuringElementEx(3, 3, 1, 1, cv2.CV_SHAPE_RECT)
        try:
            cv2.MorphologyEx(self.bitmap, ret, temp, kern, cv2.MORPH_OPEN, 1)
        except:
            cv2.MorphologyEx(self.bitmap, ret, temp, kern, cv2.CV_MOP_OPEN, 1)
            # OPENCV 2.2 vs 2.3 compatability

        return Image(ret)

    def morph_close(self):
        ret = self.zeros()
        temp = self.zeros()
        kern = cv2.CreateStructuringElementEx(3, 3, 1, 1, cv2.CV_SHAPE_RECT)
        try:
            cv2.MorphologyEx(self.bitmap, ret, temp, kern, cv2.MORPH_CLOSE, 1)
        except:
            cv2.MorphologyEx(self.bitmap, ret, temp, kern, cv2.CV_MOP_CLOSE, 1)
            # OPENCV 2.2 vs 2.3 compatability

        return Image(ret, color_space=self._color_space)

    def morph_gradient(self):
        ret = self.zeros()
        temp = self.zeros()
        kern = cv2.CreateStructuringElementEx(3, 3, 1, 1, cv2.CV_SHAPE_RECT)
        try:
            cv2.MorphologyEx(self.bitmap, ret, temp, kern, cv2.MORPH_GRADIENT, 1)
        except:
            cv2.MorphologyEx(self.bitmap, ret, temp, kern, cv2.CV_MOP_GRADIENT, 1)
        return Image(ret, color_space=self._color_space)

    def histogram(self, bins=50):
        """
        Return a numpy array of the 1D histogram of intensity for pixels in the
        image.
        :param bins: integer number of bins in a histogram
        :return: a list of histogram bin values
        """
        gray = self._get_gray_narray()

        hist, bin_edges = np.histogram(np.asarray(cv2.GetMat(gray)), bins=bins)

        return hist.tolist()

    def hue_histogram(self, bins=179, dynamic_range=True):
        """
        Returns the histogram of the hue channel for the image.
        :param bins: integer number of bins in a histogram
        :param dynamic_range:
        :return:
        """
        if dynamic_range:
            return np.histogram(self.to_hsv().narray[:, :, 2], bins=bins)[0]
        else:
            return np.histogram(self.to_hsv().narray[:, :, 2], bins=bins,
                                 range=(0.0, 360.0))[0]

    def hue_peaks(self, bins=179):
        y_axis, x_axis = np.histogram(self.to_hsv().narray[:, :, 2],
                                       bins=bins)
        x_axis = x_axis[0:bins]
        lookahead = int(bins / 17)
        delta = 0

        maxtab = []
        mintab = []
        dump = []  # Used to pop the first hit which always if false

        length = len(y_axis)
        if x_axis is None:
            x_axis = range(length)

        # perform some checks
        if length != len(x_axis):
            raise ValueError, "Input vectors y_axis and x_axis must have same length"
        if lookahead < 1:
            raise ValueError, "Lookahead must be above '1' in value"
        if not (np.isscalar(delta) and delta >= 0):
            raise ValueError, "delta must be a positive number"

        # needs to be a numpy array
        y_axis = np.asarray(y_axis)

        # maxima and minima candidates are temporarily stored in
        # mx and mn respectively
        mn, mx = np.Inf, -np.Inf

        # Only detect peak if there is 'lookahead' amount of points after it
        for index, (x, y) in enumerate(
                zip(x_axis[:-lookahead], y_axis[:-lookahead])):
            if y > mx:
                mx = y
                mxpos = x
            if y < mn:
                mn = y
                mnpos = x

            ####look for max####
            if y < mx - delta and mx != np.Inf:
                # Maxima peak candidate found
                # look ahead in signal to ensure that this is a peak and not jitter
                if y_axis[index:index + lookahead].max() < mx:
                    maxtab.append((mxpos, mx))
                    dump.append(True)
                    # set algorithm to only find minima now
                    mx = np.Inf
                    mn = np.Inf

            ####look for min####
            if y > mn + delta and mn != -np.Inf:
                # Minima peak candidate found
                # look ahead in signal to ensure that this is a peak and not jitter
                if y_axis[index:index + lookahead].min() > mn:
                    mintab.append((mnpos, mn))
                    dump.append(False)
                    # set algorithm to only find maxima now
                    mn = -np.Inf
                    mx = -np.Inf

        # Remove the false hit on the first value of the y_axis
        try:
            if dump[0]:
                maxtab.pop(0)
                # print "pop max"
            else:
                mintab.pop(0)
                # print "pop min"
            del dump
        except IndexError:
            # no peaks were found, should the function return empty lists?
            pass

        huetab = []
        for hue, pixelcount in maxtab:
            huetab.append((hue, pixelcount / float(self.width * self.height)))
        return huetab

    def __getitem__(self, key):
        ret = self.matrix[tuple(reversed(key))]
        if isinstance(ret, cv2.cvmat):
            (width, height) = cv2.GetSize(ret)
            newmat = cv2.CreateMat(height, width, ret.type)
            cv2.Copy(ret, newmat)  # this seems to be a bug in opencv
            # if you don't copy the matrix slice, when you convert to bmp you get
            # a slice-sized hunk starting at 0, 0
            return Image(newmat)

        if self.is_bgr():
            return tuple(reversed(ret))
        else:
            return tuple(ret)

    def __setitem__(self, key, value):
        value = tuple(reversed(value))  # RGB -> BGR

        if isinstance(key[0], slice):
            cv2.Set(self.matrix[tuple(reversed(key))], value)
            self._clear_buffers("_matrix")
        else:
            self.matrix[tuple(reversed(key))] = value
            self._clear_buffers("_matrix")

    def __sub__(self, other):
        bitmap = self.zeros()
        if isnum(other):
            cv2.SubS(self.bitmap, cv2.Scalar(other, other, other), bitmap)
        else:
            cv2.Sub(self.bitmap, other.bitmap, bitmap)
        return Image(bitmap, color_space=self._color_space)

    def __add__(self, other):
        bitmap = self.zeros()
        if isnum(other):
            cv2.AddS(self.bitmap, cv2.Scalar(other, other, other), bitmap)
        else:
            cv2.Add(self.bitmap, other.bitmap, bitmap)
        return Image(bitmap, color_space=self._color_space)

    def __and__(self, other):
        bitmap = self.zeros()
        if isnum(other):
            cv2.AndS(self.bitmap, cv2.Scalar(other, other, other), bitmap)
        else:
            cv2.And(self.bitmap, other.bitmap, bitmap)
        return Image(bitmap, color_space=self._color_space)

    def __or__(self, other):
        bitmap = self.zeros()
        if isnum(other):
            cv2.OrS(self.bitmap, cv2.Scalar(other, other, other), bitmap)
        else:
            cv2.Or(self.bitmap, other.bitmap, bitmap)
        return Image(bitmap, color_space=self._color_space)

    def __div__(self, other):
        bitmap = self.zeros()
        if not isnum(other):
            cv2.Div(self.bitmap, other.bitmap, bitmap)
        else:
            cv2.ConvertScale(self.bitmap, bitmap, 1.0 / float(other))
        return Image(bitmap, color_space=self._color_space)

    def __mul__(self, other):
        bitmap = self.zeros()
        if not isnum(other):
            cv2.Mul(self.bitmap, other.bitmap, bitmap)
        else:
            cv2.ConvertScale(self.bitmap, bitmap, float(other))
        return Image(bitmap, color_space=self._color_space)

    def __pow__(self, power, modulo=None):
        bitmap = self.zeros()
        cv2.Pow(self.bitmap, bitmap, power)
        return Image(bitmap, color_space=self._color_space)

    def __neg__(self):
        bitmap = self.zeros()
        cv2.Not(self.bitmap, bitmap)
        return Image(bitmap, color_space=self._color_space)

    def __invert__(self):
        return self.invert()

    def max(self, other):
        bitmap = self.zeros()
        if isnum(other):
            cv2.MaxS(self.bitmap, other, bitmap)
        else:
            if self.size() != other.size():
                warnings.warn(
                        "Both images should have same sizes. Returning None.")
                return None
            cv2.Max(self.bitmap, other.bitmap, bitmap)
        return Image(bitmap, color_space=self._color_space)

    def min(self, other):
        bitmap = self.zeros()
        if isnum(other):
            cv2.MinS(self.bitmap, other, bitmap)
        else:
            if self.size() != other.size():
                warnings.warn("Both images should have same sizes. Returning "
                              "None.")
                return None
            cv2.Min(self.bitmap, other.bitmap, bitmap)
        return Image(bitmap, color_space=self._color_space)

    def _clear_buffers(self, clearexcept='_bitmap'):
        for k, v in self._initialized_buffers.items():
            if k == clearexcept:
                continue
            self.__dict__[k] = v

    def find_barcode(self, dozlib=True, zxing_path=''):
        if doZLib:
            try:
                import zbar
            except:
                logger.warning('The zbar library is not installed, please '
                               'install to read barcodes')
                return None

            # configure zbar
            scanner = zbar.ImageScanner()
            scanner.parse_config('enable')
            raw = self.pilimg.convert('L').tostring()
            width = self.width
            height = self.height

            # wrap image data
            image = zbar.Image(width, height, 'Y800', raw)

            # scan the image for barcodes
            scanner.scan(image)
            barcode = None
            # extract results
            for symbol in image:
                # do something useful with results
                barcode = symbol
            # clean up
            del (image)

        else:
            if not ZXING_ENABLED:
                warnings.warn("Zebra Crossing (ZXing) Library not installed. "
                              "Please see the release notes.")
                return None

            if not self._barcode_reader:
                if not zxing_path:
                    self._barcode_reader = zxing.BarCodeReader()
                else:
                    self._barcode_reader = zxing.BarCodeReader(zxing_path)

            tmp_filename = os.tmpnam() + ".png"
            self.save(tmp_filename)
            barcode = self._barcode_reader.decode(tmp_filename)
            os.unlink(tmp_filename)

        if barcode:
            f = Barcode(self, barcode)
            return FeatureSet([f])
        else:
            return None

    def find_lines(self, threshold=80, minlinelen=30, maxlinegap=10,
                   cannyth1=50, cannyth2=100, standard=False, nlines=-1,
                   maxpixelgap=1):
        em = self._get_edge_map(cannyth1, cannyth2)

        linesFS = FeatureSet()

        if standard:
            lines = cv2.HoughLines2(em, cv2.CreateMemStorage(),
                                   cv2.CV_HOUGH_STANDARD, 1.0, cv2.CV_PI / 180.0,
                                   threshold, minlinelen, maxlinegap)
            if nlines == -1:
                nlines = len(lines)
            # All white points (edges) in Canny edge image
            em = Image(em)
            x, y = np.where(em.gray_narray > 128)
            # Put points in dictionary for fast checkout if point is white
            pts = dict((p, 1) for p in zip(x, y))

            w, h = self.width - 1, self.height - 1
            for rho, theta in lines[:nlines]:
                ep = []
                ls = []
                a = math.cos(theta)
                b = math.sin(theta)
                # Find endpoints of line on the image's edges
                if round(b, 4) == 0:  # slope of the line is infinity
                    ep.append((int(round(abs(rho))), 0))
                    ep.append((int(round(abs(rho))), h))
                elif round(a, 4) == 0:  # slope of the line is zero
                    ep.append((0, int(round(abs(rho)))))
                    ep.append((w, int(round(abs(rho)))))
                else:
                    # top edge
                    x = rho / float(a)
                    if 0 <= x <= w:
                        ep.append((int(round(x)), 0))
                    # bottom edge
                    x = (rho - h * b) / float(a)
                    if 0 <= x <= w:
                        ep.append((int(round(x)), h))
                    # left edge
                    y = rho / float(b)
                    if 0 <= y <= h:
                        ep.append((0, int(round(y))))
                    # right edge
                    y = (rho - w * a) / float(b)
                    if 0 <= y <= h:
                        ep.append((w, int(round(y))))

                # remove duplicates if line crosses the image at corners
                ep = list(set(ep))
                ep.sort()
                brl = self.bresenham_line(ep[0], ep[1])

                # Follow the points on Bresenham's line. Look for white points.
                # If the distance between two adjacent white points (dist) is
                # less than or equal maxpixelgap then consider them the same
                # line. If dist is bigger maxpixelgap then check if length of
                # the line is bigger than minlinelength.
                # If so then add line.

                # distance between two adjacent white points
                dist = float('inf')
                len_l = float('-inf')  # length of the line
                for p in brl:
                    if p in pts:
                        # found the end of the previous line and the start of the new line
                        if dist > maxpixelgap:
                            if len_l >= minlinelen:
                                if ls:
                                    # If the gap between current line and previous
                                    # is less than maxlinegap then merge this lines
                                    l = ls[-1]
                                    gap = round(math.sqrt(
                                            (start_p[0] - l[1][0]) ** 2 + (
                                                start_p[1] - l[1][1]) ** 2))
                                    if gap <= maxlinegap:
                                        ls.pop()
                                        start_p = l[0]
                                ls.append((start_p, last_p))
                            # First white point of the new line found
                            dist = 1
                            len_l = 1
                            start_p = p  # first endpoint of the line
                        else:
                            # dist is less than or equal maxpixelgap, so line
                            # doesn't end yet
                            len_l += dist
                            dist = 1
                        last_p = p  # last white point
                    else:
                        dist += 1

                for l in ls:
                    linesFS.append(Line(self, l))
            linesFS = linesFS[:nlines]
        else:
            lines = cv2.HoughLines2(em, cv2.CreateMemStorage(),
                                   cv2.CV_HOUGH_PROBABILISTIC, 1.0,
                                   cv2.CV_PI / 180.0, threshold, minlinelen,
                                   maxlinegap)
            if nlines == -1:
                nlines = len(lines)

            for l in lines[:nlines]:
                linesFS.append(Line(self, l))

        return linesFS

    def find_chessboard(self, dimensions=(8, 5), subpixel=True):
        corners = cv2.FindChessboardCorners(self._equalized_gray_bitmap(),
                                           dimensions,
                                           cv2.CV_CALIB_CB_ADAPTIVE_THRESH + cv2.CV_CALIB_CB_NORMALIZE_IMAGE)
        if len(corners[1]) == dimensions[0] * dimensions[1]:
            if subpixel:
                spCorners = cv2.FindCornerSubPix(self.gray_matrix,
                                                corners[1], (11, 11), (-1, -1),
                                                (
                                                    cv2.CV_TERMCRIT_ITER | cv2.CV_TERMCRIT_EPS,
                                                    10, 0.01))
            else:
                spCorners = corners[1]
            return FeatureSet([Chessboard(self, dimensions, spCorners)])
        else:
            return None

    def edges(self, t1=50, t2=100):
        return Image(self._get_edge_map(t1, t2), color_space=self._color_space)

    def _get_edge_map(self, t1=50, t2=100):
        if (self._edge_map and self._canny_param[0] == t1 and
                self._canny_param[1] == t2):
            return self._edge_map

        self._edge_map = self.zeros(1)
        cv2.Canny(self._get_gray_narray(), self._edge_map, t1, t2)
        self._canny_param = (t1, t2)

        return self._edge_map

    def rotate(self, angle, fixed=True, point=None, scale=1.0):
        if point[0] == -1 or point[1] == -1:
            point[0] = (self.width - 1) / 2
            point[1] = (self.height - 1) / 2

        if fixed:
            ret = self.zeros()
            cv2.Zero(ret)
            rot_mat = cv2.CreateMat(2, 3, cv2.CV_32FC1y)
            cv2.GetRotationMatrix2D((float(point[0]), float(point[1])),
                                   float(angle), float(scale), rot_mat)
            cv2.WarpAffine(self.bitmap, ret, rot_mat)
            return Image(ret, color_space=self._color_space)

        # otherwise, we're expanding the matrix to fit the image at original size
        rot_mat = cv2.CreateMat(2, 3, cv2.CV_32FC1)
        # first we create what we thing the rotation matrix should be
        cv2.GetRotationMatrix2D((float(point[0]), float(point[1])), float(angle),
                               float(scale), rot_mat)
        A = np.array([0, 0, 1])
        B = np.array([self.width, 0, 1])
        C = np.array([self.width, self.height, 1])
        D = np.array([0, self.height, 1])
        # So we have defined our image ABC in homogenous coordinates
        # and apply the rotation so we can figure out the image size
        a = np.dot(rot_mat, A)
        b = np.dot(rot_mat, B)
        c = np.dot(rot_mat, C)
        d = np.dot(rot_mat, D)
        # I am not sure about this but I think the a/b/c/d are transposed
        # now we calculate the extents of the rotated components.
        minY = min(a[1], b[1], c[1], d[1])
        minX = min(a[0], b[0], c[0], d[0])
        maxY = max(a[1], b[1], c[1], d[1])
        maxX = max(a[0], b[0], c[0], d[0])
        # from the extents we calculate the new size
        newWidth = np.ceil(maxX - minX)
        newHeight = np.ceil(maxY - minY)
        # now we calculate a new translation
        tX = 0
        tY = 0
        # calculate the translation that will get us centered in the new image
        if minX < 0:
            tX = -1.0 * minX
        elif maxX > newWidth - 1:
            tX = -1.0 * (maxX - newWidth)

        if minY < 0:
            tY = -1.0 * minY
        elif maxY > newHeight - 1:
            tY = -1.0 * (maxY - newHeight)

        # now we construct an affine map that will the rotation and scaling we want with the
        # the corners all lined up nicely with the output image.
        src = ((A[0], A[1]), (B[0], B[1]), (C[0], C[1]))
        dst = (
            (a[0] + tX, a[1] + tY), (b[0] + tX, b[1] + tY),
            (c[0] + tX, c[1] + tY))

        cv2.GetAffineTransform(src, dst, rot_mat)

        # calculate the translation of the corners to center the image
        # use these new corner positions as the input to cvGetAffineTransform
        ret = cv2.CreateImage((int(newWidth), int(newHeight)), 8, int(3))
        cv2.Zero(ret)

        cv2.WarpAffine(self.bitmap, ret, rot_mat)
        # cv2.AddS(ret,(0,255,0),ret)
        return Image(ret, color_space=self._color_space)

    def transpose(self):
        ret = cv2.CreateImage((self.height, self.width), cv2.IPL_DEPTH_8U, 3)
        cv2.Transpose(self.bitmap, ret)
        return Image(ret, color_space=self._color_space)

    def shear(self, cornerpoints):
        src = ((0, 0), (self.width - 1, 0), (self.width - 1, self.height - 1))
        # set the original points
        warp = cv2.CreateMat(2, 3, cv2.CV_32FC1)
        # create the empty warp matrix
        cv2.GetAffineTransform(src, cornerpoints, warp)

        return self.transform_affine(warp)

    def transform_affine(self, rot_matrix):
        ret = self.zeros()
        if isinstance(rot_matrix, np.ndarray):
            rot_matrix = npArray2cvMat(rot_matrix)
        cv2.WarpAffine(self.bitmap, ret, rot_matrix)
        return Image(ret, color_space=self._color_space)

    def warp(self, cornerpoints):
        src = ((0, 0), (self.width - 1, 0), (self.width - 1, self.height - 1),
               (0, self.height - 1))
        warp = cv2.CreateMat(3, 3, cv2.CV_32FC1)  # create an empty 3x3 matrix
        # figure out the warp matrix
        cv2.GetPerspectiveTransform(src, cornerpoints, warp)

        return self.transform_perspective(warp)

    def transform_perspective(self, rot_matrix):
        if isinstance(rot_matrix, np.ndarray):
            rot_matrix = np.array(rot_matrix)
        ret = cv2.warpPerspective(src=np.array(self.matrix),
                                  dsize=(self.width, self.height),
                                  M=rot_matrix, flags=cv2.INTER_CUBIC)
        return Image(ret, color_space=self._color_space, cv2image=True)

    def get_pixel(self, x, y):
        c = None
        ret = None
        if x < 0 or x >= self.width:
            logger.warning("getRGBPixel: X value is not valid.")
        elif y < 0 or y >= self.height:
            logger.warning("getRGBPixel: Y value is not valid.")
        else:
            c = cv2.Get2D(self.bitmap, y, x)
            if self._color_space == ColorSpace.BGR:
                ret = (c[2], c[1], c[0])
            else:
                ret = (c[0], c[1], c[2])

        return ret

    def get_gray_pixel(self, x, y):
        ret = None
        if x < 0 or x >= self.width:
            logger.warning("getGrayPixel: X value is not valid.")
        elif y < 0 or y >= self.height:
            logger.warning("getGrayPixel: Y value is not valid.")
        else:
            ret = cv2.Get2D(self._get_gray_narray(), y, x)
            ret = ret[0]
        return ret

    def get_vert_scanline(self, col):
        ret = None
        if col < 0 or col >= self.width:
            logger.warning("get_vert_scanline: col value is not valid.")
        else:
            ret = cv2.GetCol(self.bitmap, col)
            ret = np.array(ret)
            ret = ret[:, 0, :]
        return ret

    def get_horz_scanline(self, row):
        ret = None
        if row < 0 or row >= self.height:
            logger.warning("get_horz_scanline: row value is not valid.")
        else:
            ret = cv2.GetRow(self.bitmap, row)
            ret = np.array(ret)
            ret = ret[0, :, :]
        return ret

    def get_vert_scanline_gray(self, col):
        ret = None
        if col < 0 or col >= self.width:
            logger.warning("get_horz_scanline: row value is not valid.")
        else:
            ret = cv2.GetCol(self._get_gray_narray(), col)
            ret = np.array(ret)
        return ret

    def get_horz_scanline_gray(self, row):
        ret = None
        if row < 0 or row >= self.height:
            logger.warning("get_horz_scanline: row value is not valid.")
        else:
            ret = cv2.GetRow(self._get_gray_narray(), row)
            ret = np.array(ret)
            ret = ret.transpose()
        return ret

    def crop(self, x=None, y=None, w=None, h=None, centered=False, smart=False):
        if smart:
            if x > self.width:
                x = self.width
            elif x < 0:
                x = 0
            elif y > self.height:
                y = self.height
            elif y < 0:
                y = 0
            elif (x + w) > self.width:
                w = self.width - x
            elif (y + h) > self.height:
                h = self.height - y

        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(y, np.ndarray):
            y = y.tolist()

            # If it's a feature extract what we need
        if isinstance(x, Feature):
            theFeature = x
            x = theFeature.points[0][0]
            y = theFeature.points[0][1]
            w = theFeature.width()
            h = theFeature.height()

        elif (isinstance(x, (tuple, list)) and len(x) == 4 and
                  isinstance(x[0], (int, long, float))
              and y is None and w is None and h is None):
            x, y, w, h = x
            # x of the form [(x,y),(x1,y1),(x2,y2),(x3,y3)]
            # x of the form [[x,y],[x1,y1],[x2,y2],[x3,y3]]
            # x of the form ([x,y],[x1,y1],[x2,y2],[x3,y3])
            # x of the form ((x,y),(x1,y1),(x2,y2),(x3,y3))
            # x of the form (x,y,x1,y2) or [x,y,x1,y2]
        elif (isinstance(x, (list, tuple)) and
                  isinstance(x[0], (list, tuple)) and
                  (len(x) == 4 and len(x[0]) == 2) and
                      y is None and w is None and h is None):
            if (len(x[0]) == 2 and len(x[1]) == 2 and len(x[2]) == 2 and
                        len(x[3]) == 2):
                xmax = np.max([x[0][0], x[1][0], x[2][0], x[3][0]])
                ymax = np.max([x[0][1], x[1][1], x[2][1], x[3][1]])
                xmin = np.min([x[0][0], x[1][0], x[2][0], x[3][0]])
                ymin = np.min([x[0][1], x[1][1], x[2][1], x[3][1]])
                x = xmin
                y = ymin
                w = xmax - xmin
                h = ymax - ymin
            else:
                logger.warning(
                        "x should be in the form  ((x,y),(x1,y1),(x2,y2),(x3,y3))")
                return None

                # x,y of the form [x1,x2,x3,x4,x5....] and y similar
        elif (isinstance(x, (tuple, list)) and
                  isinstance(y, (tuple, list)) and
                      len(x) > 4 and len(y) > 4):
            if (isinstance(x[0], (int, long, float)) and
                    isinstance(y[0], (int, long, float))):
                xmax = np.max(x)
                ymax = np.max(y)
                xmin = np.min(x)
                ymin = np.min(y)
                x = xmin
                y = ymin
                w = xmax - xmin
                h = ymax - ymin
            else:
                logger.warning(
                        "x should be in the form x = [1,2,3,4,5] y =[0,2,4,6,8]")
                return None

                # x of the form [(x,y),(x,y),(x,y),(x,y),(x,y),(x,y)]
        elif (isinstance(x, (list, tuple)) and
                      len(x) > 4 and len(
                x[0]) == 2 and y is None and w is None and h is None):
            if isinstance(x[0][0], (int, long, float)):
                xs = [pt[0] for pt in x]
                ys = [pt[1] for pt in x]
                xmax = np.max(xs)
                ymax = np.max(ys)
                xmin = np.min(xs)
                ymin = np.min(ys)
                x = xmin
                y = ymin
                w = xmax - xmin
                h = ymax - ymin
            else:
                logger.warning(
                        "x should be in the form [(x,y),(x,y),(x,y),(x,y),(x,y),(x,y)]")
                return None

                # x of the form [(x,y),(x1,y1)]
        elif (isinstance(x, (list, tuple)) and len(x) == 2 and
                  isinstance(x[0], (list, tuple)) and
                  isinstance(x[1], (list, tuple)) and
                      y is None and w is None and h is None):
            if len(x[0]) == 2 and len(x[1]) == 2:
                xt = np.min([x[0][0], x[1][0]])
                yt = np.min([x[0][0], x[1][0]])
                w = np.abs(x[0][0] - x[1][0])
                h = np.abs(x[0][1] - x[1][1])
                x = xt
                y = yt
            else:
                logger.warning("x should be in the form [(x1,y1),(x2,y2)]")
                return None

                # x and y of the form (x,y),(x1,y2)
        elif (isinstance(x, (tuple, list)) and isinstance(y, (tuple, list))
              and w is None and h is None):
            if len(x) == 2 and len(y) == 2:
                xt = np.min([x[0], y[0]])
                yt = np.min([x[1], y[1]])
                w = np.abs(y[0] - x[0])
                h = np.abs(y[1] - x[1])
                x = xt
                y = yt

            else:
                logger.warning(
                        "if x and y are tuple it should be in the form (x1,y1) and (x2,y2)")
                return None

        if y is None or w is None or h is None:
            print("Please provide an x, y, width, height to function")

        if w <= 0 or h <= 0:
            logger.warning("Can't do a negative crop!")
            return None

        ret = cv2.CreateImage((int(w), int(h)), cv2.IPL_DEPTH_8U, 3)
        if x < 0 or y < 0:
            logger.warning(
                    "Crop will try to help you, but you have a negative crop position, your width and height may not be what you want them to be.")

        if centered:
            rectangle = (int(x - (w / 2)), int(y - (h / 2)), int(w), int(h))
        else:
            rectangle = (int(x), int(y), int(w), int(h))

        (topROI, bottomROI) = self. _rect_overlap_rois(
                (rectangle[2], rectangle[3]), (self.width, self.height),
                (rectangle[0], rectangle[1]))

        if bottomROI is None:
            logger.warning(
                    "Hi, your crop rectangle doesn't even overlap your image. I have no choice but to return None.")
            return None

        ret = np.zeros((bottomROI[3], bottomROI[2], 3), dtype='uint8')

        ret = self.cvnarray[bottomROI[1]:bottomROI[1] + bottomROI[3],
              bottomROI[0]:bottomROI[0] + bottomROI[2], :]

        img = Image(ret, color_space=self._color_space, cv2image=True)

        # Buffering the top left point (x, y) in a image.
        img._uncropped_x = self._uncropped_x + int(x)
        img._uncropped_y = self._uncropped_y + int(y)
        return img

    def region_select(self, x1, y1, x2, y2):
        w = abs(x1 - x2)
        h = abs(y1 - y2)

        ret = None
        if w <= 0 or h <= 0 or w > self.width or h > self.height:
            logger.warning("regionSelect: the given values will not fit in the "
                           "image or are too small.")
        else:
            xf = x2
            if x1 < x2:
                xf = x1
            yf = y2
            if y1 < y2:
                yf = y1
            ret = self.crop(xf, yf, w, h)

        return ret

    def clear(self):
        cv2.SetZero(self._bitmap)
        self._clear_buffers()

    def draw(self, features, color=Color.GREEN, width=1, autocolor=False):
        if type(features) == type(self):
            warnings.warn("You need to pass drawable features.")
            return None
        if hasattr(features, 'draw'):
            from copy import deepcopy
            if isinstance(features, FeatureSet):
                cfeatures = deepcopy(features)
                for cfeat in cfeatures:
                    cfeat.image = self
                cfeatures.draw(color, width, autocolor)
            else:
                cfeatures = deepcopy(features)
                cfeatures.image = self
                cfeatures.draw(color, width)
        else:
            warnings.warn("You need to pass drawable features.")
        return None

    def draw_text(self, text='', x=None, y=None, color=Color.BLUE, fontsize=16):
        if x is None:
            x = self.width / 2
        if y is None:
            y = self.height / 2

        self.get_drawing_layer().setFontSize(fontsize)
        self.get_drawing_layer().text(text, (x, y), color)

    def draw_rect(self, x, y, w, h, color=Color.RED, width=1, alpha=255):
        if width < 1:
            self.get_drawing_layer().rectangle((x, y), (w, h), color, filled=True,
                                               alpha=alpha)
        else:
            self.get_drawing_layer().rectangle((x, y), (w, h), color, width,
                                               alpha=alpha)

    def draw_rotated_rect(self, bbox, color=Color.RED, width=1):
        cv2.EllipseBox(self.bitmap, box=bbox, color=color, thicness=width)

    def show(self, type='window'):
        if type == 'browser':
            import webbrowser
            js = JpegStreamer(8080)
            self.save(js)
            webbrowser.open("http://localhost:8080", 2)
            return js
        elif type == 'window':
            if init_options_handler.on_notebook:
                d = Display(displaytype='notebook')
            else:
                d = Display(self.size())
            self.save(d)
            return d
        else:
            print("Unknown type to show")

    def _surface2image(self, surface):
        imgarray = sdl2.surfarray.array3d(surface)
        ret = Image(imgarray)
        ret._color_space = ColorSpace.RGB
        return ret.to_bgr().transpose()

    def _image2surface(self, img):
        return sdl2.image.fromstring(img.pilimg.tostring(), img.size(), "RGB")

    def to_pygame_surface(self):
        return sdl2.image.fromstring(self.pilimg.tostring(), self.size(), "RGB")

    def add_drawing_layer(self, layer=None):
        if not isinstance(layer, DrawingLayer):
            return "Please pass a DrawingLayer object"

        if not layer:
            layer = DrawingLayer(self.size())
        self._layers.append(layer)
        return len(self._layers) - 1

    def insert_drawing_layer(self, layer, index):
        self._layers.insert(index, layer)
        return None

    def remove_drawing_layer(self, index=-1):
        try:
            return self._layers.pop(index)
        except IndexError:
            print('Not a valid index or No layers to remove!')

    def get_drawing_layer(self, index=-1):
        if not len(self._layers):
            layer = DrawingLayer(self.size())
            self.add_drawing_layer(layer)
        try:
            return self._layers[index]
        except IndexError:
            print('Not a valid index')

    def dl(self, index=-1):
        return self.get_drawing_layer(index)

    def clear_layers(self):
        for i in self._layers:
            self._layers.remove(i)

        return None

    def layers(self):
        return self._layers

    def _render_image(self, layer):
        img_surf = self.surface.copy()
        img_surf.blit(layer, (0, 0))
        return Image(img_surf)

    def _render_layers(self):
        pass

    def merged_layers(self):
        final = DrawingLayer(self.size())
        for layers in self._layers:  # compose all the layers
            layers.renderToOtherLayer(final)
        return final

    def apply_layers(self, indicies=-1):
        if not len(self._layers):
            return self

        if indicies == -1 and len(self._layers) > 0:
            final = self.merged_layers()
            img_surf = self.surface.copy()
            img_surf.blit(final, (0, 0))
            return Image(img_surf)
        else:
            final = DrawingLayer((self.width, self.height))
            ret = self
            indicies.reverse()
            for idx in indicies:
                ret = self._layers[idx].renderToOtherLayer(final)
            img_surf = self.surface.copy()
            img_surf.blit(final, (0, 0))
            indicies.reverse()
            return Image(img_surf)

    def adaptive_scale(self, resolution, fit=True):
        wndwAR = float(resolution[0]) / float(resolution[1])
        imgAR = float(self.width) / float(self.height)
        img = self
        targx = 0
        targy = 0
        targw = resolution[0]
        targh = resolution[1]
        if self.size() == resolution:  # we have to resize
            ret = self
        elif imgAR == wndwAR and fit:
            ret = img.scale(resolution[0], resolution[1])
            return ret
        elif fit:
            # scale factors
            ret = np.zeros((resolution[1], resolution[0], 3), dtype='uint8')
            wscale = (float(self.width) / float(resolution[0]))
            hscale = (float(self.height) / float(resolution[1]))
            if wscale > 1:  # we're shrinking what is the percent reduction
                wscale = 1 - (1.0 / wscale)
            else:  # we need to grow the image by a percentage
                wscale = 1.0 - wscale
            if hscale > 1:
                hscale = 1 - (1.0 / hscale)
            else:
                hscale = 1.0 - hscale
            if wscale == 0:  # if we can get away with not scaling do that
                targx = 0
                targy = (resolution[1] - self.height) / 2
                targw = img.width
                targh = img.height
            elif hscale == 0:  # if we can get away with not scaling do that
                targx = (resolution[0] - img.width) / 2
                targy = 0
                targw = img.width
                targh = img.height
            elif wscale < hscale:  # the width has less distortion
                sfactor = float(resolution[0]) / float(self.width)
                targw = int(float(self.width) * sfactor)
                targh = int(float(self.height) * sfactor)
                if targw > resolution[0] or targh > resolution[1]:
                    # aw shucks that still didn't work do the other way instead
                    sfactor = float(resolution[1]) / float(self.height)
                    targw = int(float(self.width) * sfactor)
                    targh = int(float(self.height) * sfactor)
                    targx = (resolution[0] - targw) / 2
                    targy = 0
                else:
                    targx = 0
                    targy = (resolution[1] - targh) / 2
                img = img.scale(targw, targh)
            else:  # the height has more distortion
                sfactor = float(resolution[1]) / float(self.height)
                targw = int(float(self.width) * sfactor)
                targh = int(float(self.height) * sfactor)
                if targw > resolution[0] or targh > resolution[1]:
                    # aw shucks that still didn't work do the other way instead
                    sfactor = float(resolution[0]) / float(self.width)
                    targw = int(float(self.width) * sfactor)
                    targh = int(float(self.height) * sfactor)
                    targx = 0
                    targy = (resolution[1] - targh) / 2
                else:
                    targx = (resolution[0] - targw) / 2
                    targy = 0
                img = img.scale(targw, targh)

        else:  # we're going to crop instead
            # center a too small image
            if (self.width <= resolution[0] and self.height <= resolution[1]):
                # we're too small just center the thing
                ret = np.zeros((resolution[1], resolution[0], 3),
                               dtype='uint8')
                targx = (resolution[0] / 2) - (self.width / 2)
                targy = (resolution[1] / 2) - (self.height / 2)
                targh = self.height
                targw = self.width
            elif (self.width > resolution[0] and self.height > resolution[
                1]):  # crop too big on both axes
                targw = resolution[0]
                targh = resolution[1]
                targx = 0
                targy = 0
                x = (self.width - resolution[0]) / 2
                y = (self.height - resolution[1]) / 2
                img = img.crop(x, y, targw, targh)
                return img
            elif (self.width <= resolution[0] and self.height > resolution[
                1]):  # height too big
                # crop along the y dimension and center along the x dimension
                ret = np.zeros((resolution[1], resolution[0], 3),
                               dtype='uint8')
                targw = self.width
                targh = resolution[1]
                targx = (resolution[0] - self.width) / 2
                targy = 0
                x = 0
                y = (self.height - resolution[1]) / 2
                img = img.crop(x, y, targw, targh)

            elif (self.width > resolution[0] and self.height <= resolution[
                1]):  # width too big
                # crop along the y dimension and center along the x dimension
                ret = np.zeros((resolution[1], resolution[0], 3),
                                dtype='uint8')
                targw = resolution[0]
                targh = self.height
                targx = 0
                targy = (resolution[1] - self.height) / 2
                x = (self.width - resolution[0]) / 2
                y = 0
                img = img.crop(x, y, targw, targh)

        ret[targy:targy + targh, targx:targx + targw, :] = img.cvnarray
        ret = Image(ret, cv2image=True)
        return ret

    def blit(self, img, pos=None, alpha=None, mask=None, alpha_mask=None):
        ret = Image(self.zeros())
        cv2.Copy(self.bitmap, ret.bitmap)

        w = img.width
        h = img.height

        if pos is None:
            pos = (0, 0)

        topROI, bottomROI = self. _rect_overlap_rois((img.width, img.height),
                                                     (self.width, self.height),
                                                     pos)

        if alpha is not None:
            cv2.SetImageROI(img.bitmap, topROI)
            cv2.SetImageROI(ret.bitmap, bottomROI)
            a = float(alpha)
            b = float(1.00 - a)
            g = float(0.00)
            cv2.AddWeighted(img.bitmap, a, ret.bitmap, b, g,
                           ret.bitmap)
            cv2.ResetImageROI(img.bitmap)
            cv2.ResetImageROI(ret.bitmap)
        elif alpha_mask is not None:
            if (alpha_mask is not None and
                    (alpha_mask.width != img.width or alpha_mask.height != img.height)):
                logger.warning(
                        "Image.blit: your mask and image don't match sizes, if the mask doesn't fit, you can not blit! Try using the scale function.")
                return None

            cImg = img.crop(topROI[0], topROI[1], topROI[2], topROI[3])
            cMask = alpha_mask.crop(topROI[0], topROI[1], topROI[2], topROI[3])
            retC = ret.crop(bottomROI[0], bottomROI[1], bottomROI[2],
                            bottomROI[3])
            r = cImg.zeros(1)
            g = cImg.zeros(1)
            b = cImg.zeros(1)
            cv2.Split(cImg.bitmap, b, g, r, None)
            rf = cv2.CreateImage((cImg.width, cImg.height), cv2.IPL_DEPTH_32F, 1)
            gf = cv2.CreateImage((cImg.width, cImg.height), cv2.IPL_DEPTH_32F, 1)
            bf = cv2.CreateImage((cImg.width, cImg.height), cv2.IPL_DEPTH_32F, 1)
            af = cv2.CreateImage((cImg.width, cImg.height), cv2.IPL_DEPTH_32F, 1)
            cv2.ConvertScale(r, rf)
            cv2.ConvertScale(g, gf)
            cv2.ConvertScale(b, bf)
            cv2.ConvertScale(cMask._get_gray_narray(), af)
            cv2.ConvertScale(af, af, scale=(1.0 / 255.0))
            cv2.Mul(rf, af, rf)
            cv2.Mul(gf, af, gf)
            cv2.Mul(bf, af, bf)

            dr = retC.zeros(1)
            dg = retC.zeros(1)
            db = retC.zeros(1)
            cv2.Split(retC.bitmap, db, dg, dr, None)
            drf = cv2.CreateImage((retC.width, retC.height),
                                 cv2.IPL_DEPTH_32F, 1)
            dgf = cv2.CreateImage((retC.width, retC.height),
                                 cv2.IPL_DEPTH_32F, 1)
            dbf = cv2.CreateImage((retC.width, retC.height),
                                 cv2.IPL_DEPTH_32F, 1)
            daf = cv2.CreateImage((retC.width, retC.height),
                                 cv2.IPL_DEPTH_32F, 1)
            cv2.ConvertScale(dr, drf)
            cv2.ConvertScale(dg, dgf)
            cv2.ConvertScale(db, dbf)
            cv2.ConvertScale(cMask.invert()._get_gray_narray(), daf)
            cv2.ConvertScale(daf, daf, scale=(1.0 / 255.0))
            cv2.Mul(drf, daf, drf)
            cv2.Mul(dgf, daf, dgf)
            cv2.Mul(dbf, daf, dbf)

            cv2.Add(rf, drf, rf)
            cv2.Add(gf, dgf, gf)
            cv2.Add(bf, dbf, bf)

            cv2.ConvertScaleAbs(rf, r)
            cv2.ConvertScaleAbs(gf, g)
            cv2.ConvertScaleAbs(bf, b)

            cv2.Merge(b, g, r, None, retC.bitmap)
            cv2.SetImageROI(ret.bitmap, bottomROI)
            cv2.Copy(retC.bitmap, ret.bitmap)
            cv2.ResetImageROI(ret.bitmap)

        elif mask is not None:
            if (mask is not None and (
                            mask.width != img.width or mask.height != img.height)):
                logger.warning(
                        "Image.blit: your mask and image don't match sizes, if the mask doesn't fit, you can not blit! Try using the scale function. ")
                return None
            cv2.SetImageROI(img.bitmap, topROI)
            cv2.SetImageROI(mask.bitmap, topROI)
            cv2.SetImageROI(ret.bitmap, bottomROI)
            cv2.Copy(img.bitmap, ret.bitmap, mask.bitmap)
            cv2.ResetImageROI(img.bitmap)
            cv2.ResetImageROI(mask.bitmap)
            cv2.ResetImageROI(ret.bitmap)
        else:  # vanilla blit
            cv2.SetImageROI(img.bitmap, topROI)
            cv2.SetImageROI(ret.bitmap, bottomROI)
            cv2.Copy(img.bitmap, ret.bitmap)
            cv2.ResetImageROI(img.bitmap)
            cv2.ResetImageROI(ret.bitmap)

        return ret

    def side_by_side(self, img, side='right', scale=True):
        ret = None
        if side == "top":
            # clever
            ret = img. side_by_side(self, "bottom", scale)
        elif side == "bottom":
            if self.width > img.width:
                if (scale):
                    # scale the other img width to fit
                    resized = img.resize(w=self.width)
                    nW = self.width
                    nH = self.height + resized.height
                    canvas = cv2.CreateImage((nW, nH), cv2.IPL_DEPTH_8U, 3)
                    cv2.SetZero(canvas)
                    cv2.SetImageROI(canvas, (0, 0, nW, self.height))
                    cv2.Copy(self.bitmap, canvas)
                    cv2.ResetImageROI(canvas)
                    cv2.SetImageROI(canvas, (
                        0, self.height, resized.width, resized.height))
                    cv2.Copy(resized.bitmap, canvas)
                    cv2.ResetImageROI(canvas)
                    ret = Image(canvas, color_space=self._color_space)
                else:
                    nW = self.width
                    nH = self.height + img.height
                    canvas = cv2.CreateImage((nW, nH), cv2.IPL_DEPTH_8U, 3)
                    cv2.SetZero(canvas)
                    cv2.SetImageROI(canvas, (0, 0, nW, self.height))
                    cv2.Copy(self.bitmap, canvas)
                    cv2.ResetImageROI(canvas)
                    xc = (self.width - img.width) / 2
                    cv2.SetImageROI(canvas,
                                   (xc, self.height, img.width, img.height))
                    cv2.Copy(img.bitmap, canvas)
                    cv2.ResetImageROI(canvas)
                    ret = Image(canvas, color_space=self._color_space)
            else:  # our width is smaller than the other img
                if scale:
                    # scale the other img width to fit
                    resized = self.resize(w=img.width)
                    nW = img.width
                    nH = resized.height + img.height
                    canvas = cv2.CreateImage((nW, nH), cv2.IPL_DEPTH_8U, 3)
                    cv2.SetZero(canvas)
                    cv2.SetImageROI(canvas,
                                   (0, 0, resized.width, resized.height))
                    cv2.Copy(resized.bitmap, canvas)
                    cv2.ResetImageROI(canvas)
                    cv2.SetImageROI(canvas,
                                   (0, resized.height, nW, img.height))
                    cv2.Copy(img.bitmap, canvas)
                    cv2.ResetImageROI(canvas)
                    ret = Image(canvas, color_space=self._color_space)
                else:
                    nW = img.width
                    nH = self.height + img.height
                    canvas = cv2.CreateImage((nW, nH), cv2.IPL_DEPTH_8U, 3)
                    cv2.SetZero(canvas)
                    xc = (img.width - self.width) / 2
                    cv2.SetImageROI(canvas, (xc, 0, self.width, self.height))
                    cv2.Copy(self.bitmap, canvas)
                    cv2.ResetImageROI(canvas)
                    cv2.SetImageROI(canvas,
                                   (0, self.height, img.width, img.height))
                    cv2.Copy(img.bitmap, canvas)
                    cv2.ResetImageROI(canvas)
                    ret = Image(canvas, color_space=self._color_space)

        elif side == "right":
            ret = img. side_by_side(self, "left", scale)
        else:  # default to left
            if self.height > img.height:
                if scale:
                    # scale the other img height to fit
                    resized = img.resize(h=self.height)
                    nW = self.width + resized.width
                    nH = self.height
                    canvas = cv2.CreateImage((nW, nH), cv2.IPL_DEPTH_8U, 3)
                    cv2.SetZero(canvas)
                    cv2.SetImageROI(canvas,
                                   (0, 0, resized.width, resized.height))
                    cv2.Copy(resized.bitmap, canvas)
                    cv2.ResetImageROI(canvas)
                    cv2.SetImageROI(canvas,
                                   (resized.width, 0, self.width, self.height))
                    cv2.Copy(self.bitmap, canvas)
                    cv2.ResetImageROI(canvas)
                    ret = Image(canvas, color_space=self._color_space)
                else:
                    nW = self.width + img.width
                    nH = self.height
                    canvas = cv2.CreateImage((nW, nH), cv2.IPL_DEPTH_8U, 3)
                    cv2.SetZero(canvas)
                    yc = (self.height - img.height) / 2
                    cv2.SetImageROI(canvas,
                                   (0, yc, img.width, img.height))
                    cv2.Copy(img.bitmap, canvas)
                    cv2.ResetImageROI(canvas)
                    cv2.SetImageROI(canvas,
                                   (img.width, 0, self.width, self.height))
                    cv2.Copy(self.bitmap, canvas)
                    cv2.ResetImageROI(canvas)
                    ret = Image(canvas, color_space=self._color_space)
            else:  # our height is smaller than the other img
                if scale:
                    # scale our height to fit
                    resized = self.resize(h=img.height)
                    nW = img.width + resized.width
                    nH = img.height
                    canvas = cv2.CreateImage((nW, nH), cv2.IPL_DEPTH_8U, 3)
                    cv2.SetZero(canvas)
                    cv2.SetImageROI(canvas, (0, 0, img.width, img.height))
                    cv2.Copy(img.bitmap, canvas)
                    cv2.ResetImageROI(canvas)
                    cv2.SetImageROI(canvas, (
                        img.width, 0, resized.width, resized.height))
                    cv2.Copy(resized.bitmap, canvas)
                    cv2.ResetImageROI(canvas)
                    ret = Image(canvas, color_space=self._color_space)
                else:
                    nW = img.width + self.width
                    nH = img.height
                    canvas = cv2.CreateImage((nW, nH), cv2.IPL_DEPTH_8U, 3)
                    cv2.SetZero(canvas)
                    cv2.SetImageROI(canvas, (0, 0, img.width, img.height))
                    cv2.Copy(img.bitmap, canvas)
                    cv2.ResetImageROI(canvas)
                    yc = (img.height - self.height) / 2
                    cv2.SetImageROI(canvas,
                                   (img.width, yc, self.width, self.height))
                    cv2.Copy(self.bitmap, canvas)
                    cv2.ResetImageROI(canvas)
                    ret = Image(canvas, color_space=self._color_space)
        return ret

    def embiggen(self, size=None, color=Color.BLACK, pos=None):
        if not isinstance(size, tuple) and size > 1:
            size = (self.width * size, self.height * size)

        if size is None or size[0] < self.width or size[1] < self.height:
            logger.warning("image.embiggenCanvas: the size provided is invalid")
            return None

        newCanvas = cv2.CreateImage(size, cv2.IPL_DEPTH_8U, 3)
        cv2.SetZero(newCanvas)
        newColor = cv2.RGB(color[0], color[1], color[2])
        cv2.AddS(newCanvas, newColor, newCanvas)
        topROI = None
        bottomROI = None
        if pos is None:
            pos = (((size[0] - self.width) / 2), ((size[1] - self.height) / 2))

        (topROI, bottomROI) = self._rect_overlap_rois((self.width, self.height),
                                                    size, pos)
        if topROI is None or bottomROI is None:
            logger.warning(
                    "image.embiggenCanvas: the position of the old image doesn't make sense, there is no overlap")
            return None

        cv2.SetImageROI(newCanvas, bottomROI)
        cv2.SetImageROI(self.bitmap, topROI)
        cv2.Copy(self.bitmap, newCanvas)
        cv2.ResetImageROI(newCanvas)
        cv2.ResetImageROI(self.bitmap)
        return Image(newCanvas)

    def _rect_overlap_rois(self, top, bottom, pos):
        tr = (pos[0] + top[0], pos[1])
        tl = pos
        br = (pos[0] + top[0], pos[1] + top[1])
        bl = (pos[0], pos[1] + top[1])

        # do an overlap test to weed out corner cases and errors
        def inBounds((w, h), (x, y)):
            ret = True
            if x < 0 or y < 0 or x > w or y > h:
                ret = False
            return ret

        trc = inBounds(bottom, tr)
        tlc = inBounds(bottom, tl)
        brc = inBounds(bottom, br)
        blc = inBounds(bottom, bl)
        if not trc and not tlc and not brc and not blc:  # no overlap
            return None, None
        elif trc and tlc and brc and blc:  # easy case top is fully inside bottom
            tRet = (0, 0, top[0], top[1])
            bRet = (pos[0], pos[1], top[0], top[1])
            return tRet, bRet
        # let's figure out where the top rectangle sits on the bottom
        # we clamp the corners of the top rectangle to live inside
        # the bottom rectangle and from that get the x,y,w,h
        tl = (np.clip(tl[0], 0, bottom[0]), np.clip(tl[1], 0, bottom[1]))
        br = (np.clip(br[0], 0, bottom[0]), np.clip(br[1], 0, bottom[1]))

        bx = tl[0]
        by = tl[1]
        bw = abs(tl[0] - br[0])
        bh = abs(tl[1] - br[1])
        # now let's figure where the bottom rectangle is in the top rectangle
        # we do the same thing with different coordinates
        pos = (-1 * pos[0], -1 * pos[1])
        # recalculate the bottoms's corners with respect to the top.
        tr = (pos[0] + bottom[0], pos[1])
        tl = pos
        br = (pos[0] + bottom[0], pos[1] + bottom[1])
        bl = (pos[0], pos[1] + bottom[1])
        tl = (np.clip(tl[0], 0, top[0]), np.clip(tl[1], 0, top[1]))
        br = (np.clip(br[0], 0, top[0]), np.clip(br[1], 0, top[1]))
        tx = tl[0]
        ty = tl[1]
        tw = abs(br[0] - tl[0])
        th = abs(br[1] - tl[1])
        return (tx, ty, tw, th), (bx, by, bw, bh)

    def create_binary_mask(self, color1=(0, 0, 0), color2=(255, 255, 255)):
        if (color1[0] - color2[0] == 0 or
                        color1[1] - color2[1] == 0 or
                        color1[2] - color2[2] == 0):
            logger.warning("No _color range selected, the result will be black, "
                           "returning None instead.")
            return None
        if (color1[0] > 255 or color1[0] < 0 or
                    color1[1] > 255 or color1[1] < 0 or
                    color1[2] > 255 or color1[2] < 0 or
                    color2[0] > 255 or color2[0] < 0 or
                    color2[1] > 255 or color2[1] < 0 or
                    color2[2] > 255 or color2[2] < 0):
            logger.warning("One of the tuple values falls outside of the range "
                           "of 0 to 255")
            return None

        r = self.zeros(1)
        g = self.zeros(1)
        b = self.zeros(1)

        rl = self.zeros(1)
        gl = self.zeros(1)
        bl = self.zeros(1)

        rh = self.zeros(1)
        gh = self.zeros(1)
        bh = self.zeros(1)

        cv2.Split(self.bitmap, b, g, r, None)
        # the difference == 255 case is where open CV
        # kinda screws up, this should just be a white image
        if abs(color1[0] - color2[0]) == 255:
            cv2.Zero(rl)
            cv2.AddS(rl, 255, rl)
        # there is a corner case here where difference == 0
        # right now we throw an error on this case.
        # also we use the triplets directly as OpenCV is
        # SUPER FINICKY about the type of the threshold.
        elif color1[0] < color2[0]:
            cv2.Threshold(r, rl, color1[0], 255, cv2.CV_THRESH_BINARY)
            cv2.Threshold(r, rh, color2[0], 255, cv2.CV_THRESH_BINARY)
            cv2.Sub(rl, rh, rl)
        else:
            cv2.Threshold(r, rl, color2[0], 255, cv2.CV_THRESH_BINARY)
            cv2.Threshold(r, rh, color1[0], 255, cv2.CV_THRESH_BINARY)
            cv2.Sub(rl, rh, rl)

        if abs(color1[1] - color2[1]) == 255:
            cv2.Zero(gl)
            cv2.AddS(gl, 255, gl)
        elif color1[1] < color2[1]:
            cv2.Threshold(g, gl, color1[1], 255, cv2.CV_THRESH_BINARY)
            cv2.Threshold(g, gh, color2[1], 255, cv2.CV_THRESH_BINARY)
            cv2.Sub(gl, gh, gl)
        else:
            cv2.Threshold(g, gl, color2[1], 255, cv2.CV_THRESH_BINARY)
            cv2.Threshold(g, gh, color1[1], 255, cv2.CV_THRESH_BINARY)
            cv2.Sub(gl, gh, gl)

        if abs(color1[2] - color2[2]) == 255:
            cv2.Zero(bl)
            cv2.AddS(bl, 255, bl)
        elif color1[2] < color2[2]:
            cv2.Threshold(b, bl, color1[2], 255, cv2.CV_THRESH_BINARY)
            cv2.Threshold(b, bh, color2[2], 255, cv2.CV_THRESH_BINARY)
            cv2.Sub(bl, bh, bl)
        else:
            cv2.Threshold(b, bl, color2[2], 255, cv2.CV_THRESH_BINARY)
            cv2.Threshold(b, bh, color1[2], 255, cv2.CV_THRESH_BINARY)
            cv2.Sub(bl, bh, bl)

        cv2.And(rl, gl, rl)
        cv2.And(rl, bl, rl)
        return Image(rl)

    def apply_binary_mask(self, mask, bgcolor=Color.BLACK):
        newCanvas = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_8U,
                                   3)
        cv2.SetZero(newCanvas)
        newBG = cv2.RGB(bgcolor[0], bgcolor[1], bgcolor[2])
        cv2.AddS(newCanvas, newBG, newCanvas)

        if mask.width != self.width or mask.height != self.height:
            logger.warning(
                    "Image.applyBinaryMask: your mask and image don't match sizes, if the mask doesn't fit, you can't apply it! Try using the scale function. ")
            return None

        cv2.Copy(self.bitmap, newCanvas, mask.bitmap)

        return Image(newCanvas, color_space=self._color_space)

    def create_alpha_mask(self, hue=60, hue_lb=None, hue_ub=None):
        if hue < 0 or hue > 180:
            logger.warning("Invalid hue _color, valid hue range is 0 to 180.")

        if self._color_space != ColorSpace.HSV:
            hsv = self.to_hsv()
        else:
            hsv = self
        h = hsv.zeros(1)
        s = hsv.zeros(1)
        ret = hsv.zeros(1)
        mask = hsv.zeros(1)
        cv2.Split(hsv.bitmap, h, None, s, None)
        # thankfully we're not doing a LUT on saturation
        hlut = np.zeros((256, 1), dtype=uint8)
        if hue_lb is not None and hue_ub is not None:
            hlut[hue_lb:hue_ub] = 255
        else:
            hlut[hue] = 255
        cv2.LUT(h, mask, cv2.fromarray(hlut))
        cv2.Copy(s, ret, mask)  # we'll save memory using hue
        return Image(ret)

    def apply_pixel_function(self, func):
        # there should be a way to do this faster using numpy vectorize
        # but I can get vectorize to work with the three channels together... have to split them
        # TODO: benchmark this against vectorize
        pixels = np.array(self.narray).reshape(-1, 3).tolist()
        result = np.array(map(func, pixels), dtype=uint8).reshape(self.width,
                                                                   self.height,
                                                                   3)
        return Image(result)

    def integral_image(self, titled=False):
        if titled:
            img2 = cv2.CreateImage((self.width + 1, self.height + 1),
                                  cv2.IPL_DEPTH_32F, 1)
            img3 = cv2.CreateImage((self.width + 1, self.height + 1),
                                  cv2.IPL_DEPTH_32F, 1)
            cv2.Integral(self._get_gray_narray(), img3, None, img2)
        else:
            img2 = cv2.CreateImage((self.width + 1, self.height + 1),
                                  cv2.IPL_DEPTH_32F, 1)
            cv2.Integral(self._get_gray_narray(), img2)
        return np.array(cv2.GetMat(img2))

    def convolve(self, kernel=np.eye(3), center=None):
        if isinstance(kernel, list):
            kernel = np.array(kernel)

        if isinstance(kernel, np.ndarray):
            sz = kernel.shape
            kernel = kernel.astype(np.float32)
            ker = cv2.CreateMat(sz[0], sz[1], cv2.CV_32FC1)
            cv2.SetData(ker, kernel.tostring(),
                       kernel.dtype.itemsize * kernel.shape[1])
        elif isinstance(kernel, cv2.mat):
            ker = kernel
        else:
            logger.warning("Convolution uses numpy arrays or cv2.mat type.")
            return None
        ret = self.zeros(3)
        if center is None:
            cv2.Filter2D(self.bitmap, ret, ker)
        else:
            cv2.Filter2D(self.bitmap, ret, ker, center)
        return Image(ret)

    def find_template(self, template=None, threshold=5, method='SQR_DIFF_NORM',
                      grayscale=True, rawmatches=False):
        if template is None:
            logger.info("Need image for matching")
            return

        if template.width > self.width:
            # logger.info( "Image too wide")
            return

        if template.height > self.height:
            logger.info("Image too tall")
            return

        check = 0  # if check = 0 we want maximal value, otherwise minimal
        if (
                            method is None or method == "" or method == "SQR_DIFF_NORM"):  # minimal
            method = cv2.CV_TM_SQDIFF_NORMED
            check = 1
        elif method == "SQR_DIFF":  # minimal
            method = cv2.CV_TM_SQDIFF
            check = 1
        elif method == "CCOEFF":  # maximal
            method = cv2.CV_TM_CCOEFF
        elif method == "CCOEFF_NORM":  # maximal
            method = cv2.CV_TM_CCOEFF_NORMED
        elif method == "CCORR":  # maximal
            method = cv2.CV_TM_CCORR
        elif method == "CCORR_NORM":  # maximal
            method = cv2.CV_TM_CCORR_NORMED
        else:
            logger.warning("ooops.. I don't know what template matching method "
                           "you are looking for.")
            return None
        # create new image for template matching computation
        matches = cv2.CreateMat((self.height - template.height + 1),
                               (self.width - template.width + 1),
                               cv2.CV_32FC1)

        # choose template matching method to be used
        if grayscale:
            cv2.MatchTemplate(self._get_gray_narray(),
                              template._get_gray_narray(), matches,
                              method)
        else:
            cv2.MatchTemplate(self.bitmap, template.bitmap,
                             matches, method)
        mean = np.mean(matches)
        sd = np.std(matches)
        if check > 0:
            compute = np.where((matches < mean - threshold * sd))
        else:
            compute = np.where((matches > mean + threshold * sd))

        mapped = map(tuple, np.col_stack(compute))
        fs = FeatureSet()
        for location in mapped:
            fs.append(
                    TemplateMatch(self, template, (location[1], location[0]),
                                  matches[location[0], location[1]]))

        if rawmatches:
            return fs
        # cluster overlapping template matches
        finalfs = FeatureSet()
        if len(fs) > 0:
            finalfs.append(fs[0])
            for f in fs:
                match = False
                for f2 in finalfs:
                    if f2._template_overlaps(f):  # if they overlap
                        f2.consume(f)  # merge them
                        match = True
                        break

                if not match:
                    finalfs.append(f)

            for f in finalfs:  # rescale the resulting clusters to fit the template size
                f.rescale(template.width, template.height)

            fs = finalfs

        return fs

    def find_template_once(self, template=None, threshold=0.2,
                           method='SQR_DIFF_NORM',
                           grayscale=True):
        if template is None:
            logger.info("Need image for template matching.")
            return

        if template.width > self.width:
            logger.info("Template image is too wide for the given image.")
            return

        if template.height > self.height:
            logger.info("Template image too tall for the given image.")
            return

        check = 0  # if check = 0 we want maximal value, otherwise minimal
        if method is None or method == "" or method == "SQR_DIFF_NORM":  # minimal
            method = cv2.CV_TM_SQDIFF_NORMED
            check = 1
        elif method == "SQR_DIFF":  # minimal
            method = cv2.CV_TM_SQDIFF
            check = 1
        elif method == "CCOEFF":  # maximal
            method = cv2.CV_TM_CCOEFF
        elif method == "CCOEFF_NORM":  # maximal
            method = cv2.CV_TM_CCOEFF_NORMED
        elif method == "CCORR":  # maximal
            method = cv2.CV_TM_CCORR
        elif method == "CCORR_NORM":  # maximal
            method = cv2.CV_TM_CCORR_NORMED
        else:
            logger.warning(
                    "ooops.. I don't know what template matching method you are looking for.")
            return None
        # create new image for template matching computation
        matches = cv2.CreateMat((self.height - template.height + 1),
                               (self.width - template.width + 1),
                               cv2.CV_32FC1)

        # choose template matching method to be used
        if grayscale:
            cv2.MatchTemplate(self._get_gray_narray(),
                              template._get_gray_narray(), matches,
                              method)
        else:
            cv2.MatchTemplate(self.bitmap, template.bitmap,
                             matches, method)
        mean = np.mean(matches)
        sd = np.std(matches)
        if check > 0:
            if np.min(matches) <= threshold:
                compute = np.where(matches == np.min(matches))
            else:
                return []
        else:
            if np.max(matches) >= threshold:
                compute = np.where(matches == np.max(matches))
            else:
                return []
        mapped = map(tuple, np.col_stack(compute))
        fs = FeatureSet()
        for location in mapped:
            fs.append(
                    TemplateMatch(self, template, (location[1], location[0]),
                                  matches[location[0], location[1]]))

        return fs

    def read_text(self):
        if not OCR_ENABLED:
            return "Please install the correct OCR library required - http://code.google.com/p/tesseract-ocr/ http://code.google.com/p/python-tesseract/"

        api = tesseract.TessBaseAPI()
        api.SetOutputName("outputName")
        api.Init(".", "eng", tesseract.OEM_DEFAULT)
        api.SetPageSegMode(tesseract.PSM_AUTO)

        jpgdata = StringIO()
        self.pilimg.save(jpgdata, "jpeg")
        jpgdata.seek(0)
        stringbuffer = jpgdata.read()
        result = tesseract.ProcessPagesBuffer(stringbuffer, len(stringbuffer),
                                              api)
        return result

    def find_circle(self, canny=100, thresh=350, distance=-1):
        storage = cv2.CreateMat(self.width, 1, cv2.CV_32FC3)
        # a distance metric for how apart our circles should be - this is sa good bench mark
        if distance < 0:
            distance = 1 + max(self.width, self.height) / 50
        cv2.HoughCircles(self._get_gray_narray(), storage,
                         cv2.CV_HOUGH_GRADIENT, 2, distance, canny, thresh)
        if storage.rows == 0:
            return None
        circs = np.asarray(storage)
        sz = circs.shape
        circleFS = FeatureSet()
        for i in range(sz[0]):
            circleFS.append(
                    Circle(self, int(circs[i][0][0]), int(circs[i][0][1]),
                           int(circs[i][0][2])))
        return circleFS

    def white_balance(self, method='simple'):
        img = self
        if method == "GrayWorld":
            avg = cv2.Avg(img.bitmap);
            bf = float(avg[0])
            gf = float(avg[1])
            rf = float(avg[2])
            af = (bf + gf + rf) / 3.0
            if bf == 0.00:
                b_factor = 1.00
            else:
                b_factor = af / bf

            if gf == 0.00:
                g_factor = 1.00
            else:
                g_factor = af / gf

            if rf == 0.00:
                r_factor = 1.00
            else:
                r_factor = af / rf

            b = img.zeros(1)
            g = img.zeros(1)
            r = img.zeros(1)
            cv2.Split(self.bitmap, b, g, r, None)
            bfloat = cv2.CreateImage((img.width, img.height), cv2.IPL_DEPTH_32F,
                                    1)
            gfloat = cv2.CreateImage((img.width, img.height), cv2.IPL_DEPTH_32F,
                                    1)
            rfloat = cv2.CreateImage((img.width, img.height), cv2.IPL_DEPTH_32F,
                                    1)

            cv2.ConvertScale(b, bfloat, b_factor)
            cv2.ConvertScale(g, gfloat, g_factor)
            cv2.ConvertScale(r, rfloat, r_factor)

            (minB, maxB, minBLoc, maxBLoc) = cv2.MinMaxLoc(bfloat)
            (minG, maxG, minGLoc, maxGLoc) = cv2.MinMaxLoc(gfloat)
            (minR, maxR, minRLoc, maxRLoc) = cv2.MinMaxLoc(rfloat)
            scale = max([maxR, maxG, maxB])
            sfactor = 1.00
            if (scale > 255):
                sfactor = 255.00 / float(scale)

            cv2.ConvertScale(bfloat, b, sfactor);
            cv2.ConvertScale(gfloat, g, sfactor);
            cv2.ConvertScale(rfloat, r, sfactor);

            ret = img.zeros()
            cv2.Merge(b, g, r, None, ret);
            ret = Image(ret)
        elif method == "Simple":
            thresh = 0.003
            sz = img.width * img.height
            tmp_mat = img.narray
            bcf = sss.cumfreq(tmp_mat[:, :, 0], numbins=256)
            # get our cumulative histogram of values for this _color
            bcf = bcf[0]

            blb = -1  # our upper bound
            bub = 256  # our lower bound
            lower_thresh = 0.00
            upper_thresh = 0.00
            # now find the upper and lower thresh% of our values live
            while lower_thresh < thresh:
                blb = blb + 1
                lower_thresh = bcf[blb] / sz
            while upper_thresh < thresh:
                bub = bub - 1
                upper_thresh = (sz - bcf[bub]) / sz

            gcf = sss.cumfreq(tmp_mat[:, :, 1], numbins=256)
            gcf = gcf[0]
            glb = -1  # our upper bound
            gub = 256  # our lower bound
            lower_thresh = 0.00
            upper_thresh = 0.00
            # now find the upper and lower thresh% of our values live
            while lower_thresh < thresh:
                glb += 1
                lower_thresh = gcf[glb] / sz
            while upper_thresh < thresh:
                gub -= 1
                upper_thresh = (sz - gcf[gub]) / sz

            rcf = sss.cumfreq(tmp_mat[:, :, 2], numbins=256)
            rcf = rcf[0]
            rlb = -1  # our upper bound
            rub = 256  # our lower bound
            lower_thresh = 0.00
            upper_thresh = 0.00
            # now find the upper and lower thresh% of our values live
            while lower_thresh < thresh:
                rlb += 1
                lower_thresh = rcf[rlb] / sz
            while upper_thresh < thresh:
                rub -= 1
                upper_thresh = (sz - rcf[rub]) / sz
            # now we create the scale factors for the remaining pixels
            rlbf = float(rlb)
            rubf = float(rub)
            glbf = float(glb)
            gubf = float(gub)
            blbf = float(blb)
            bubf = float(bub)

            rLUT = np.ones((256, 1), dtype=uint8)
            gLUT = np.ones((256, 1), dtype=uint8)
            bLUT = np.ones((256, 1), dtype=uint8)
            for i in range(256):
                if i <= rlb:
                    rLUT[i][0] = 0
                elif i >= rub:
                    rLUT[i][0] = 255
                else:
                    rf = ((float(i) - rlbf) * 255.00 / (rubf - rlbf))
                    rLUT[i][0] = int(rf)
                if glb >= i:
                    gLUT[i][0] = 0
                elif i >= gub:
                    gLUT[i][0] = 255
                else:
                    gf = ((float(i) - glbf) * 255.00 / (gubf - glbf))
                    gLUT[i][0] = int(gf)
                if i <= blb:
                    bLUT[i][0] = 0
                elif i >= bub:
                    bLUT[i][0] = 255
                else:
                    bf = ((float(i) - blbf) * 255.00 / (bubf - blbf))
                    bLUT[i][0] = int(bf)
            ret = img.apply_lut(bLUT, rLUT, gLUT)
        return ret

    def apply_lut(self, rlut=None, blut=None, glut=None):
        # BUG NOTE: error on the LUT map for some versions of OpenCV
        r = self.zeros(1)
        g = self.zeros(1)
        b = self.zeros(1)
        cv2.Split(self.bitmap, b, g, r, None);

        if rlut is not None:
            cv2.LUT(r, r, cv2.fromarray(rlut))
        if glut is not None:
            cv2.LUT(g, g, cv2.fromarray(glut))
        if blut is not None:
            cv2.LUT(b, b, cv2.fromarray(blut))

        temp = self.zeros()
        cv2.Merge(b, g, r, None, temp)

        return Image(temp)

    def _get_raw_keypoints(self, thresh=500.00, flavor='SURF', highQuality=1,
                           force_reset=False):
        try:
            import cv2
            ver = cv2.__version__
            new_version = 0
            # For OpenCV versions till 2.4.0,  cv2.__versions__ are of the form "$Rev: 4557 $"
            if not ver.startswith('$Rev:'):
                if int(ver.replace('.', '0')) >= 20400:
                    new_version = 1
        except:
            warnings.warn("Can't run Keypoints without OpenCV >= 2.3.0")
            return None, None

        if force_reset:
            self._keypoints = None
            self._kp_descriptors = None

        _detectors = ["SIFT", "SURF", "FAST", "STAR", "FREAK", "ORB", "BRISK",
                      "MSER", "Dense"]
        _descriptors = ["SIFT", "SURF", "ORB", "FREAK", "BRISK"]
        if flavor not in _detectors:
            warnings.warn("Invalid choice of keypoint detector.")
            return (None, None)

        if self._keypoints is None and self._kp_flavor == flavor:
            return self._keypoints, self._kp_descriptors

        if hasattr(cv2, flavor):
            if flavor == "SURF":
                detector = cv2.SURF(thresh, 4, 2, highQuality, 1)
                if new_version == 0:
                    self._keypoints, self._kp_descriptors = detector.detect(
                            self.gray_narray, None, False)
                else:
                    self._keypoints, self._kp_descriptors = detector.detectAndCompute(
                            self.gray_narray, None, False)
                if len(self._keypoints) == 0:
                    return None, None
                if highQuality == 1:
                    self._kp_descriptors = self._kp_descriptors.reshape(
                            (-1, 128))
                else:
                    self._kp_descriptors = self._kp_descriptors.reshape(
                            (-1, 64))

            elif flavor in _descriptors:
                detector = getattr(cv2, flavor)()
                self._keypoints, self._kp_descriptors = detector.detectAndCompute(
                        self.gray_narray, None, False)
            elif flavor == "MSER":
                if hasattr(cv2, "FeatureDetector_create"):
                    detector = cv2.FeatureDetector_create("MSER")
                    self._keypoints = detector.detect(self.gray_narray)
        elif flavor == "STAR":
            detector = cv2.StarDetector()
            self._keypoints = detector.detect(self.gray_narray)
        elif flavor == "FAST":
            if not hasattr(cv2, "FastFeatureDetector"):
                warnings.warn("You need OpenCV >= 2.4.0 to support FAST")
                return None, None
            detector = cv2.FastFeatureDetector(int(thresh), True)
            self._keypoints = detector.detect(self.gray_narray, None)
        elif hasattr(cv2, "FeatureDetector_create"):
            if flavor in _descriptors:
                extractor = cv2.DescriptorExtractor_create(flavor)
                if flavor == "FREAK":
                    if new_version == 0:
                        warnings.warn(
                            "You need OpenCV >= 2.4.3 to support FAST")
                    flavor = "SIFT"
                detector = cv2.FeatureDetector_create(flavor)
                self._keypoints = detector.detect(self.gray_narray)
                self._keypoints, self._kp_descriptors = extractor.compute(
                        self.gray_narray, self._keypoints)
            else:
                detector = cv2.FeatureDetector_create(flavor)
                self._keypoints = detector.detect(self.gray_narray)
        else:
            warnings.warn(
                    "PhloxAR can't seem to find appropriate function with your OpenCV version.")
            return None, None
        return self._keypoints, self._kp_descriptors

    def _get_FLANN_matches(self, sd, td):
        try:
            import cv2
        except:
            logger.warning("Can't run FLANN Matches without OpenCV >= 2.3.0")
            return
        FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
        flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
        flann = cv2.flann_Index(sd, flann_params)
        idx, dist = flann.knnSearch(td, 1,
                                    params={})  # bug: need to provide empty dict
        del flann
        return idx, dist

    def draw_keypoints_matches(self, template, thresh=500.00, min_dist=0.15,
                               width=1):
        if template is None:
            return None

        resultImg = template. side_by_side(self, scale=False)
        hdif = (self.height - template.height) / 2
        skp, sd = self._get_raw_keypoints(thresh)
        tkp, td = template._get_raw_keypoints(thresh)
        if td is None or sd is None:
            logger.warning(
                    "We didn't get any descriptors. Image might be too uniform or blurry.")
            return resultImg
        template_points = float(td.shape[0])
        sample_points = float(sd.shape[0])
        magic_ratio = 1.00
        if sample_points > template_points:
            magic_ratio = float(sd.shape[0]) / float(td.shape[0])

        idx, dist = self._get_FLANN_matches(sd,
                                          td)  # match our keypoint descriptors
        p = dist[:, 0]
        result = p * magic_ratio < min_dist  # , = np.where( p*magic_ratio < minDist )
        for i in range(0, len(idx)):
            if result[i]:
                pt_a = (tkp[i].pt[1], tkp[i].pt[0] + hdif)
                pt_b = (skp[idx[i]].pt[1] + template.width, skp[idx[i]].pt[0])
                resultImg.drawLine(pt_a, pt_b, color=Color.random(),
                                   thickness=width)
        return resultImg

    def find_keypoint_match(self, template, quality=500.00, min_dist=0.2,
                            min_match=0.4):
        try:
            import cv2
        except:
            warnings.warn("Can't Match Keypoints without OpenCV >= 2.3.0")
            return

        if template is None:
            return None
        fs = FeatureSet()
        skp, sd = self._get_raw_keypoints(quality)
        tkp, td = template._get_raw_keypoints(quality)
        if skp is None or tkp is None:
            warnings.warn(
                    "I didn't get any keypoints. Image might be too uniform or blurry.")
            return None

        template_points = float(td.shape[0])
        sample_points = float(sd.shape[0])
        magic_ratio = 1.00
        if sample_points > template_points:
            magic_ratio = float(sd.shape[0]) / float(td.shape[0])

        idx, dist = self._get_FLANN_matches(sd,
                                          td)  # match our keypoint descriptors
        p = dist[:, 0]
        result = p * magic_ratio < min_dist  # , = np.where( p*magic_ratio < minDist )
        pr = result.shape[0] / float(dist.shape[0])

        if pr > min_match and len(result) > 4:  # if more than minMatch % matches we go ahead and get the data
            lhs = []
            rhs = []
            for i in range(0, len(idx)):
                if result[i]:
                    lhs.append((tkp[i].pt[1], tkp[i].pt[0]))
                    rhs.append((skp[idx[i]].pt[0], skp[idx[i]].pt[1]))

            rhs_pt = np.array(rhs)
            lhs_pt = np.array(lhs)
            if len(rhs_pt) < 16 or len(lhs_pt) < 16:
                return None
            homography = []
            (homography, mask) = cv2.findHomography(lhs_pt, rhs_pt, cv2.RANSAC,
                                                    ransacReprojThreshold=1.0)
            w = template.width
            h = template.height

            pts = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype="float32")

            pPts = cv2.perspectiveTransform(np.array([pts]), homography)

            pt0i = (pPts[0][0][1], pPts[0][0][0])
            pt1i = (pPts[0][1][1], pPts[0][1][0])
            pt2i = (pPts[0][2][1], pPts[0][2][0])
            pt3i = (pPts[0][3][1], pPts[0][3][0])

            # construct the feature set and return it.
            fs = FeatureSet()
            fs.append(
                    KeypointMatch(self, template, (pt0i, pt1i, pt2i, pt3i),
                                  homography))
            # the homography matrix is necessary for many purposes like image stitching.
            # fs.append(homography) # No need to add homography as it is already being
            # added in KeyPointMatch class.
            return fs
        else:
            return None

    def find_keypoints(self, min_quality=300.00, flavor='SURF',
                       high_quality=False):
        try:
            import cv2
        except:
            logger.warning("Can't use Keypoints without OpenCV >= 2.3.0")
            return None

        fs = FeatureSet()
        kp = []
        d = []
        if high_quality:
            kp, d = self._get_raw_keypoints(thresh=min_quality, force_reset=True,
                                            flavor=flavor, high_quality=1)
        else:
            kp, d = self._get_raw_keypoints(thresh=min_quality, force_reset=True,
                                            flavor=flavor, high_quality=0)

        if (flavor in ["ORB", "SIFT", "SURF", "BRISK", "FREAK"] and
                    kp is not None and d is not None):
            for i in range(0, len(kp)):
                fs.append(KeyPoint(self, kp[i], d[i], flavor))
        elif flavor in ["FAST", "STAR", "MSER", "Dense"] and kp is not None:
            for i in range(0, len(kp)):
                fs.append(KeyPoint(self, kp[i], None, flavor))
        else:
            logger.warning(
                    "Image.Keypoints: I don't know the method you want to use")
            return None

        return fs

    def find_motion(self, previous_frame, window=11, method='BM',
                    aggregate=True):
        try:
            import cv2
            ver = cv2.__version__
            # For OpenCV versions till 2.4.0,  cv2.__versions__ are of the form "$Rev: 4557 $"
            if not ver.startswith('$Rev:'):
                if int(ver.replace('.', '0')) >= 20400:
                    FLAG_VER = 1
                    if (window > 9):
                        window = 9
            else:
                FLAG_VER = 0
        except:
            FLAG_VER = 0

        if (
                        self.width != previous_frame.width or self.height != previous_frame.height):
            logger.warning("Image.getMotion: To find motion the current and"
                           "previous frames must match")
            return None
        fs = FeatureSet()
        max_mag = 0.00

        if method == "LK" or method == "HS":
            # create the result images.
            xf = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_32F, 1)
            yf = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_32F, 1)
            win = (window, window)
            if method == "LK":
                cv2.CalcOpticalFlowLK(self._get_gray_narray(),
                                      previous_frame._get_gray_narray(), win,
                                      xf,
                                      yf)
            else:
                cv2.CalcOpticalFlowHS(previous_frame._get_gray_narray(),
                                      self._get_gray_narray(), 0, xf, yf, 1.0,
                                      (
                                         cv2.CV_TERMCRIT_ITER | cv2.CV_TERMCRIT_EPS,
                                         10,
                                         0.01))

            w = math.floor((float(window)) / 2.0)
            cx = ((self.width - window) / window) + 1  # our sample rate
            cy = ((self.height - window) / window) + 1
            vx = 0.00
            vy = 0.00
            for x in range(0, int(cx)):  # go through our sample grid
                for y in range(0, int(cy)):
                    xi = (x * window) + w  # calculate the sample point
                    yi = (y * window) + w
                    if aggregate:
                        lowx = int(xi - w)
                        highx = int(xi + w)
                        lowy = int(yi - w)
                        highy = int(yi + w)
                        # get the average x/y components in the output
                        xderp = xf[lowy:highy, lowx:highx]
                        yderp = yf[lowy:highy, lowx:highx]
                        vx = np.average(xderp)
                        vy = np.average(yderp)
                    else:  # other wise just sample
                        vx = xf[yi, xi]
                        vy = yf[yi, xi]

                    mag = (vx * vx) + (vy * vy)

                    # calculate the max magnitude for normalizing our vectors
                    if (mag > max_mag):
                        max_mag = mag

                    # add the sample to the feature set
                    fs.append(Motion(self, xi, yi, vx, vy, window))

        elif method == "BM":
            # In the interest of keep the parameter list short
            # I am pegging these to the window size.
            # For versions with OpenCV 2.4.0 and below.
            if FLAG_VER == 0:
                block = (window, window)  # block size
                shift = (
                    int(window * 1.2),
                    int(window * 1.2))  # how far to shift the block
                spread = (window * 2, window * 2)  # the search windows.
                wv = (self.width - block[0]) / shift[0]  # the result image size
                hv = (self.height - block[1]) / shift[1]
                xf = cv2.CreateMat(hv, wv, cv2.CV_32FC1)
                yf = cv2.CreateMat(hv, wv, cv2.CV_32FC1)
                cv2.CalcOpticalFlowBM(previous_frame._get_gray_narray(),
                                      self._get_gray_narray(), block, shift,
                                      spread, 0, xf, yf)

            # For versions with OpenCV 2.4.0 and above.
            elif FLAG_VER == 1:
                block = (window, window)  # block size
                shift = (
                    int(window * 0.2),
                    int(window * 0.2))  # how far to shift the block
                spread = (window, window)  # the search windows.
                wv = self.width - block[0] + shift[0]
                hv = self.height - block[1] + shift[1]
                xf = cv2.CreateImage((wv, hv), cv2.IPL_DEPTH_32F, 1)
                yf = cv2.CreateImage((wv, hv), cv2.IPL_DEPTH_32F, 1)
                cv2.CalcOpticalFlowBM(previous_frame._get_gray_narray(),
                                      self._get_gray_narray(), block, shift,
                                      spread, 0, xf, yf)

            for x in range(0, int(wv)):  # go through the sample grid
                for y in range(0, int(hv)):
                    xi = (shift[0] * (x)) + block[
                        0]  # where on the input image the samples live
                    yi = (shift[1] * (y)) + block[1]
                    vx = xf[y, x]  # the result image values
                    vy = yf[y, x]
                    fs.append(
                            Motion(self, xi, yi, vx, vy,
                                   window))  # add the feature
                    mag = (vx * vx) + (vy * vy)  # same the magnitude
                    if (mag > max_mag):
                        max_mag = mag
        else:
            logger.warning("Image.findMotion: I don't know what algorithm you "
                           "want to use. Valid method choices are Block "
                           "Matching -> \"BM\" Horn-Schunck -> \"HS\" and "
                           "Lucas-Kanade->\"LK\" ")
            return None

        max_mag = math.sqrt(max_mag)  # do the normalization
        for f in fs:
            f.normalizeTo(max_mag)

    def _generate_palette(self, bins, hue, centroids=None):
        if (self._palette_bins != bins or
                    self._do_hue_palette != hue):
            total = float(self.width * self.height)
            percentages = []
            result = None
            if not hue:
                # reshape our matrix to 1xN
                pixels = np.array(self.narray).reshape(-1, 3)
                if centroids is None:
                    result = scv2.kmeans(pixels, bins)
                else:
                    if isinstance(centroids, list):
                        centroids = np.array(centroids, dtype='uint8')
                    result = scv2.kmeans(pixels, centroids)

                self._palette_members = scv2.vq(pixels, result[0])[0]

            else:
                hsv = self
                if self._color_space != ColorSpace.HSV:
                    hsv = self.to_hsv()

                h = hsv.zeros(1)
                cv2.Split(hsv.bitmap, None, None, h, None)
                mat = cv2.GetMat(h)
                pixels = np.array(mat).reshape(-1, 1)

                if centroids is None:
                    result = scv2.kmeans(pixels, bins)
                else:
                    if isinstance(centroids, list):
                        centroids = np.array(centroids, dtype='uint8')
                        centroids = centroids.reshape(centroids.shape[0], 1)
                    result = scv2.kmeans(pixels, centroids)

                self._palette_members = scv2.vq(pixels, result[0])[0]

            for i in range(0, bins):
                count = np.where(self._palette_members == i)
                v = float(count[0].shape[0]) / total
                percentages.append(v)

            self._do_hue_palette = hue
            self._palette_bins = bins
            self._palette = np.array(result[0], dtype='uint8')
            self._palette_percentages = percentages

    def get_palette(self, bins=10, hue=False, centroids=None):
        self._generate_palette(bins, hue, centroids)
        return self._palette

    def re_palette(self, palette, hue=False):
        ret = None
        if hue:
            hsv = self
            if self._color_space != ColorSpace.HSV:
                hsv = self.to_hsv()

            h = hsv.zeros(1)
            cv2.Split(hsv.bitmap, None, None, h, None)
            mat = cv2.GetMat(h)
            pixels = np.array(mat).reshape(-1, 1)
            result = scv2.vq(pixels, palette)
            derp = palette[result[0]]
            ret = Image(derp[::-1].reshape(self.height, self.width)[::-1])
            ret = ret.rotate(-90, fixed=False)
            ret._do_hue_palette = True
            ret._palette_bins = len(palette)
            ret._palette= palette
            ret._palette_members = result[0]

        else:
            result = scv2.vq(self.narray.reshape(-1, 3), palette)
            ret = Image(palette[result[0]].reshape(self.width, self.height, 3))
            ret._do_hue_palette = False
            ret._palette_bins = len(palette)
            ret._palette= palette
            pixels = np.array(self.narray).reshape(-1, 3)
            ret._palette_members = scv2.vq(pixels, palette)[0]

        percentages = []
        total = self.width * self.height
        for i in range(0, len(palette)):
            count = np.where(self._palette_members == i)
            v = float(count[0].shape[0]) / total
            percentages.append(v)
        self._palette_percentages = percentages
        return ret

    def draw_palette_colors(self, size=(-1, -1), horizontal=True, bins=10,
                            hue=False):
        self._generate_palette(bins, hue)
        ret = None
        if not hue:
            if horizontal:
                if (size[0] == -1 or size[1] == -1):
                    size = (int(self.width), int(self.height * .1))
                pal = cv2.CreateImage(size, cv2.IPL_DEPTH_8U, 3)
                cv2.Zero(pal)
                idxL = 0
                idxH = 0
                for i in range(0, bins):
                    idxH = np.clip(
                            idxH + (
                            self._palette_percentages[i] * float(size[0])), 0,
                            size[0] - 1)
                    roi = (int(idxL), 0, int(idxH - idxL), size[1])
                    cv2.SetImageROI(pal, roi)
                    color = np.array((float(self._palette[i][2]),
                                      float(self._palette[i][1]),
                                      float(self._palette[i][0])))
                    cv2.AddS(pal, color, pal)
                    cv2.ResetImageROI(pal)
                    idxL = idxH
                ret = Image(pal)
            else:
                if (size[0] == -1 or size[1] == -1):
                    size = (int(self.width * .1), int(self.height))
                pal = cv2.CreateImage(size, cv2.IPL_DEPTH_8U, 3)
                cv2.Zero(pal)
                idxL = 0
                idxH = 0
                for i in range(0, bins):
                    idxH = np.clip(
                        idxH + self._palette_percentages[i] * size[1], 0,
                        size[1] - 1)
                    roi = (0, int(idxL), size[0], int(idxH - idxL))
                    cv2.SetImageROI(pal, roi)
                    color = np.array((float(self._palette[i][2]),
                                      float(self._palette[i][1]),
                                      float(self._palette[i][0])))
                    cv2.AddS(pal, color, pal)
                    cv2.ResetImageROI(pal)
                    idxL = idxH
                ret = Image(pal)
        else:  # do hue
            if (horizontal):
                if (size[0] == -1 or size[1] == -1):
                    size = (int(self.width), int(self.height * .1))
                pal = cv2.CreateImage(size, cv2.IPL_DEPTH_8U, 1)
                cv2.Zero(pal)
                idxL = 0
                idxH = 0
                for i in range(0, bins):
                    idxH = np.clip(
                            idxH + (
                            self._palette_percentages[i] * float(size[0])), 0,
                            size[0] - 1)
                    roi = (int(idxL), 0, int(idxH - idxL), size[1])
                    cv2.SetImageROI(pal, roi)
                    cv2.AddS(pal, float(self._palette[i]), pal)
                    cv2.ResetImageROI(pal)
                    idxL = idxH
                ret = Image(pal)
            else:
                if (size[0] == -1 or size[1] == -1):
                    size = (int(self.width * .1), int(self.height))
                pal = cv2.CreateImage(size, cv2.IPL_DEPTH_8U, 1)
                cv2.Zero(pal)
                idxL = 0
                idxH = 0
                for i in range(0, bins):
                    idxH = np.clip(
                        idxH + self._palette_percentages[i] * size[1], 0,
                        size[1] - 1)
                    roi = (0, int(idxL), size[0], int(idxH - idxL))
                    cv2.SetImageROI(pal, roi)
                    cv2.AddS(pal, float(self._palette[i]), pal)
                    cv2.ResetImageROI(pal)
                    idxL = idxH
                ret = Image(pal)

        return ret

    def palettize(self, bins=10, hue=False, centroids=None):
        ret = None
        self._generate_palette(bins, hue, centroids)
        if (hue):
            derp = self._palette[self._palette_members]
            ret = Image(derp[::-1].reshape(self.height, self.width)[::-1])
            ret = ret.rotate(-90, fixed=False)
        else:
            ret = Image(
                self._palette[self._palette_members].reshape(self.width,
                                                              self.height,
                                                              3))
        return ret

    def find_blobs_from_palette(self, palette_selection, dilate=0, minsize=5,
                                maxsize=0, appx_level=3):
        bwimg = self.binarizeFromPalette(palette_selection)
        if dilate > 0:
            bwimg = bwimg.dilate(dilate)

        if maxsize == 0:
            maxsize = self.width * self.height
        # create a single channel image, thresholded to parameters

        blobmaker = BlobMaker()
        blobs = blobmaker.extract_from_binary(bwimg,
                                              self, minsize=minsize,
                                              maxsize=maxsize,
                                              appx_level=appx_level)

        if not len(blobs):
            return None
        return blobs

    def binarize_from_palette(self, palette_selection):
        if self._palette is None:
            logger.warning(
                    "Image.binarizeFromPalette: No palette exists, call getPalette())")
            return None
        ret = None
        img = self.palettize(self._palette_bins, hue=self._do_hue_palette)
        if not self._do_hue_palette:
            npimg = img.narray
            white = np.array([255, 255, 255])
            black = np.array([0, 0, 0])

            for p in palette_selection:
                npimg = np.where(npimg != p, npimg, white)

            npimg = np.where(npimg != white, black, white)
            ret = Image(npimg)
        else:
            npimg = img.narray[:, :, 1]
            white = np.array([255])
            black = np.array([0])

            for p in palette_selection:
                npimg = np.where(npimg != p, npimg, white)

            npimg = np.where(npimg != white, black, white)
            ret = Image(npimg)

        return ret

    def skeletonize(self, radius=5):
        img = self.to_gray().narray[:, :, 0]
        distance_img = ndimage.distance_transform_edt(img)
        morph_laplace_img = ndimage.morphological_laplace(distance_img,
                                                          (radius, radius))
        skeleton = morph_laplace_img < morph_laplace_img.min() / 2
        ret = np.zeros([self.width, self.height])
        ret[skeleton] = 255
        return Image(ret)

    def smart_threshold(self, mask=None, rect=None):
        try:
            import cv2
        except:
            logger.warning("Can't Do GrabCut without OpenCV >= 2.3.0")
            return
        ret = []
        if (mask is not None):
            bmp = mask._get_gray_narray()
            # translate the human readable images to something opencv wants using a lut
            LUT = np.zeros((256, 1), dtype=uint8)
            LUT[255] = 1
            LUT[64] = 2
            LUT[192] = 3
            cv2.LUT(bmp, bmp, cv2.fromarray(LUT))
            mask_in = np.array(cv2.GetMat(bmp))
            # get our image in a flavor grab cut likes
            npimg = np.array(cv2.GetMat(self.bitmap))
            # require by opencv
            tmp1 = np.zeros((1, 13 * 5))
            tmp2 = np.zeros((1, 13 * 5))
            # do the algorithm
            cv2.grabCut(npimg, mask_in, None, tmp1, tmp2, 10,
                        mode=cv2.GC_INIT_WITH_MASK)
            # generate the output image
            output = cv2.CreateImageHeader((mask_in.shape[1], mask_in.shape[0]),
                                          cv2.IPL_DEPTH_8U, 1)
            cv2.SetData(output, mask_in.tostring(),
                       mask_in.dtype.itemsize * mask_in.shape[1])
            # remap the _color space
            LUT = np.zeros((256, 1), dtype=uint8)
            LUT[1] = 255
            LUT[2] = 64
            LUT[3] = 192
            cv2.LUT(output, output, cv2.fromarray(LUT))
            # and create the return value
            mask._graybitmap = None  # don't ask me why... but this gets corrupted
            ret = Image(output)

        elif (rect is not None):
            npimg = np.array(cv2.GetMat(self.bitmap))
            tmp1 = np.zeros((1, 13 * 5))
            tmp2 = np.zeros((1, 13 * 5))
            mask = np.zeros((self.height, self.width), dtype='uint8')
            cv2.grabCut(npimg, mask, rect, tmp1, tmp2, 10,
                        mode=cv2.GC_INIT_WITH_RECT)
            bmp = cv2.CreateImageHeader((mask.shape[1], mask.shape[0]),
                                       cv2.IPL_DEPTH_8U, 1)
            cv2.SetData(bmp, mask.tostring(),
                       mask.dtype.itemsize * mask.shape[1])
            LUT = np.zeros((256, 1), dtype=uint8)
            LUT[1] = 255
            LUT[2] = 64
            LUT[3] = 192
            cv2.LUT(bmp, bmp, cv2.fromarray(LUT))
            ret = Image(bmp)
        else:
            logger.warning(
                    "Image.findBlobsSmart requires either a mask or a selection rectangle. Failure to provide one of these causes your bytes to splinter and bit shrapnel to hit your pipeline making it asplode in a ball of fire. Okay... not really")
        return ret

    def smart_find_blobs(self, mask=None, rect=None, thresh_level=2,
                         appx_level=3):
        result = self.smartThreshold(mask, rect)
        binary = None
        ret = None

        if result:
            if (thresh_level == 1):
                result = result.threshold(192)
            elif (thresh_level == 2):
                result = result.threshold(128)
            elif (thresh_level > 2):
                result = result.threshold(1)
            bm = BlobMaker()
            ret = bm.extract_from_binary(result, self, appx_level)

        return ret

    def threshold(self, value):
        gray = self._get_gray_narray()
        result = self.zeros(1)
        cv2.Threshold(gray, result, value, 255, cv2.CV_THRESH_BINARY)
        ret = Image(result)
        return ret

    def flood_fill(self, points, tolerance=None, color=Color.WHITE, lower=None,
                   upper=None, fixed_range=True):
        if (isinstance(color, np.ndarray)):
            color = color.tolist()
        elif (isinstance(color, dict)):
            color = (color['R'], color['G'], color['B'])

        if (isinstance(points, tuple)):
            points = np.array(points)
        # first we guess what the user wants to do
        # if we get and int/float convert it to a tuple
        if (upper is None and lower is None and tolerance is None):
            upper = (0, 0, 0)
            lower = (0, 0, 0)

        if (tolerance is not None and
                (isinstance(tolerance, float) or isinstance(tolerance, int))):
            tolerance = (int(tolerance), int(tolerance), int(tolerance))

        if (lower is not None and
                (isinstance(lower, float) or isinstance(lower, int))):
            lower = (int(lower), int(lower), int(lower))
        elif (lower is None):
            lower = tolerance

        if (upper is not None and
                (isinstance(upper, float) or isinstance(upper, int))):
            upper = (int(upper), int(upper), int(upper))
        elif (upper is None):
            upper = tolerance

        if (isinstance(points, tuple)):
            points = np.array(points)

        flags = 8
        if (fixed_range):
            flags = flags + cv2.CV_FLOODFILL_FIXED_RANGE

        bmp = self.zeros()
        cv2.Copy(self.bitmap, bmp)

        if (len(points.shape) != 1):
            for p in points:
                cv2.FloodFill(bmp, tuple(p), color, lower, upper, flags)
        else:
            cv2.FloodFill(bmp, tuple(points), color, lower, upper, flags)

        ret = Image(bmp)

        return ret

    def flood_fill_to_mask(self, points, tolerance=None, color=Color.WHITE,
                           lower=None, upper=None, fixed_range=True, mask=None):
        mask_flag = 255  # flag weirdness

        if (isinstance(color, np.ndarray)):
            color = color.tolist()
        elif (isinstance(color, dict)):
            color = (color['R'], color['G'], color['B'])

        if (isinstance(points, tuple)):
            points = np.array(points)

        # first we guess what the user wants to do
        # if we get and int/float convert it to a tuple
        if (upper is None and lower is None and tolerance is None):
            upper = (0, 0, 0)
            lower = (0, 0, 0)

        if (tolerance is not None and
                (isinstance(tolerance, float) or isinstance(tolerance, int))):
            tolerance = (int(tolerance), int(tolerance), int(tolerance))

        if (lower is not None and
                (isinstance(lower, float) or isinstance(lower, int))):
            lower = (int(lower), int(lower), int(lower))
        elif (lower is None):
            lower = tolerance

        if (upper is not None and
                (isinstance(upper, float) or isinstance(upper, int))):
            upper = (int(upper), int(upper), int(upper))
        elif (upper is None):
            upper = tolerance

        if (isinstance(points, tuple)):
            points = np.array(points)

        flags = (mask_flag << 8) + 8
        if (fixed_range):
            flags = flags + cv2.CV_FLOODFILL_FIXED_RANGE

        localMask = None
        # opencv wants a mask that is slightly larger
        if (mask is None):
            localMask = cv2.CreateImage((self.width + 2, self.height + 2),
                                       cv2.IPL_DEPTH_8U, 1)
            cv2.Zero(localMask)
        else:
            localMask = mask.embiggen(
                    size=(
                    self.width + 2, self.height + 2))._get_gray_narray()

        bmp = self.zeros()
        cv2.Copy(self.bitmap, bmp)
        if (len(points.shape) != 1):
            for p in points:
                cv2.FloodFill(bmp, tuple(p), color, lower, upper, flags,
                             localMask)
        else:
            cv2.FloodFill(bmp, tuple(points), color, lower, upper, flags,
                         localMask)

        ret = Image(localMask)
        ret = ret.crop(1, 1, self.width, self.height)
        return ret

    def find_blobs_from_mask(self, mask, threshold=128, minsize=10, maxsize=0,
                             appx_level=3):
        if (maxsize == 0):
            maxsize = self.width * self.height
        # create a single channel image, thresholded to parameters
        if (mask.width != self.width or mask.height != self.height):
            logger.warning(
                    "Image.find_blobs_from_mask - your mask does not match the size of your image")
            return None

        blobmaker = BlobMaker()
        gray = mask._get_gray_narray()
        result = mask.zeros(1)
        cv2.Threshold(gray, result, threshold, 255, cv2.CV_THRESH_BINARY)
        blobs = blobmaker.extract_from_binary(Image(result), self,
                                            minsize=minsize,
                                            maxsize=maxsize,
                                            appx_level=appx_level)

        if not len(blobs):
            return None

        return FeatureSet(blobs).sortArea()

    def find_flood_fill_blobs(self, points, tolerance=None, lower=None,
                              upper=None,
                              fixed_range=True, minsize=30, maxsize=-1):
        mask = self.floodFillToMask(points, tolerance, color=Color.WHITE,
                                    lower=lower, upper=upper,
                                    fixed_range=fixed_range)
        return self.find_blobs_from_mask(mask, minsize, maxsize)

    def _do_DFT(self, grayscale=False):
        if (grayscale and (len(self._DFT) == 0 or len(self._DFT) == 3)):
            self._DFT = []
            img = self._get_gray_narray()
            width, height = cv2.GetSize(img)
            src = cv2.CreateImage((width, height), cv2.IPL_DEPTH_64F, 2)
            dst = cv2.CreateImage((width, height), cv2.IPL_DEPTH_64F, 2)
            data = cv2.CreateImage((width, height), cv2.IPL_DEPTH_64F, 1)
            blank = cv2.CreateImage((width, height), cv2.IPL_DEPTH_64F, 1)
            cv2.ConvertScale(img, data, 1.0)
            cv2.Zero(blank)
            cv2.Merge(data, blank, None, None, src)
            cv2.Merge(data, blank, None, None, dst)
            cv2.DFT(src, dst, cv2.CV_DXT_FORWARD)
            self._DFT.append(dst)
        elif (not grayscale and (len(self._DFT) < 2)):
            self._DFT = []
            r = self.zeros(1)
            g = self.zeros(1)
            b = self.zeros(1)
            cv2.Split(self.bitmap, b, g, r, None)
            chans = [b, g, r]
            width = self.width
            height = self.height
            data = cv2.CreateImage((width, height), cv2.IPL_DEPTH_64F, 1)
            blank = cv2.CreateImage((width, height), cv2.IPL_DEPTH_64F, 1)
            src = cv2.CreateImage((width, height), cv2.IPL_DEPTH_64F, 2)
            for c in chans:
                dst = cv2.CreateImage((width, height), cv2.IPL_DEPTH_64F, 2)
                cv2.ConvertScale(c, data, 1.0)
                cv2.Zero(blank)
                cv2.Merge(data, blank, None, None, src)
                cv2.Merge(data, blank, None, None, dst)
                cv2.DFT(src, dst, cv2.CV_DXT_FORWARD)
                self._DFT.append(dst)

    def _get_DFT_clone(self, grayscale=False):
        self._doDFT(grayscale)
        ret = []
        if (grayscale):
            gs = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_64F, 2)
            cv2.Copy(self._DFT[0], gs)
            ret.append(gs)
        else:
            for img in self._DFT:
                temp = cv2.CreateImage((self.width, self.height),
                                      cv2.IPL_DEPTH_64F,
                                      2)
                cv2.Copy(img, temp)
                ret.append(temp)
        return ret

    def raw_DFT_image(self, grayscale=False):
        self._doDFT(grayscale)
        return self._DFT

    def get_DFT_log_magnitude(self, grayscale=False):
        dft = self._getDFTClone(grayscale)
        chans = []
        if (grayscale):
            chans = [self.zeros(1)]
        else:
            chans = [self.zeros(1), self.zeros(1), self.zeros(1)]
        data = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_64F, 1)
        blank = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_64F, 1)

        for i in range(0, len(chans)):
            cv2.Split(dft[i], data, blank, None, None)
            cv2.Pow(data, data, 2.0)
            cv2.Pow(blank, blank, 2.0)
            cv2.Add(data, blank, data, None)
            cv2.Pow(data, data, 0.5)
            cv2.AddS(data, cv2.ScalarAll(1.0), data, None)  # 1 + Mag
            cv2.Log(data, data)  # log(1 + Mag
            min, max, pt1, pt2 = cv2.MinMaxLoc(data)
            cv2.Scale(data, data, 1.0 / (max - min), 1.0 * (-min) / (max - min))
            cv2.Mul(data, data, data, 255.0)
            cv2.Convert(data, chans[i])

        ret = None
        if grayscale:
            ret = Image(chans[0])
        else:
            ret = self.zeros()
            cv2.Merge(chans[0], chans[1], chans[2], None, ret)
            ret = Image(ret)
        return ret

    def _bounds_from_percentage(self, floatVal, bound):
        return np.clip(int(floatVal * bound), 0, bound)

    def apply_DFT_filter(self, flt, grayscale=False):
        if isinstance(flt, DFT):
            filteredimage = flt.applyFilter(self, grayscale)
            return filteredimage

        if (flt.width != self.width and
                    flt.height != self.height):
            logger.warning(
                    "Image.applyDFTFilter - Your filter must match the size of the image")
        dft = []
        if (grayscale):
            dft = self._getDFTClone(grayscale)
            flt = flt._get_gray_narray()
            flt64f = cv2.CreateImage((flt.width, flt.height), cv2.IPL_DEPTH_64F,
                                    1)
            cv2.ConvertScale(flt, flt64f, 1.0)
            finalFilt = cv2.CreateImage((flt.width, flt.height),
                                       cv2.IPL_DEPTH_64F, 2)
            cv2.Merge(flt64f, flt64f, None, None, finalFilt)
            for d in dft:
                cv2.MulSpectrums(d, finalFilt, d, 0)
        else:  # break down the filter and then do each channel
            dft = self._getDFTClone(grayscale)
            flt = flt.bitmap
            b = cv2.CreateImage((flt.width, flt.height), cv2.IPL_DEPTH_8U, 1)
            g = cv2.CreateImage((flt.width, flt.height), cv2.IPL_DEPTH_8U, 1)
            r = cv2.CreateImage((flt.width, flt.height), cv2.IPL_DEPTH_8U, 1)
            cv2.Split(flt, b, g, r, None)
            chans = [b, g, r]
            for c in range(0, len(chans)):
                flt64f = cv2.CreateImage((chans[c].width, chans[c].height),
                                        cv2.IPL_DEPTH_64F, 1)
                cv2.ConvertScale(chans[c], flt64f, 1.0)
                finalFilt = cv2.CreateImage((chans[c].width, chans[c].height),
                                           cv2.IPL_DEPTH_64F, 2)
                cv2.Merge(flt64f, flt64f, None, None, finalFilt)
                cv2.MulSpectrums(dft[c], finalFilt, dft[c], 0)

        return self._inverseDFT(dft)

    def _bounds_from_percentage(self, floatVal, bound):
        return np.clip(int(floatVal * (bound / 2.00)), 0, (bound / 2))

    def high_pass_filter(self, xCutoff, yCutoff=None, grayscale=False):
        if (isinstance(xCutoff, float)):
            xCutoff = [xCutoff, xCutoff, xCutoff]
        if (isinstance(yCutoff, float)):
            yCutoff = [yCutoff, yCutoff, yCutoff]
        if (yCutoff is None):
            yCutoff = [xCutoff[0], xCutoff[1], xCutoff[2]]

        for i in range(0, len(xCutoff)):
            xCutoff[i] = self._boundsFromPercentage(xCutoff[i], self.width)
            yCutoff[i] = self._boundsFromPercentage(yCutoff[i], self.height)

        filter = None
        h = self.height
        w = self.width

        if (grayscale):
            filter = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_8U,
                                    1)
            cv2.Zero(filter)
            cv2.AddS(filter, 255, filter)  # make everything white
            # now make all of the corners black
            cv2.Rectangle(filter, (0, 0), (xCutoff[0], yCutoff[0]), (0, 0, 0),
                         thickness=-1)  # TL
            cv2.Rectangle(filter, (0, h - yCutoff[0]), (xCutoff[0], h),
                         (0, 0, 0), thickness=-1)  # BL
            cv2.Rectangle(filter, (w - xCutoff[0], 0), (w, yCutoff[0]),
                         (0, 0, 0), thickness=-1)  # TR
            cv2.Rectangle(filter, (w - xCutoff[0], h - yCutoff[0]), (w, h),
                         (0, 0, 0), thickness=-1)  # BR

        else:
            # I need to looking into CVMERGE/SPLIT... I would really need to know
            # how much memory we're allocating here
            filterB = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_8U,
                                     1)
            filterG = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_8U,
                                     1)
            filterR = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_8U,
                                     1)
            cv2.Zero(filterB)
            cv2.Zero(filterG)
            cv2.Zero(filterR)
            cv2.AddS(filterB, 255, filterB)  # make everything white
            cv2.AddS(filterG, 255, filterG)  # make everything whit
            cv2.AddS(filterR, 255, filterR)  # make everything white
            # now make all of the corners black
            temp = [filterB, filterG, filterR]
            i = 0
            for f in temp:
                cv2.Rectangle(f, (0, 0), (xCutoff[i], yCutoff[i]), 0,
                             thickness=-1)
                cv2.Rectangle(f, (0, h - yCutoff[i]), (xCutoff[i], h), 0,
                             thickness=-1)
                cv2.Rectangle(f, (w - xCutoff[i], 0), (w, yCutoff[i]), 0,
                             thickness=-1)
                cv2.Rectangle(f, (w - xCutoff[i], h - yCutoff[i]), (w, h), 0,
                             thickness=-1)
                i = i + 1

            filter = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_8U,
                                    3)
            cv2.Merge(filterB, filterG, filterR, None, filter)

        scvFilt = Image(filter)
        ret = self.applyDFTFilter(scvFilt, grayscale)
        return ret

    def low_pass_filter(self, xCutoff, yCutoff=None, grayscale=False):
        if (isinstance(xCutoff, float)):
            xCutoff = [xCutoff, xCutoff, xCutoff]
        if (isinstance(yCutoff, float)):
            yCutoff = [yCutoff, yCutoff, yCutoff]
        if (yCutoff is None):
            yCutoff = [xCutoff[0], xCutoff[1], xCutoff[2]]

        for i in range(0, len(xCutoff)):
            xCutoff[i] = self._boundsFromPercentage(xCutoff[i], self.width)
            yCutoff[i] = self._boundsFromPercentage(yCutoff[i], self.height)

        filter = None
        h = self.height
        w = self.width

        if (grayscale):
            filter = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_8U,
                                    1)
            cv2.Zero(filter)
            # now make all of the corners black

            cv2.Rectangle(filter, (0, 0), (xCutoff[0], yCutoff[0]), 255,
                         thickness=-1)  # TL
            cv2.Rectangle(filter, (0, h - yCutoff[0]), (xCutoff[0], h), 255,
                         thickness=-1)  # BL
            cv2.Rectangle(filter, (w - xCutoff[0], 0), (w, yCutoff[0]), 255,
                         thickness=-1)  # TR
            cv2.Rectangle(filter, (w - xCutoff[0], h - yCutoff[0]), (w, h), 255,
                         thickness=-1)  # BR

        else:
            # I need to looking into CVMERGE/SPLIT... I would really need to know
            # how much memory we're allocating here
            filterB = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_8U,
                                     1)
            filterG = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_8U,
                                     1)
            filterR = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_8U,
                                     1)
            cv2.Zero(filterB)
            cv2.Zero(filterG)
            cv2.Zero(filterR)
            # now make all of the corners black
            temp = [filterB, filterG, filterR]
            i = 0
            for f in temp:
                cv2.Rectangle(f, (0, 0), (xCutoff[i], yCutoff[i]), 255,
                             thickness=-1)
                cv2.Rectangle(f, (0, h - yCutoff[i]), (xCutoff[i], h), 255,
                             thickness=-1)
                cv2.Rectangle(f, (w - xCutoff[i], 0), (w, yCutoff[i]), 255,
                             thickness=-1)
                cv2.Rectangle(f, (w - xCutoff[i], h - yCutoff[i]), (w, h), 255,
                             thickness=-1)
                i = i + 1

            filter = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_8U,
                                    3)
            cv2.Merge(filterB, filterG, filterR, None, filter)

        scvFilt = Image(filter)
        ret = self.applyDFTFilter(scvFilt, grayscale)
        return ret

    def band_pass_filter(self, xCutoffLow, xCutoffHigh,
                         yCutoffLow=None, yCutoffHigh=None,
                         grayscale=False):
        if (isinstance(xCutoffLow, float)):
            xCutoffLow = [xCutoffLow, xCutoffLow, xCutoffLow]
        if (isinstance(yCutoffLow, float)):
            yCutoffLow = [yCutoffLow, yCutoffLow, yCutoffLow]
        if (isinstance(xCutoffHigh, float)):
            xCutoffHigh = [xCutoffHigh, xCutoffHigh, xCutoffHigh]
        if (isinstance(yCutoffHigh, float)):
            yCutoffHigh = [yCutoffHigh, yCutoffHigh, yCutoffHigh]

        if (yCutoffLow is None):
            yCutoffLow = [xCutoffLow[0], xCutoffLow[1], xCutoffLow[2]]
        if (yCutoffHigh is None):
            yCutoffHigh = [xCutoffHigh[0], xCutoffHigh[1], xCutoffHigh[2]]

        for i in range(0, len(xCutoffLow)):
            xCutoffLow[i] = self._boundsFromPercentage(xCutoffLow[i],
                                                       self.width)
            xCutoffHigh[i] = self._boundsFromPercentage(xCutoffHigh[i],
                                                        self.width)
            yCutoffHigh[i] = self._boundsFromPercentage(yCutoffHigh[i],
                                                        self.height)
            yCutoffLow[i] = self._boundsFromPercentage(yCutoffLow[i],
                                                       self.height)

        filter = None
        h = self.height
        w = self.width
        if (grayscale):
            filter = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_8U,
                                    1)
            cv2.Zero(filter)
            # now make all of the corners black
            cv2.Rectangle(filter, (0, 0), (xCutoffHigh[0], yCutoffHigh[0]), 255,
                         thickness=-1)  # TL
            cv2.Rectangle(filter, (0, h - yCutoffHigh[0]), (xCutoffHigh[0], h),
                         255, thickness=-1)  # BL
            cv2.Rectangle(filter, (w - xCutoffHigh[0], 0), (w, yCutoffHigh[0]),
                         255, thickness=-1)  # TR
            cv2.Rectangle(filter, (w - xCutoffHigh[0], h - yCutoffHigh[0]),
                         (w, h), 255, thickness=-1)  # BR
            cv2.Rectangle(filter, (0, 0), (xCutoffLow[0], yCutoffLow[0]), 0,
                         thickness=-1)  # TL
            cv2.Rectangle(filter, (0, h - yCutoffLow[0]), (xCutoffLow[0], h), 0,
                         thickness=-1)  # BL
            cv2.Rectangle(filter, (w - xCutoffLow[0], 0), (w, yCutoffLow[0]), 0,
                         thickness=-1)  # TR
            cv2.Rectangle(filter, (w - xCutoffLow[0], h - yCutoffLow[0]), (w, h),
                         0, thickness=-1)  # BR


        else:
            # I need to looking into CVMERGE/SPLIT... I would really need to know
            # how much memory we're allocating here
            filterB = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_8U,
                                     1)
            filterG = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_8U,
                                     1)
            filterR = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_8U,
                                     1)
            cv2.Zero(filterB)
            cv2.Zero(filterG)
            cv2.Zero(filterR)
            # now make all of the corners black
            temp = [filterB, filterG, filterR]
            i = 0
            for f in temp:
                cv2.Rectangle(f, (0, 0), (xCutoffHigh[i], yCutoffHigh[i]), 255,
                             thickness=-1)  # TL
                cv2.Rectangle(f, (0, h - yCutoffHigh[i]), (xCutoffHigh[i], h),
                             255, thickness=-1)  # BL
                cv2.Rectangle(f, (w - xCutoffHigh[i], 0), (w, yCutoffHigh[i]),
                             255, thickness=-1)  # TR
                cv2.Rectangle(f, (w - xCutoffHigh[i], h - yCutoffHigh[i]),
                             (w, h), 255, thickness=-1)  # BR
                cv2.Rectangle(f, (0, 0), (xCutoffLow[i], yCutoffLow[i]), 0,
                             thickness=-1)  # TL
                cv2.Rectangle(f, (0, h - yCutoffLow[i]), (xCutoffLow[i], h), 0,
                             thickness=-1)  # BL
                cv2.Rectangle(f, (w - xCutoffLow[i], 0), (w, yCutoffLow[i]), 0,
                             thickness=-1)  # TR
                cv2.Rectangle(f, (w - xCutoffLow[i], h - yCutoffLow[i]), (w, h),
                             0, thickness=-1)  # BR
                i = i + 1

            filter = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_8U,
                                    3)
            cv2.Merge(filterB, filterG, filterR, None, filter)

        scvFilt = Image(filter)
        ret = self.applyDFTFilter(scvFilt, grayscale)
        return ret

    def _inverse_DFT(self, input):
        # a destructive IDFT operation for internal calls
        w = input[0].width
        h = input[0].height
        if (len(input) == 1):
            cv2.DFT(input[0], input[0], cv2.CV_DXT_INV_SCALE)
            result = cv2.CreateImage((w, h), cv2.IPL_DEPTH_8U, 1)
            data = cv2.CreateImage((w, h), cv2.IPL_DEPTH_64F, 1)
            blank = cv2.CreateImage((w, h), cv2.IPL_DEPTH_64F, 1)
            cv2.Split(input[0], data, blank, None, None)
            min, max, pt1, pt2 = cv2.MinMaxLoc(data)
            denom = max - min
            if (denom == 0):
                denom = 1
            cv2.Scale(data, data, 1.0 / (denom), 1.0 * (-min) / (denom))
            cv2.Mul(data, data, data, 255.0)
            cv2.Convert(data, result)
            ret = Image(result)
        else:  # DO RGB separately
            results = []
            data = cv2.CreateImage((w, h), cv2.IPL_DEPTH_64F, 1)
            blank = cv2.CreateImage((w, h), cv2.IPL_DEPTH_64F, 1)
            for i in range(0, len(input)):
                cv2.DFT(input[i], input[i], cv2.CV_DXT_INV_SCALE)
                result = cv2.CreateImage((w, h), cv2.IPL_DEPTH_8U, 1)
                cv2.Split(input[i], data, blank, None, None)
                min, max, pt1, pt2 = cv2.MinMaxLoc(data)
                denom = max - min
                if (denom == 0):
                    denom = 1
                cv2.Scale(data, data, 1.0 / (denom), 1.0 * (-min) / (denom))
                cv2.Mul(data, data, data, 255.0)  # this may not be right
                cv2.Convert(data, result)
                results.append(result)

            ret = cv2.CreateImage((w, h), cv2.IPL_DEPTH_8U, 3)
            cv2.Merge(results[0], results[1], results[2], None, ret)
            ret = Image(ret)
        del input
        return ret

    def inverse_dft(self, raw_dft_image):
        input = []
        w = raw_dft_image[0].width
        h = raw_dft_image[0].height
        if (len(raw_dft_image) == 1):
            gs = cv2.CreateImage((w, h), cv2.IPL_DEPTH_64F, 2)
            cv2.Copy(self._DFT[0], gs)
            input.append(gs)
        else:
            for img in raw_dft_image:
                temp = cv2.CreateImage((w, h), cv2.IPL_DEPTH_64F, 2)
                cv2.Copy(img, temp)
                input.append(img)

        if (len(input) == 1):
            cv2.DFT(input[0], input[0], cv2.CV_DXT_INV_SCALE)
            result = cv2.CreateImage((w, h), cv2.IPL_DEPTH_8U, 1)
            data = cv2.CreateImage((w, h), cv2.IPL_DEPTH_64F, 1)
            blank = cv2.CreateImage((w, h), cv2.IPL_DEPTH_64F, 1)
            cv2.Split(input[0], data, blank, None, None)
            min, max, pt1, pt2 = cv2.MinMaxLoc(data)
            denom = max - min
            if (denom == 0):
                denom = 1
            cv2.Scale(data, data, 1.0 / (denom), 1.0 * (-min) / (denom))
            cv2.Mul(data, data, data, 255.0)
            cv2.Convert(data, result)
            ret = Image(result)
        else:  # DO RGB separately
            results = []
            data = cv2.CreateImage((w, h), cv2.IPL_DEPTH_64F, 1)
            blank = cv2.CreateImage((w, h), cv2.IPL_DEPTH_64F, 1)
            for i in range(0, len(raw_dft_image)):
                cv2.DFT(input[i], input[i], cv2.CV_DXT_INV_SCALE)
                result = cv2.CreateImage((w, h), cv2.IPL_DEPTH_8U, 1)
                cv2.Split(input[i], data, blank, None, None)
                min, max, pt1, pt2 = cv2.MinMaxLoc(data)
                denom = max - min
                if (denom == 0):
                    denom = 1
                cv2.Scale(data, data, 1.0 / (denom), 1.0 * (-min) / (denom))
                cv2.Mul(data, data, data, 255.0)  # this may not be right
                cv2.Convert(data, result)
                results.append(result)

            ret = cv2.CreateImage((w, h), cv2.IPL_DEPTH_8U, 3)
            cv2.Merge(results[0], results[1], results[2], None, ret)
            ret = Image(ret)

        return ret

    def apply_butterworth_filter(self, dia=400, order=2, highpass=False,
                                 grayscale=False):
        # reimplemented with faster, vectorized filter kernel creation
        w, h = self.size()
        intensity_scale = 2 ** 8 - 1  # for now 8-bit
        sz_x = 64  # for now constant, symmetric
        sz_y = 64  # for now constant, symmetric
        x0 = sz_x / 2.0  # for now, on center
        y0 = sz_y / 2.0  # for now, on center
        # efficient "vectorized" computation
        X, Y = np.meshgrid(np.arange(sz_x), np.arange(sz_y))
        D = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
        flt = intensity_scale / (1.0 + (D / dia) ** (order * 2))
        if highpass:  # then invert the filter
            flt = intensity_scale - flt
        flt = Image(
                flt)  # numpy arrays are in row-major form...doesn't matter for symmetric filter
        flt_re = flt.resize(w, h)
        img = self.applyDFTFilter(flt_re, grayscale)
        return img

    def apply_gaussian_filter(self, dia=400, highpass=False, grayscale=False):
        # reimplemented with faster, vectorized filter kernel creation
        w, h = self.size()
        intensity_scale = 2 ** 8 - 1  # for now 8-bit
        sz_x = 64  # for now constant, symmetric
        sz_y = 64  # for now constant, symmetric
        x0 = sz_x / 2.0  # for now, on center
        y0 = sz_y / 2.0  # for now, on center
        # efficient "vectorized" computation
        X, Y = np.meshgrid(np.arange(sz_x), np.arange(sz_y))
        D = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
        flt = intensity_scale * np.exp(-0.5 * (D / dia) ** 2)
        if highpass:  # then invert the filter
            flt = intensity_scale - flt
        flt = Image(
                flt)  # numpy arrays are in row-major form...doesn't matter for symmetric filter
        flt_re = flt.resize(w, h)
        img = self.applyDFTFilter(flt_re, grayscale)
        return img

    def apply_unsharp_mask(self, boost=1, dia=400, grayscale=False):
        if boost < 0:
            print
            "boost >= 1"
            return None

        lpIm = self.applyGaussianFilter(dia=dia, grayscale=grayscale,
                                        highpass=False)
        im = Image(self.bitmap)
        mask = im - lpIm
        img = im
        for i in range(boost):
            img = img + mask
        return img

    def list_haar_features(self):
        features_directory = os.path.join(LAUNCH_PATH, 'Features',
                                          'HaarCascades')
        features = os.listdir(features_directory)
        print
        features

    def _copy_avg(self, src, dst, roi, levels, levels_f, mode):
        if (mode):  # get the peak hue for an area
            h = src[roi[0]:roi[0] + roi[2],
                roi[1]:roi[1] + roi[3]].hueHistogram()
            myHue = np.argmax(h)
            C = (float(myHue), float(255), float(255), float(0))
            cv2.SetImageROI(dst, roi)
            cv2.AddS(dst, c, dst)
            cv2.ResetImageROI(dst)
        else:  # get the average value for an area optionally set levels
            cv2.SetImageROI(src.bitmap, roi)
            cv2.SetImageROI(dst, roi)
            avg = cv2.Avg(src.bitmap)
            avg = (float(avg[0]), float(avg[1]), float(avg[2]), 0)
            if levels is not None:
                avg = (int(avg[0] / levels) * levels_f,
                       int(avg[1] / levels) * levels_f,
                       int(avg[2] / levels) * levels_f, 0)
            cv2.AddS(dst, avg, dst)
            cv2.ResetImageROI(src.bitmap)
            cv2.ResetImageROI(dst)

    def pixelize(self, block_size=10, region=None, levels=None, doHue=False):
        if isinstance(block_size, int):
            block_size = (block_size, block_size)

        ret = self.zeros()

        levels_f = 0.00
        if levels is not None:
            levels = 255 / int(levels)
            if (levels <= 1):
                levels = 2
            levels_f = float(levels)

        if region is not None:
            cv2.Copy(self.bitmap, ret)
            cv2.SetImageROI(ret, region)
            cv2.Zero(ret)
            cv2.ResetImageROI(ret)
            xs = region[0]
            ys = region[1]
            w = region[2]
            h = region[3]
        else:
            xs = 0
            ys = 0
            w = self.width
            h = self.height

        # if( region is None ):
        hc = w / block_size[0]  # number of horizontal blocks
        vc = h / block_size[1]  # number of vertical blocks
        # when we fit in the blocks, we're going to spread the round off
        # over the edges 0->x_0, 0->y_0  and x_0+hc*block_size
        x_lhs = int(np.ceil(
                float(w % block_size[0]) / 2.0))  # this is the starting point
        y_lhs = int(np.ceil(float(h % block_size[1]) / 2.0))
        x_rhs = int(np.floor(
                float(w % block_size[0]) / 2.0))  # this is the starting point
        y_rhs = int(np.floor(float(h % block_size[1]) / 2.0))
        x_0 = xs + x_lhs
        y_0 = ys + y_lhs
        x_f = (x_0 + (block_size[0] * hc))  # this would be the end point
        y_f = (y_0 + (block_size[1] * vc))

        for i in range(0, hc):
            for j in range(0, vc):
                xt = x_0 + (block_size[0] * i)
                yt = y_0 + (block_size[1] * j)
                roi = (xt, yt, block_size[0], block_size[1])
                self._CopyAvg(self, ret, roi, levels, levels_f, doHue)

        if (x_lhs > 0):  # add a left strip
            xt = xs
            wt = x_lhs
            ht = block_size[1]
            for j in range(0, vc):
                yt = y_0 + (j * block_size[1])
                roi = (xt, yt, wt, ht)
                self._CopyAvg(self, ret, roi, levels, levels_f, doHue)

        if (x_rhs > 0):  # add a right strip
            xt = (x_0 + (block_size[0] * hc))
            wt = x_rhs
            ht = block_size[1]
            for j in range(0, vc):
                yt = y_0 + (j * block_size[1])
                roi = (xt, yt, wt, ht)
                self._CopyAvg(self, ret, roi, levels, levels_f, doHue)

        if (y_lhs > 0):  # add a left strip
            yt = ys
            ht = y_lhs
            wt = block_size[0]
            for i in range(0, hc):
                xt = x_0 + (i * block_size[0])
                roi = (xt, yt, wt, ht)
                self._CopyAvg(self, ret, roi, levels, levels_f, doHue)

        if (y_rhs > 0):  # add a right strip
            yt = (y_0 + (block_size[1] * vc))
            ht = y_rhs
            wt = block_size[0]
            for i in range(0, hc):
                xt = x_0 + (i * block_size[0])
                roi = (xt, yt, wt, ht)
                self._CopyAvg(self, ret, roi, levels, levels_f, doHue)

        # now the corner cases
        if (x_lhs > 0 and y_lhs > 0):
            roi = (xs, ys, x_lhs, y_lhs)
            self._CopyAvg(self, ret, roi, levels, levels_f, doHue)

        if (x_rhs > 0 and y_rhs > 0):
            roi = (x_f, y_f, x_rhs, y_rhs)
            self._CopyAvg(self, ret, roi, levels, levels_f, doHue)

        if (x_lhs > 0 and y_rhs > 0):
            roi = (xs, y_f, x_lhs, y_rhs)
            self._CopyAvg(self, ret, roi, levels, levels_f, doHue)

        if (x_rhs > 0 and y_lhs > 0):
            roi = (x_f, ys, x_rhs, y_lhs)
            self._CopyAvg(self, ret, roi, levels, levels_f, doHue)

        if (doHue):
            cv2.cvtColor(ret, ret, cv2.CV_HSV2BGR)

        return Image(ret)

    def anonymize(self, block_size=10, features=None, transform=None):
        regions = []

        if features is None:
            regions.append(self.findHaarFeatures("face"))
            regions.append(self.findHaarFeatures("profile"))
        else:
            for feature in features:
                regions.append(self.findHaarFeatures(feature))

        found = [f for f in regions if f is not None]

        img = self.copy()

        if found:
            for feature_set in found:
                for region in feature_set:
                    rect = (
                        region.topLeftCorner()[0], region.topLeftCorner()[1],
                        region.width(), region.height())
                    if transform is None:
                        img = img.pixelize(block_size=block_size, region=rect)
                    else:
                        img = transform(img, rect)

        return img

    def fill_holes(self):
        des = cv2.bitwise_not(self.gray_narray)
        return cv2.inPaint(des)
        contour, hier = cv2.findContours(des, cv2.RETR_CCOMP,
                                         cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contour:
            cv2.drawContours(des, [cnt], 0, 255, -1)
            print
            'yep'

        gray = cv2.bitwise_not(des)
        return gray

    def edge_intersections(self, pt0, pt1, width=1, canny1=0, canny2=100):
        w = abs(pt0[0] - pt1[0])
        h = abs(pt0[1] - pt1[1])
        x = np.min([pt0[0], pt1[0]])
        y = np.min([pt0[1], pt1[1]])
        if (w <= 0):
            w = width
            x = np.clip(x - (width / 2), 0, x - (width / 2))
        if (h <= 0):
            h = width
            y = np.clip(y - (width / 2), 0, y - (width / 2))
        # got some corner cases to catch here
        p0p = np.array([(pt0[0] - x, pt0[1] - y)])
        p1p = np.array([(pt1[0] - x, pt1[1] - y)])
        edges = self.crop(x, y, w, h)._getEdgeMap(canny1, canny2)
        line = cv2.CreateImage((w, h), cv2.IPL_DEPTH_8U, 1)
        cv2.Zero(line)
        cv2.Line(line, ((pt0[0] - x), (pt0[1] - y)),
                ((pt1[0] - x), (pt1[1] - y)), cv2.Scalar(255.00), width, 8)
        cv2.Mul(line, edges, line)
        intersections = uint8(np.array(cv2.GetMat(line)).transpose())
        (xs, ys) = np.where(intersections == 255)
        points = zip(xs, ys)
        if (len(points) == 0):
            return [None, None]
        A = np.argmin(spsd.cdist(p0p, points, 'cityblock'))
        B = np.argmin(spsd.cdist(p1p, points, 'cityblock'))
        ptA = (int(xs[A] + x), int(ys[A] + y))
        ptB = (int(xs[B] + x), int(ys[B] + y))
        # we might actually want this to be list of all the points
        return [ptA, ptB]

    def fit_contour(self, initial_curve, window=(11, 11),
                    params=(0.1, 0.1, 0.1),
                    doAppx=True, appx_level=1):
        alpha = [params[0]]
        beta = [params[1]]
        gamma = [params[2]]
        if (window[0] % 2 == 0):
            window = (window[0] + 1, window[1])
            logger.warn(
                "Yo dawg, just a heads up, snakeFitPoints wants an odd window size. I fixed it for you, but you may want to take a look at your code.")
        if (window[1] % 2 == 0):
            window = (window[0], window[1] + 1)
            logger.warn(
                "Yo dawg, just a heads up, snakeFitPoints wants an odd window size. I fixed it for you, but you may want to take a look at your code.")
        raw = cv2.SnakeImage(self._get_gray_narray(), initial_curve, alpha,
                             beta, gamma, window,
                             (cv2.CV_TERMCRIT_ITER, 10, 0.01))
        if (doAppx):
            try:
                import cv2
            except:
                logger.warning(
                    "Can't Do snakeFitPoints without OpenCV >= 2.3.0")
                return
            appx = cv2.approxPolyDP(np.array([raw], 'float32'), appx_level,
                                    True)
            ret = []
            for p in appx:
                ret.append((int(p[0][0]), int(p[0][1])))
        else:
            ret = raw

        return ret

    def fit_edge(self, guess, window=10, threshold=128, measurements=5,
                 darktolight=True, lighttodark=True, departurethreshold=1):
        searchLines = FeatureSet()
        fitPoints = FeatureSet()
        x1 = guess[0][0]
        x2 = guess[1][0]
        y1 = guess[0][1]
        y2 = guess[1][1]
        dx = float((x2 - x1)) / (measurements - 1)
        dy = float((y2 - y1)) / (measurements - 1)
        s = np.zeros((measurements, 2))
        lpstartx = np.zeros(measurements)
        lpstarty = np.zeros(measurements)
        lpendx = np.zeros(measurements)
        lpendy = np.zeros(measurements)
        linefitpts = np.zeros((measurements, 2))

        # obtain equation for initial guess line
        if (
                    x1 == x2):  # vertical line must be handled as special case since slope isn't defined
            m = 0
            mo = 0
            b = x1
            for i in xrange(0, measurements):
                s[i][0] = x1
                s[i][1] = y1 + i * dy
                lpstartx[i] = s[i][0] + window
                lpstarty[i] = s[i][1]
                lpendx[i] = s[i][0] - window
                lpendy[i] = s[i][1]
                Cur_line = Line(self, (
                    (lpstartx[i], lpstarty[i]), (lpendx[i], lpendy[i])))
                ((lpstartx[i], lpstarty[i]), (
                    lpendx[i],
                    lpendy[i])) = Cur_line.crop_to_image_edges().end_points

                searchLines.append(Cur_line)
                tmp = self.getThresholdCrossing(
                        (int(lpstartx[i]), int(lpstarty[i])),
                        (int(lpendx[i]), int(lpendy[i])), threshold=threshold,
                        lighttodark=lighttodark, darktolight=darktolight,
                        departurethreshold=departurethreshold)
                fitPoints.append(Circle(self, tmp[0], tmp[1], 3))
                linefitpts[i] = tmp

        else:
            m = float((y2 - y1)) / (x2 - x1)
            b = y1 - m * x1
            mo = -1 / m  # slope of orthogonal line segments

            # obtain points for measurement along the initial guess line
            for i in xrange(0, measurements):
                s[i][0] = x1 + i * dx
                s[i][1] = y1 + i * dy
                fx = (math.sqrt(math.pow(window, 2)) / (1 + mo)) / 2
                fy = fx * mo
                lpstartx[i] = s[i][0] + fx
                lpstarty[i] = s[i][1] + fy
                lpendx[i] = s[i][0] - fx
                lpendy[i] = s[i][1] - fy
                Cur_line = Line(self, (
                    (lpstartx[i], lpstarty[i]), (lpendx[i], lpendy[i])))
                ((lpstartx[i], lpstarty[i]), (
                    lpendx[i],
                    lpendy[i])) = Cur_line.crop_to_image_edges().end_points
                searchLines.append(Cur_line)
                tmp = self.getThresholdCrossing(
                        (int(lpstartx[i]), int(lpstarty[i])),
                        (int(lpendx[i]), int(lpendy[i])), threshold=threshold,
                        lighttodark=lighttodark, darktolight=darktolight,
                        departurethreshold=departurethreshold)
                fitPoints.append((tmp[0], tmp[1]))
                linefitpts[i] = tmp

        badpts = []
        for j in range(len(linefitpts)):
            if (linefitpts[j, 0] == -1) or (linefitpts[j, 1] == -1):
                badpts.append(j)
        for pt in badpts:
            linefitpts = np.delete(linefitpts, pt, axis=0)

        x = linefitpts[:, 0]
        y = linefitpts[:, 1]
        ymin = np.min(y)
        ymax = np.max(y)
        xmax = np.max(x)
        xmin = np.min(x)

        if ((xmax - xmin) > (ymax - ymin)):
            # do the least squares
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = nla.lstsq(A, y)[0]
            y0 = int(m * xmin + c)
            y1 = int(m * xmax + c)
            finalLine = Line(self, ((xmin, y0), (xmax, y1)))
        else:
            # do the least squares
            A = np.vstack([y, np.ones(len(y))]).T
            m, c = nla.lstsq(A, x)[0]
            x0 = int(ymin * m + c)
            x1 = int(ymax * m + c)
            finalLine = Line(self, ((x0, ymin), (x1, ymax)))

        return finalLine, searchLines, fitPoints

    def get_threshold_crossing(self, pt1, pt2, thresh=128, darktolight=True,
                               lighttodark=True, departurethreshold=1):
        linearr = self.getDiagonalScanlineGrey(pt1, pt2)
        ind = 0
        crossing = -1
        if departurethreshold == 1:
            while ind < linearr.size - 1:
                if darktolight:
                    if linearr[ind] <= thresh and linearr[ind + 1] > thresh:
                        crossing = ind
                        break
                if lighttodark:
                    if linearr[ind] >= thresh and linearr[ind + 1] < thresh:
                        crossing = ind
                        break
                ind += 1
            if crossing != -1:
                xind = pt1[0] + int(
                        round((pt2[0] - pt1[0]) * crossing / linearr.size))
                yind = pt1[1] + int(
                        round((pt2[1] - pt1[1]) * crossing / linearr.size))
                ret = (xind, yind)
            else:
                ret = (-1, -1)
                # print 'Edgepoint not found.'
        else:
            while ind < linearr.size - (departurethreshold + 1):
                if darktolight:
                    if linearr[ind] <= thresh and (linearr[
                                                   ind + 1:ind + 1 + departurethreshold] > thresh).all():
                        crossing = ind
                        break
                if lighttodark:
                    if linearr[ind] >= thresh and (linearr[
                                                   ind + 1:ind + 1 + departurethreshold] < thresh).all():
                        crossing = ind
                        break
                ind += 1
            if crossing != -1:
                xind = pt1[0] + int(
                        round((pt2[0] - pt1[0]) * crossing / linearr.size))
                yind = pt1[1] + int(
                        round((pt2[1] - pt1[1]) * crossing / linearr.size))
                ret = (xind, yind)
            else:
                ret = (-1, -1)
                # print 'Edgepoint not found.'
        return ret

    def get_diagonal_scanline_gray(self, pt1, pt2):
        if not self.is_gray():
            self = self.to_gray()
        # self = self._get_gray_narray()
        width = round(math.sqrt(
                math.pow(pt2[0] - pt1[0], 2) + math.pow(pt2[1] - pt1[1], 2)))
        ret = np.zeros(width)

        for x in range(0, ret.size):
            xind = pt1[0] + int(round((pt2[0] - pt1[0]) * x / ret.size))
            yind = pt1[1] + int(round((pt2[1] - pt1[1]) * x / ret.size))
            current_pixel = self.get_pixel(xind, yind)
            ret[x] = current_pixel[0]
        return ret

    def fit_lines(self, guesses, window=10, threshold=128):
        ret = FeatureSet()
        i = 0
        for g in guesses:
            # Guess the size of the crop region from the line guess and the window.
            ymin = np.min([g[0][1], g[1][1]])
            ymax = np.max([g[0][1], g[1][1]])
            xmin = np.min([g[0][0], g[1][0]])
            xmax = np.max([g[0][0], g[1][0]])

            xminW = np.clip(xmin - window, 0, self.width)
            xmaxW = np.clip(xmax + window, 0, self.width)
            yminW = np.clip(ymin - window, 0, self.height)
            ymaxW = np.clip(ymax + window, 0, self.height)
            temp = self.crop(xminW, yminW, xmaxW - xminW, ymaxW - yminW)
            temp = temp.gray_narray

            # pick the lines above our threshold
            x, y = np.where(temp > threshold)
            pts = zip(x, y)
            gpv = np.array([float(g[0][0] - xminW), float(g[0][1] - yminW)])
            gpw = np.array([float(g[1][0] - xminW), float(g[1][1] - yminW)])

            def line_seg2pt(p):
                w = gpw
                v = gpv
                # print w,v
                p = np.array([float(p[0]), float(p[1])])
                l2 = np.sum((w - v) ** 2)
                t = float(np.dot((p - v), (w - v))) / float(l2)
                if t < 0.00:
                    return np.sqrt(np.sum((p - v) ** 2))
                elif t > 1.0:
                    return np.sqrt(np.sum((p - w) ** 2))
                else:
                    project = v + (t * (w - v))
                    return np.sqrt(np.sum((p - project) ** 2))

            # http://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment

            distances = np.array(map(line_seg2pt, pts))
            closepoints = np.where(distances < window)[0]

            pts = np.array(pts)

            if (len(closepoints) < 3):
                continue

            good_pts = pts[closepoints]
            good_pts = good_pts.astype(float)

            x = good_pts[:, 0]
            y = good_pts[:, 1]
            # do the shift from our crop
            # generate the line values
            x = x + xminW
            y = y + yminW

            ymin = np.min(y)
            ymax = np.max(y)
            xmax = np.max(x)
            xmin = np.min(x)

            if (xmax - xmin) > (ymax - ymin):
                # do the least squares
                A = np.vstack([x, np.ones(len(x))]).T
                m, c = nla.lstsq(A, y)[0]
                y0 = int(m * xmin + c)
                y1 = int(m * xmax + c)
                ret.append(Line(self, ((xmin, y0), (xmax, y1))))
            else:
                # do the least squares
                A = np.vstack([y, np.ones(len(y))]).T
                m, c = nla.lstsq(A, x)[0]
                x0 = int(ymin * m + c)
                x1 = int(ymax * m + c)
                ret.append(Line(self, ((x0, ymin), (x1, ymax))))

        return ret

    def fit_line_points(self, guesses, window=(11, 11), samples=20,
                        params=(0.1, 0.1, 0.1)):
        pts = []
        for g in guesses:
            # generate the approximation
            bestGuess = []
            dx = float(g[1][0] - g[0][0])
            dy = float(g[1][1] - g[0][1])
            l = np.sqrt((dx * dx) + (dy * dy))
            if (l <= 0):
                logger.warning(
                        "Can't Do snakeFitPoints without OpenCV >= 2.3.0")
                return

            dx = dx / l
            dy = dy / l
            for i in range(-1, samples + 1):
                t = i * (l / samples)
                bestGuess.append(
                        (int(g[0][0] + (t * dx)), int(g[0][1] + (t * dy))))
            # do the snake fitting
            appx = self.fit_contour(bestGuess, window=window, params=params,
                                    doAppx=False)
            pts.append(appx)

        return pts

    def draw_points(self, pts, color=Color.RED, sz=3, width=-1):
        for p in pts:
            self.draw_circle(p, sz, color, width)
        return None

    def sobel(self, xorder=1, yorder=1, doGray=True, aperture=5,
              aperature=None):
        aperture = aperature if aperature else aperture
        ret = None
        try:
            import cv2
        except:
            logger.warning("Can't do Sobel without OpenCV >= 2.3.0")
            return None

        if (
                                aperture != 1 and aperture != 3 and aperture != 5 and aperture != 7):
            logger.warning("Bad Sobel Aperture, values are [1,3,5,7].")
            return None

        if doGray:
            dst = cv2.Sobel(self.gray_narray, cv2.cv2.CV_32F, xorder, yorder,
                            ksize=aperture)
            minv = np.min(dst)
            maxv = np.max(dst)
            cscale = 255 / (maxv - minv)
            shift = -1 * minv

            t = np.zeros(self.size(), dtype='uint8')
            t = cv2.convertScaleAbs(dst, t, cscale, shift / 255.0)
            ret = Image(t)

        else:
            layers = self.split_channels(grayscale=False)
            sobel_layers = []
            for layer in layers:
                dst = cv2.Sobel(layer.gray_narray, cv2.cv2.CV_32F, xorder,
                                yorder, ksize=aperture)

                minv = np.min(dst)
                maxv = np.max(dst)
                cscale = 255 / (maxv - minv)
                shift = -1 * (minv)

                t = np.zeros(self.size(), dtype='uint8')
                t = cv2.convertScaleAbs(dst, t, cscale, shift / 255.0)
                sobel_layers.append(Image(t))
            b, g, r = sobel_layers

            ret = self.merge_channels(b, g, r)
        return ret

    def track(self, method="CAMShift", ts=None, img=None, bb=None, **kwargs):
        if not ts and not img:
            print("Invalid Input. Must provide FeatureSet or Image")
            return None

        if not ts and not bb:
            print("Invalid Input. Must provide Bounding Box with Image")
            return None

        if not ts:
            ts = TrackSet()
        else:
            img = ts[-1].image
            bb = ts[-1].bb
        try:
            import cv2
        except ImportError:
            print("Tracking is available for OpenCV >= 2.3")
            return None

        if isinstance(img, list):
            ts = self.track(method, ts, img[0], bb, **kwargs)
            for i in img:
                ts = i.track(method, ts, **kwargs)
            return ts

        # Issue #256 - (Bug) Memory management issue due to too many number of images.
        nframes = 300
        if 'nframes' in kwargs:
            nframes = kwargs['nframes']

        if len(ts) > nframes:
            ts.trimList(50)

        if method.lower() == "camshift":
            track = camshift_tracker(self, bb, ts, **kwargs)
            ts.append(track)

        elif method.lower() == "lk":
            track = lk_tracker(self, bb, ts, img, **kwargs)
            ts.append(track)

        elif method.lower() == "surf":
            try:
                from scipy.spatial import distance as Dis
                from sklearn.cluster import DBSCAN
            except ImportError:
                logger.warning("sklearn required")
                return None
            if not hasattr(cv2, "FeatureDetector_create"):
                warnings.warn("OpenCV >= 2.4.3 required. Returning None.")
                return None
            track = surf_tracker(self, bb, ts, **kwargs)
            ts.append(track)

        elif method.lower() == "mftrack":
            track = mf_tracker(self, bb, ts, img, **kwargs)
            ts.append(track)

        return ts

    def _to32f(self):
        ret = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_32F, 3)
        cv2.Convert(self.bitmap, ret)
        return ret

    def __getstate__(self):
        return dict(size=self.size(), colorspace=self._color_space,
                    image=self.apply_layers().bitmap.tostring())

    def __setstate__(self, state):
        self._bitmap = cv2.CreateImageHeader(state['size'], cv2.IPL_DEPTH_8U, 3)
        cv2.SetData(self._bitmap, state['image'])
        self._color_space = state['colorspace']
        self.width = state['size'][0]
        self.height = state['size'][1]

    def area(self):
        return self.width * self.height

    def _get_header_anim(self):
        bb = "GIF89a"
        bb += int2bin(self.size()[0])
        bb += int2bin(self.size()[1])
        bb += "\x87\x00\x00"
        return bb

    def rotate270(self):
        ret = cv2.CreateImage((self.height, self.width), cv2.IPL_DEPTH_8U, 3)
        cv2.Transpose(self.bitmap, ret)
        cv2.Flip(ret, ret, 1)
        return Image(ret, color_space=self._color_space)

    def rotate90(self):
        ret = cv2.CreateImage((self.height, self.width), cv2.IPL_DEPTH_8U, 3)
        cv2.Transpose(self.bitmap, ret)
        cv2.Flip(ret, ret, 0)  # vertical
        return Image(ret, color_space=self._color_space)

    def rotate180(self):
        ret = cv2.CreateImage((self.width, self.height), cv2.IPL_DEPTH_8U, 3)
        cv2.Flip(self.bitmap, ret, 0)  # vertical
        cv2.Flip(ret, ret, 1)  # horizontal
        return Image(ret, color_space=self._color_space)

    def rotate_left(self):
        return self.rotate90()

    def rotate_right(self):
        return self.rotate270()

    def vertical_histogram(self, bins=10, threshold=128, normalize=False,
                           forPlot=False):
        if bins <= 0:
            raise Exception("Not enough bins")

        img = self.gray_narray
        pts = np.where(img > threshold)
        y = pts[1]
        hist = np.histogram(y, bins=bins, range=(0, self.height),
                            normed=normalize)
        ret = None
        if forPlot:
            # for using matplotlib bar command
            # bin labels, bin values, bin width
            ret = (hist[1][0:-1], hist[0], self.height / bins)
        else:
            ret = hist[0]
        return ret

    def horizontal_histogram(self, bins=10, threshold=128, normalize=False,
                             forPlot=False):
        if bins <= 0:
            raise Exception("Not enough bins")

        img = self.gray_narray
        pts = np.where(img > threshold)
        x = pts[0]
        hist = np.histogram(x, bins=bins, range=(0, self.width),
                            normed=normalize)
        ret = None
        if forPlot:
            # for using matplotlib bar command
            # bin labels, bin values, bin width
            ret = (hist[1][0:-1], hist[0], self.width / bins)
        else:
            ret = hist[0]
        return ret

    def get_linescan(self, x=None, y=None, pt1=None, pt2=None, channel=-1):
        if channel == -1:
            img = self.gray_narray
        else:
            try:
                img = self.narray[:, :, channel]
            except IndexError:
                print('Channel missing!')
                return None

        ret = None
        if x is not None and y is None and pt1 is None and pt2 is None:
            if 0 <= x < self.width:
                ret = LineScan(img[x, :])
                ret.image = self.x
                ret.pt1 = (x, 0)
                ret.pt2 = (x, self.height)
                ret.col = x
                x = np.ones((1, self.height))[0] * x
                y = range(0, self.height, 1)
                pts = zip(x, y)
                ret.pointLoc = pts
            else:
                warnings.warn(
                        "Image.get_linescan - that is not valid scanline.")
                return None

        elif x is None and y is not None and pt1 is None and pt2 is None:
            if 0 <= y < self.height:
                ret = LineScan(img[:, y])
                ret.image = self
                ret.pt1 = (0, y)
                ret.pt2 = (self.width, y)
                ret.row = y
                y = np.ones((1, self.width))[0] * y
                x = range(0, self.width, 1)
                pts = zip(x, y)
                ret.pointLoc = pts

            else:
                warnings.warn(
                        "Image.get_linescan - that is not valid scanline.")
                return None

            pass
        elif ((isinstance(pt1, tuple) or isinstance(pt1, list)) and
              (isinstance(pt2, tuple) or isinstance(pt2, list)) and
              len(pt1) == 2 and len(pt2) == 2 and x is None and y is None):

            pts = self.bresenham_line(pt1, pt2)
            ret = LineScan([img[p[0], p[1]] for p in pts])
            ret.pointLoc = pts
            ret.image = self
            ret.pt1 = pt1
            ret.pt2 = pt2

        else:
            # an invalid combination - warn
            warnings.warn(
                    "Image.get_linescan - that is not valid scanline.")
            return None
        ret.channel = channel
        return ret

    def set_linescan(self, linescan, x=None, y=None, pt1=None, pt2=None,
                    channel=-1):
        if channel == -1:
            img = np.copy(self.gray_narray)
        else:
            try:
                img = np.copy(self.narray[:, :, channel])
            except IndexError:
                print
                'Channel missing!'
                return None

        if (x is None and y is None and pt1 is None and pt2 is None):
            if (linescan.pt1 is None or linescan.pt2 is None):
                warnings.warn(
                        "Image.set_linescan: No coordinates to re-insert linescan.")
                return None
            else:
                pt1 = linescan.pt1
                pt2 = linescan.pt2
                if (pt1[0] == pt2[0] and np.abs(
                            pt1[1] - pt2[1]) == self.height):
                    x = pt1[0]  # vertical line
                    pt1 = None
                    pt2 = None

                elif (pt1[1] == pt2[1] and np.abs(
                            pt1[0] - pt2[0]) == self.width):
                    y = pt1[1]  # horizontal line
                    pt1 = None
                    pt2 = None

        ret = None
        if x is not None and y is None and pt1 is None and pt2 is None:
            if 0 <= x < self.width:
                if len(linescan) != self.height:
                    linescan = linescan.resample(self.height)
                # check for number of points
                # linescan = np.array(linescan)
                img[x, :] = np.clip(linescan[:], 0, 255)
            else:
                warnings.warn(
                        "Image.set_linescan: No coordinates to re-insert linescan.")
                return None
        elif x is None and y is not None and pt1 is None and pt2 is None:
            if 0 <= y < self.height:
                if len(linescan) != self.width:
                    linescan = linescan.resample(self.width)
                # check for number of points
                # linescan = np.array(linescan)
                img[:, y] = np.clip(linescan[:], 0, 255)
            else:
                warnings.warn(
                        "Image.set_linescan: No coordinates to re-insert linescan.")
                return None
        elif ((isinstance(pt1, tuple) or isinstance(pt1, list)) and
                  (isinstance(pt2, tuple) or isinstance(pt2, list)) and
                      len(pt1) == 2 and len(pt2) == 2 and
                      x is None and y is None):

            pts = self.bresenham_line(pt1, pt2)
            if len(linescan) != len(pts):
                linescan = linescan.resample(len(pts))
            # linescan = np.array(linescan)
            linescan = np.clip(linescan[:], 0, 255)
            idx = 0
            for pt in pts:
                img[pt[0], pt[1]] = linescan[idx]
                idx += 1
        else:
            warnings.warn(
                    "Image.set_linescan: No coordinates to re-insert linescan.")
            return None
        if channel == -1:
            ret = Image(img)
        else:
            temp = np.copy(self.narray)
            temp[:, :, channel] = img
            ret = Image(temp)
        return ret

    def replaceLineScan(self, linescan, x=None, y=None, pt1=None, pt2=None,
                        channel=None):
        if x is None and y is None and pt1 is None and pt2 is None and channel is None:

            if linescan.channel == -1:
                img = np.copy(self.gray_narray)
            else:
                try:
                    img = np.copy(self.narray[:, :, linescan.channel])
                except IndexError:
                    print('Channel missing!')
                    return None

            if linescan.row is not None:
                if len(linescan) == self.width:
                    ls = np.clip(linescan, 0, 255)
                    img[:, linescan.row] = ls[:]
                else:
                    warnings.warn("LineScan Size and Image size do not match")
                    return None

            elif linescan.col is not None:
                if len(linescan) == self.height:
                    ls = np.clip(linescan, 0, 255)
                    img[linescan.col, :] = ls[:]
                else:
                    warnings.warn("LineScan Size and Image size do not match")
                    return None
            elif linescan.pt1 and linescan.pt2:
                pts = self.bresenham_line(linescan.pt1, linescan.pt2)
                if (len(linescan) != len(pts)):
                    linescan = linescan.resample(len(pts))
                ls = np.clip(linescan[:], 0, 255)
                idx = 0
                for pt in pts:
                    img[pt[0], pt[1]] = ls[idx]
                    idx = idx + 1

            if linescan.channel == -1:
                ret = Image(img)
            else:
                temp = np.copy(self.narray)
                temp[:, :, linescan.channel] = img
                ret = Image(temp)

        else:
            if channel is None:
                ret = self.set_linescan(linescan, x, y, pt1, pt2,
                                       linescan.channel)
            else:
                ret = self.set_linescan(linescan, x, y, pt1, pt2, channel)
        return ret

    def get_pixels_online(self, pt1, pt2):
        ret = None
        if ((isinstance(pt1, tuple) or isinstance(pt1, list)) and
                (isinstance(pt2, tuple) or isinstance(pt2, list)) and
                    len(pt1) == 2 and len(pt2) == 2):
            pts = self.bresenham_line(pt1, pt2)
            ret = [self.get_pixel(p[0], p[1]) for p in pts]
        else:
            warnings.warn("Image.get_pixels_online - The line you provided is "
                          "not valid")

        return ret

    def bresenham_line(self, (x, y), (x2, y2)):
        if (not 0 <= x <= self.width - 1 or not 0 <= y <= self.height - 1 or
                not 0 <= x2 <= self.width - 1 or not 0 <= y2 <= self.height - 1):
            l = Line(self, ((x, y), (x2, y2))).crop2image_edges()
            if l:
                ep = list(l.end_points)
                ep.sort()
                x, y = ep[0]
                x2, y2 = ep[1]
            else:
                return []

        steep = 0
        coords = []
        dx = abs(x2 - x)
        if (x2 - x) > 0:
            sx = 1
        else:
            sx = -1
        dy = abs(y2 - y)
        if (y2 - y) > 0:
            sy = 1
        else:
            sy = -1
        if dy > dx:
            steep = 1
            x, y = y, x
            dx, dy = dy, dx
            sx, sy = sy, sx
        d = (2 * dy) - dx
        for i in range(0, dx):
            if steep:
                coords.append((y, x))
            else:
                coords.append((x, y))
            while d >= 0:
                y += sy
                d -= 2 * dx
            x += sx
            d += 2 * dy
        coords.append((x2, y2))
        return coords

    def uncrop(self, ListofPts):
        return [(i[0] + self._uncropped_x, i[1] + self._uncropped_y) for i in
                ListofPts]

    def grid(self, dimensions=(10, 10), color=(0, 0, 0), width=1,
             antialias=True, alpha=-1):
        ret = self.copy()
        try:
            step_row = self.size()[1] / dimensions[0]
            step_col = self.size()[0] / dimensions[1]
        except ZeroDivisionError:
            return imgTemp

        i = 1
        j = 1

        grid = DrawingLayer(self.size())  # add a new layer for grid
        while i < dimensions[0] and j < dimensions[1]:
            if dimensions[0] > i:
                grid.line((0, step_row * i), (self.size()[0], step_row * i),
                          color, width, antialias, alpha)
                i += 1
            if j < dimensions[1]:
                grid.line((step_col * j, 0), (step_col * j, self.size()[1]),
                          color, width, antialias, alpha)
                j += 1
        ret._grid_layer[0] = ret.add_drawing_layer(grid)  # store grid layer index
        ret._grid_layer[1] = dimensions
        return ret

    def remove_grid(self):
        if self._grid_layer[0] is not None:
            grid = self.remove_drawing_layer(self._grid_layer[0])
            self._grid_layer = [None, [0, 0]]
            return grid
        else:
            return None

    def find_grid_lines(self):
        gridIndex = self.get_drawing_layer(self._grid_layer[0])
        if self._grid_layer[0] == -1:
            print("Cannot find grid on the image, Try adding a grid first")

        lineFS = FeatureSet()
        try:
            step_row = self.size()[1] / self._grid_layer[1][0]
            step_col = self.size()[0] / self._grid_layer[1][1]
        except ZeroDivisionError:
            return None

        i = 1
        j = 1

        while i < self._grid_layer[1][0]:
            lineFS.append(
                    Line(self,
                         ((0, step_row * i), (self.size()[0], step_row * i))))
            i += 1
        while j < self._grid_layer[1][1]:
            lineFS.append(
                    Line(self,
                         ((step_col * j, 0), (step_col * j, self.size()[1]))))
            j += 1

        return lineFS

    def logicalAND(self, img, grayscale=True):
        if not self.size() == img.size():
            print("Both images must have same sizes")
            return None
        try:
            import cv2
        except ImportError:
            print("This function is available for OpenCV >= 2.3")
        if grayscale:
            retval = cv2.bitwise_and(self.gray_narray,
                                     img.gray_narray)
        else:
            retval = cv2.bitwise_and(self.cvnarray, img.cvnarray)
        return Image(retval, cv2image=True)

    def logicalNAND(self, img, grayscale=True):
        if not self.size() == img.size():
            print("Both images must have same sizes")
            return None
        try:
            import cv2
        except ImportError:
            print("This function is available for OpenCV >= 2.3")
        if grayscale:
            retval = cv2.bitwise_and(self.gray_narray,
                                     img.gray_narray)
        else:
            retval = cv2.bitwise_and(self.cvnarray, img.cvnarray)
        retval = cv2.bitwise_not(retval)
        return Image(retval, cv2image=True)

    def logicalOR(self, img, grayscale=True):
        if not self.size() == img.size():
            print("Both images must have same sizes")
            return None
        try:
            import cv2
        except ImportError:
            print("This function is available for OpenCV >= 2.3")
        if grayscale:
            retval = cv2.bitwise_or(self.gray_narray,
                                    img.gray_narray)
        else:
            retval = cv2.bitwise_or(self.cvnarray, img.cvnarray)
        return Image(retval, cv2image=True)

    def logicalXOR(self, img, grayscale=True):
        if not self.size() == img.size():
            print("Both images must have same sizes")
            return None
        try:
            import cv2
        except ImportError:
            print("This function is available for OpenCV >= 2.3")
        if grayscale:
            retval = cv2.bitwise_xor(self.gray_narray,
                                     img.gray_narray)
        else:
            retval = cv2.bitwise_xor(self.cvnarray, img.cvnarray)
        return Image(retval, cv2image=True)

    def matchSIFTKeyPoints(self, template, quality=200):
        try:
            import cv2
        except ImportError:
            warnings.warn("OpenCV >= 2.4.3 required")
            return None
        if not hasattr(cv2, "FeatureDetector_create"):
            warnings.warn("OpenCV >= 2.4.3 required")
            return None
        if template is None:
            return None
        detector = cv2.FeatureDetector_create("SIFT")
        descriptor = cv2.DescriptorExtractor_create("SIFT")
        img = self.cvnarray
        template_img = template.cvnarray

        skp = detector.detect(img)
        skp, sd = descriptor.compute(img, skp)

        tkp = detector.detect(template_img)
        tkp, td = descriptor.compute(template_img, tkp)

        idx, dist = self._get_FLANN_matches(sd, td)
        dist = dist[:, 0] / 2500.0
        dist = dist.reshape(-1, ).tolist()
        idx = idx.reshape(-1).tolist()
        indices = range(len(dist))
        indices.sort(key=lambda i: dist[i])
        dist = [dist[i] for i in indices]
        idx = [idx[i] for i in indices]
        sfs = []
        for i, dis in itertools.izip(idx, dist):
            if dis < quality:
                sfs.append(KeyPoint(template, skp[i], sd, "SIFT"))
            else:
                break  # since sorted

        idx, dist = self._get_FLANN_matches(td, sd)
        dist = dist[:, 0] / 2500.0
        dist = dist.reshape(-1, ).tolist()
        idx = idx.reshape(-1).tolist()
        indices = range(len(dist))
        indices.sort(key=lambda i: dist[i])
        dist = [dist[i] for i in indices]
        idx = [idx[i] for i in indices]
        tfs = []
        for i, dis in itertools.izip(idx, dist):
            if dis < quality:
                tfs.append(KeyPoint(template, tkp[i], td, "SIFT"))
            else:
                break

        return sfs, tfs

    def drawSIFTKeyPointMatch(self, template, distance=200, num=-1, width=1):
        if template is None:
            return
        resultImg = template. side_by_side(self, scale=False)
        hdif = (self.height - template.height) / 2
        sfs, tfs = self.matchSIFTKeyPoints(template, distance)
        maxlen = min(len(sfs), len(tfs))
        if num < 0 or num > maxlen:
            num = maxlen
        for i in range(num):
            skp = sfs[i]
            tkp = tfs[i]
            pt_a = (int(tkp.y), int(tkp.x) + hdif)
            pt_b = (int(skp.y) + template.width, int(skp.x))
            resultImg.drawLine(pt_a, pt_b, color=Color.random(),
                               thickness=width)
        return resultImg

    def stega_encode(self, message):
        try:
            import stepic
        except ImportError:
            logger.warning("stepic library required")
            return None
        warnings.simplefilter("ignore")
        pilImg = PILImage.frombuffer("RGB", self.size(), self.toString())
        stepic.encode_inplace(pilImg, message)
        ret = Image(pilImg)
        return ret.flip_vertical()

    def stega_decode(self):
        try:
            import stepic
        except ImportError:
            logger.warning("stepic library required")
            return None
        warnings.simplefilter("ignore")
        pilImg = PILImage.frombuffer("RGB", self.size(), self.toString())
        result = stepic.decode(pilImg)
        return result

    def find_features(self, method="szeliski", threshold=1000):
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV >= 2.3.0 required")
            return None
        img = self.gray_narray
        blur = cv2.GaussianBlur(img, (3, 3), 0)

        Ix = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
        Iy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)

        Ix_Ix = np.multiply(Ix, Ix)
        Iy_Iy = np.multiply(Iy, Iy)
        Ix_Iy = np.multiply(Ix, Iy)

        Ix_Ix_blur = cv2.GaussianBlur(Ix_Ix, (5, 5), 0)
        Iy_Iy_blur = cv2.GaussianBlur(Iy_Iy, (5, 5), 0)
        Ix_Iy_blur = cv2.GaussianBlur(Ix_Iy, (5, 5), 0)

        harris_thresh = threshold * 5000
        alpha = 0.06
        detA = Ix_Ix_blur * Iy_Iy_blur - Ix_Iy_blur ** 2
        traceA = Ix_Ix_blur + Iy_Iy_blur
        feature_list = []
        if method == "szeliski":
            harmonic_mean = detA / traceA
            for j, i in np.argwhere(harmonic_mean > threshold):
                feature_list.append(
                        Feature(self, i, j, ((i, j), (i, j), (i, j), (i, j))))

        elif method == "harris":
            harris_function = detA - (alpha * traceA * traceA)
            for j, i in np.argwhere(harris_function > harris_thresh):
                feature_list.append(
                        Feature(self, i, j, ((i, j), (i, j), (i, j), (i, j))))
        else:
            logger.warning("Invalid method.")
            return None
        return feature_list

    def watershed(self, mask=None, erode=2, dilate=2, useMyMask=False):
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV >= 2.3.0 required")
            return None
        output = self.zeros(3)
        if mask is None:
            mask = self.binarize().invert()
        newmask = None
        if (not useMyMask):
            newmask = Image((self.width, self.height))
            newmask = newmask.floodFill((0, 0), color=Color.WATERSHED_BG)
            newmask = (newmask - mask.dilate(dilate) + mask.erode(erode))
        else:
            newmask = mask
        m = np.int32(newmask.gray_narray)
        cv2.watershed(self.cvnarray, m)
        m = cv2.convertScaleAbs(m)
        ret, thresh = cv2.threshold(m, 0, 255, cv2.cv2.CV_THRESH_OTSU)
        ret = Image(thresh, cv2image=True)
        return ret

    def findBlobsFromWatershed(self, mask=None, erode=2, dilate=2,
                               useMyMask=False, invert=False, minsize=20,
                               maxsize=None):
        newmask = self.watershed(mask, erode, dilate, useMyMask)
        if (invert):
            newmask = mask.invert()
        return self.find_blobs_from_mask(newmask, minsize=minsize, maxsize=maxsize)

    def maxValue(self, locations=False):
        if (locations):
            val = np.max(self.gray_narray)
            x, y = np.where(self.gray_narray == val)
            locs = zip(x.tolist(), y.tolist())
            return int(val), locs
        else:
            val = np.max(self.gray_narray)
            return int(val)

    def minValue(self, locations=False):
        if (locations):
            val = np.min(self.gray_narray)
            x, y = np.where(self.gray_narray == val)
            locs = zip(x.tolist(), y.tolist())
            return int(val), locs
        else:
            val = np.min(self.gray_narray)
            return int(val)

    def findKeypointClusters(self, num_of_clusters=5, order='dsc',
                             flavor='surf'):
        if flavor.lower() == 'corner':
            keypoints = self.find_corners()  # fallback to corners
        else:
            keypoints = self.find_keypoints(
                    flavor=flavor.upper())  # find the keypoints
        if keypoints is None or keypoints <= 0:
            return None

        xypoints = np.array([(f.x, f.y) for f in keypoints])
        xycentroids, xylabels = scv2.kmeans2(xypoints,
                                            num_of_clusters)  # find the clusters of keypoints
        xycounts = np.array([])

        for i in range(
                num_of_clusters):  # count the frequency of occurences for sorting
            xycounts = np.append(xycounts, len(np.where(xylabels == i)[-1]))

        merged = np.msort(np.hstack(
                (np.vstack(xycounts), xycentroids)))  # sort based on occurence
        clusters = [c[1:] for c in
                    merged]  # strip out just the values ascending
        if order.lower() == 'dsc':
            clusters = clusters[::-1]  # reverse if descending

        fs = FeatureSet()
        for x, y in clusters:  # map the values to a feature set
            f = Corner(self, x, y)
            fs.append(f)

        return fs

    def getFREAKDescriptor(self, flavor="SURF"):
        try:
            import cv2
        except ImportError:
            warnings.warn("OpenCV version >= 2.4.2 requierd")
            return None

        if cv2.__version__.startswith('$Rev:'):
            warnings.warn("OpenCV version >= 2.4.2 requierd")
            return None

        if int(cv2.__version__.replace('.', '0')) < 20402:
            warnings.warn("OpenCV version >= 2.4.2 requierd")
            return None

        flavors = ["SIFT", "SURF", "BRISK", "ORB", "STAR", "MSER", "FAST",
                   "Dense"]
        if flavor not in flavors:
            warnings.warn("Unkown Keypoints detector. Returning None.")
            return None
        detector = cv2.FeatureDetector_create(flavor)
        extractor = cv2.DescriptorExtractor_create("FREAK")
        self._keypoints = detector.detect(self.gray_narray)
        self._keypoints, self._kp_descriptors = extractor.compute(
                self.gray_narray,
                self._keypoints)
        fs = FeatureSet()
        for i in range(len(self._keypoints)):
            fs.append(
                    KeyPoint(self, self._keypoints[i], self._kp_descriptors[i],
                             flavor))

        return fs, self._kp_descriptors

    def get_gray_histogram_counts(self, bins=255, limit=-1):
        hist = self.histogram(bins)
        vals = [(e, h) for h, e in enumerate(hist)]
        vals.sort()
        vals.reverse()

        if limit == -1:
            limit = bins

        return vals[:limit]

    def gray_peaks(self, bins=255, delta=0, lookahead=15):
        y_axis, x_axis = np.histogram(self.gray_narray, bins=range(bins + 2))
        x_axis = x_axis[0:bins + 1]
        maxtab = []
        mintab = []
        length = len(y_axis)
        if x_axis is None:
            x_axis = range(length)

        # perform some checks
        if length != len(x_axis):
            raise ValueError("Input vectors y_axis and x_axis must have same length")
        if lookahead < 1:
            raise ValueError("Lookahead must be above '1' in value")
        if not (np.isscalar(delta) and delta >= 0):
            raise ValueError("delta must be a positive number")

        # needs to be a numpy array
        y_axis = np.asarray(y_axis)

        # maxima and minima candidates are temporarily stored in
        # mx and mn respectively
        mn, mx = np.Inf, -np.Inf

        # Only detect peak if there is 'lookahead' amount of points after it
        for index, (x, y) in enumerate(
                zip(x_axis[:-lookahead], y_axis[:-lookahead])):
            if y > mx:
                mx = y
                mxpos = x
            if y < mn:
                mn = y
                mnpos = x

            ####look for max####
            if y < mx - delta and mx != np.Inf:
                # Maxima peak candidate found
                # look ahead in signal to ensure that this is a peak and not jitter
                if y_axis[index:index + lookahead].max() < mx:
                    maxtab.append((mxpos, mx))
                    # set algorithm to only find minima now
                    mx = np.Inf
                    mn = np.Inf

            if y > mn + delta and mn != -np.Inf:
                # Minima peak candidate found
                # look ahead in signal to ensure that this is a peak and not jitter
                if y_axis[index:index + lookahead].min() > mn:
                    mintab.append((mnpos, mn))
                    # set algorithm to only find maxima now
                    mn = -np.Inf
                    mx = -np.Inf

        ret = []
        for intensity, pixelcount in maxtab:
            ret.append(
                    (intensity, pixelcount / float(self.width * self.height)))
        return ret

    def tvDenoising(self, gray=False, weight=50, eps=0.0002, max_iter=200,
                    resize=1):
        try:
            from skimage.filter import denoise_tv_chambolle
        except ImportError:
            logger.warn('Scikit-image Library not installed!')
            return None

        img = self.copy()

        if resize <= 0:
            print
            'Enter a valid resize value'
            return None

        if resize != 1:
            img = img.resize(int(img.width * resize), int(img.height * resize))

        if gray is True:
            img = img.gray_narray
            multichannel = False
        elif gray is False:
            img = img.narray
            multichannel = True
        else:
            warnings.warn('gray value not valid')
            return None

        denoise_mat = denoise_tv_chambolle(img, weight, eps, max_iter,
                                           multichannel)
        ret = img * denoise_mat

        ret = Image(ret)
        if resize != 1:
            return ret.resize(int(ret.width / resize),
                              int(ret.width / resize))
        else:
            return ret

    @multipledispatch.dispatch
    def motionBlur(self, intensity=15, direction='NW'):
        mid = int(intensity / 2)
        tmp = np.identity(intensity)

        if intensity == 0:
            warnings.warn("0 intensity means no blurring")
            return self

        elif intensity % 2 is 0:
            div = mid
            for i in range(mid, intensity - 1):
                tmp[i][i] = 0
        else:
            div = mid + 1
            for i in range(mid + 1, intensity - 1):
                tmp[i][i] = 0

        if direction == 'right' or direction.upper() == 'E':
            kernel = np.concatenate((np.zeros((1, mid)), np.ones((1, mid + 1))),
                                    axis=1)
        elif direction == 'left' or direction.upper() == 'W':
            kernel = np.concatenate((np.ones((1, mid + 1)), np.zeros((1, mid))),
                                    axis=1)
        elif direction == 'up' or direction.upper() == 'N':
            kernel = np.concatenate((np.ones((1 + mid, 1)), np.zeros((mid, 1))),
                                    axis=0)
        elif direction == 'down' or direction.upper() == 'S':
            kernel = np.concatenate((np.zeros((mid, 1)), np.ones((mid + 1, 1))),
                                    axis=0)
        elif direction.upper() == 'NW':
            kernel = tmp
        elif direction.upper() == 'NE':
            kernel = np.fliplr(tmp)
        elif direction.upper() == 'SW':
            kernel = np.flipud(tmp)
        elif direction.upper() == 'SE':
            kernel = np.flipud(np.fliplr(tmp))
        else:
            warnings.warn("Please enter a proper direction")
            return None

        retval = self.convolve(kernel=kernel / div)
        return retval

    def recognizeFace(self, recognizer=None):
        try:
            import cv2
            if not hasattr(cv2, "createFisherFaceRecognizer"):
                warnings.warn("OpenCV >= 2.4.4 required to use this.")
                return None
        except ImportError:
            warnings.warn("OpenCV >= 2.4.4 required to use this.")
            return None

        if not isinstance(recognizer, FaceRecognizer):
            warnings.warn("PhloxAR.Features.FaceRecognizer object required.")
            return None

        w, h = recognizer.imageSize
        label = recognizer.predict(self.resize(w, h))
        return label

    def findAndRecognizeFaces(self, recognizer, cascade=None):
        try:
            import cv2
            if not hasattr(cv2, "createFisherFaceRecognizer"):
                warnings.warn("OpenCV >= 2.4.4 required to use this.")
                return None
        except ImportError:
            warnings.warn("OpenCV >= 2.4.4 required to use this.")
            return None

        if not isinstance(recognizer, FaceRecognizer):
            warnings.warn("PhloxAR.Features.FaceRecognizer object required.")
            return None

        if not cascade:
            cascade = "/".join([LAUNCH_PATH, "/Features/HaarCascades/face.xml"])

        faces = self.find_haar_features(cascade)
        if not faces:
            warnings.warn("Faces not found in the image.")
            return None

        ret = []
        for face in faces:
            label, confidence = face.crop().recognizeFace(recognizer)
            ret.append([face, label, confidence])
        return ret

    def channelMixer(self, channel='r', weight=(100, 100, 100)):
        r, g, b = self.split_channels()
        if weight[0] > 200 or weight[1] > 200 or weight[2] >= 200:
            if weight[0] < -200 or weight[1] < -200 or weight[2] < -200:
                warnings.warn('Value of weights can be from -200 to 200%')
                return None

        weight = map(float, weight)
        channel = channel.lower()
        if channel == 'r':
            r = r * (weight[0] / 100.0) + g * (weight[1] / 100.0) + b * (
                weight[2] / 100.0)
        elif channel == 'g':
            g = r * (weight[0] / 100.0) + g * (weight[1] / 100.0) + b * (
                weight[2] / 100.0)
        elif channel == 'b':
            b = r * (weight[0] / 100.0) + g * (weight[1] / 100.0) + b * (
                weight[2] / 100.0)
        else:
            warnings.warn('Please enter a valid channel(r/g/b)')
            return None

        ret = self.merge_channels(r=r, g=g, b=b)
        return ret

    def prewitt(self):
        img = self.copy()
        grayimg = img.grayscale()
        gx = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
        gy = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        grayx = grayimg.convolve(gx)
        grayy = grayimg.convolve(gy)
        grayxnp = np.uint64(grayx.gray_narray)
        grayynp = np.uint64(grayy.gray_narray)
        ret = Image(np.sqrt(grayxnp ** 2 + grayynp ** 2))
        return ret

    def edge_snap(self, pointList, step=1):
        img_arr = self.gray_narray
        c1 = np.count_nonzero(img_arr)
        c2 = np.count_nonzero(img_arr - 255)

        # checking that all values are 0 and 255
        if c1 + c2 != img_arr.size:
            raise ValueError("Image must be binary")

        if len(pointList) < 2:
            return None

        finalList = [pointList[0]]
        featureSet = FeatureSet()
        last = pointList[0]
        for point in pointList[1:None]:
            finalList += self._edge_snap2(last, point, step)
            last = point

        last = finalList[0]
        for point in finalList:
            featureSet.append(Line(self, (last, point)))
            last = point
        return featureSet

    def _edge_snap2(self, start, end, step):
        edge_map = np.copy(self.gray_narray)

        # Size of the box around a point which is checked for edges.
        box = step * 4

        xmin = min(start[0], end[0])
        xmax = max(start[0], end[0])
        ymin = min(start[1], end[1])
        ymax = max(start[1], end[1])

        line = self.bresenham_line(start, end)

        # List of Edge Points.
        finalList = []
        i = 0

        # Closest any point has ever come to the end point
        overallMinDist = None

        while i < len(line):

            x, y = line[i]

            # Get the matrix of points fromx around current point.
            region = edge_map[x - box:x + box, y - box:y + box]

            # Condition at the boundary of the image
            if (region.shape[0] == 0 or region.shape[1] == 0):
                i += step
                continue

            # Index of all Edge points
            indexList = np.argwhere(region > 0)
            if indexList.size > 0:

                # Center the coordinates around the point
                indexList -= box
                minDist = None

                # Incase multiple edge points exist, choose the one closest
                # to the end point
                for ix, iy in indexList:
                    dist = math.hypot(x + ix - end[0], iy + y - end[1])
                    if minDist is None or dist < minDist:
                        dx, dy = ix, iy
                        minDist = dist

                # The distance of the new point is compared with the least
                # distance computed till now, the point is rejected if it's
                # comparitively more. This is done so that edge points don't
                # wrap around a _curve instead of heading towards the end point
                if overallMinDist is not None and minDist > overallMinDist * 1.1:
                    i += step
                    continue

                if overallMinDist is None or minDist < overallMinDist:
                    overallMinDist = minDist

                # Reset the points in the box so that they are not detected
                # during the next iteration.
                edge_map[x - box:x + box, y - box:y + box] = 0

                # Keep all the points in the bounding box
                if xmin <= x + dx <= xmax and ymin <= y + dx <= ymax:
                    # Add the point to list and redefine the line
                    line = [(x + dx, y + dy)] + self.bresenham_line(
                            (x + dx, y + dy), end)
                    finalList += [(x + dx, y + dy)]

                    i = 0

            i += step
        finalList += [end]
        return finalList

    @multipledispatch.dispatch
    def motion_blur(self, intensity=15, angle=0):
        intensity = int(intensity)

        if intensity <= 1:
            logger.warning('power less than 1 will result in no change')
            return self

        kernel = np.zeros((intensity, intensity))

        rad = math.radians(angle)
        x1, y1 = intensity / 2, intensity / 2

        x2 = int(x1 - (intensity - 1) / 2 * math.sin(rad))
        y2 = int(y1 - (intensity - 1) / 2 * math.cos(rad))

        line = self.bresenham_line((x1, y1), (x2, y2))

        x = [p[0] for p in line]
        y = [p[1] for p in line]

        kernel[x, y] = 1
        kernel /= len(line)
        return self.convolve(kernel=kernel)

    @property
    def lightness(self):
        if (self._color_space == ColorSpace.BGR or
                self._color_space == ColorSpace.UNKNOWN):
            img_mat = np.array(self.cvnarray, dtype=np.int)
            ret = np.array((np.max(img_mat, 2) + np.min(img_mat, 2)) / 2,
                            dtype=np.uint8)

        else:
            logger.warnings('Input a RGB image')
            return None
        return Image(ret, cv2image=True)

    @property
    def luminosity(self):
        if (self._color_space == ColorSpace.BGR or
                self._color_space == ColorSpace.UNKNOWN):
            img_mat = np.array(self.cvnarray, dtype=np.int)
            ret = np.array(np.average(img_mat, 2, (0.07, 0.71, 0.21)),
                           dtype=np.uint8)

        else:
            logger.warnings('Input a RGB image')
            return None
        return Image(ret, cv2image=True)

    @property
    def average(self):
        if (self._color_space == ColorSpace.BGR or
                self._color_space == ColorSpace.UNKNOWN):
            img_mat = np.array(self.cvnarray, dtype=np.int)
            ret = np.array(img_mat.mean(2), dtype=np.uint8)

        else:
            logger.warnings('Input a RGB image')
            return None
        return Image(ret, cv2image=True)

    def smart_rotate(self, bins=18, point=[-1, -1], auto=True, threshold=80,
                    minLength=30, maxGap=10, t1=150, t2=200, fixed=True):
        lines = self.find_lines(threshold, minLength, maxGap, t1, t2)

        if len(lines) == 0:
            logger.warning("No lines found in the image")
            return self

        # Initialize empty bins
        binn = [[] for i in range(bins)]

        # Convert angle to bin number
        conv = lambda x: int(x + 90) / bins

        # Adding lines to bins
        [binn[conv(line.angle())].append(line) for line in lines]

        # computing histogram, value of each col is total length of all lines
        # in the bin
        hist = [sum([line.length() for line in lines]) for lines in binn]

        # The maximum histogram
        index = np.argmax(np.array(hist))

        # Good ol weighted mean, for the selected bin
        avg = sum([line.angle() * line.length() for line in binn[index]]) / sum(
                [line.length() for line in binn[index]])

        # Mean of centers of all lines in selected bin
        if auto:
            x = sum([line.end_points[0][0] + line.end_points[1][0] for line in
                     binn[index]]) / 2 / len(binn[index])
            y = sum([line.end_points[0][1] + line.end_points[1][1] for line in
                     binn[index]]) / 2 / len(binn[index])
            point = [x, y]

        # Determine whether to rotate the lines to vertical or horizontal
        if -45 <= avg <= 45:
            return self.rotate(avg, fixed=fixed, point=point)
        elif avg > 45:
            return self.rotate(avg - 90, fixed=fixed, point=point)
        else:
            return self.rotate(avg + 90, fixed=fixed, point=point)

    def normalize(self, newMin=0, newMax=255, minCut=2, maxCut=98):
        if newMin < 0 or newMax > 255:
            warnings.warn("newMin and newMax can vary from 0-255")
            return None
        if newMax < newMin:
            warnings.warn("newMin should be less than newMax")
            return None
        if minCut > 100 or maxCut > 100:
            warnings.warn("minCut and maxCut")
            return None
        # avoiding the effect of odd pixels
        try:
            hist = self.get_gray_histogram_counts()
            freq, val = zip(*hist)
            max_freq = (freq[0] - freq[-1]) * maxCut / 100.0
            min_freq = (freq[0] - freq[-1]) * minCut / 100.0
            closestMatch = lambda a, l: min(l, key=lambda x: abs(x - a))
            maxval = closestMatch(max_freq, val)
            minval = closestMatch(min_freq, val)
            ret = (self.grayscale() - minval) * (
                (newMax - newMin) / float(maxval - minval)) + newMin
        # catching zero division in case there are very less intensities present
        # Normalizing based on absolute max and min intensities present
        except ZeroDivisionError:
            maxval = self.maxValue()
            minval = self.minValue()
            ret = (self.grayscale() - minval) * (
                (newMax - newMin) / float(maxval - minval)) + newMin
        # catching the case where there is only one intensity throughout
        except:
            warnings.warn(
                    "All pixels of the image have only one intensity value")
            return None
        return ret

    def get_normalized_hue_histogram(self, roi=None):
        try:
            import cv2
        except ImportError:
            warnings.warn("OpenCV >= 2.3 required to use this.")
            return None

        if roi:  # roi is anything that can be taken to be an roi
            roi = ROI(roi, self)
            hsv = roi.crop().to_hsv().cvnarray
        else:
            hsv = self.to_hsv().cvnarray
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist

    def back_project_hue_histogram(self, model, smooth=True, fullColor=False,
                                   thresh=None):
        try:
            import cv2
        except ImportError:
            warnings.warn("OpenCV >= 2.3 required to use this.")
            return None

        if model is None:
            warnings.warn('Backproject requires a model')
            return None
        # this is the easier test, try to cajole model into ROI
        if isinstance(model, Image):
            model = model.get_normalized_hue_histogram()
        if not isinstance(model, np.ndarray) or model.shape != (180, 256):
            model = self.get_normalized_hue_histogram(model)
        if isinstance(model, np.ndarray) and model.shape == (180, 256):
            hsv = self.to_hsv().cvnarray
            dst = cv2.calcBackProject([hsv], [0, 1], model, [0, 180, 0, 256], 1)
            if smooth:
                disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                cv2.filter2D(dst, -1, disc, dst)
            result = Image(dst, cv2image=True)
            result = result.to_bgr()
            if thresh:
                result = result.threshold(thresh)
            if fullColor:
                temp = Image((self.width, self.height))
                result = temp.blit(self, alphaMask=result)
            return result
        else:
            warnings.warn('Backproject model does not appear to be valid')
            return None

    def find_blobs_from_hue_histogram(self, model, threshold=1, smooth=True,
                                      minsize=10, maxsize=None):
        mask = self.back_project_hue_histogram(model, smooth, fullColor=False,
                                               thresh=threshold)
        return self.find_blobs_from_mask(mask, minsize=minsize, maxsize=maxsize)

    def filter(self, flt, grayscale=False):
        filtered_image = flt.applyFilter(self, grayscale)
        return filtered_image


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
