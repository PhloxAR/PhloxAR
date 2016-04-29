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


#def enum(*seq):
#    n = 0
#    enums = {}
#    if isinstance(seq, dict):
#        enums = seq
#    else:
#        for elem in seq:
#            enums[elem] = n
#            n += 1
#    return type('Enum', (), enums)

#ColorSpace = enum('UNKNOWN', 'BGR', 'GRAY', 'RGB', 'HLS', 'HSV', 'XYZ', 'YCrCb')

class ColorSpace(object):
    UNKNOWN = 0
    BGR = 1
    GRAY = 2
    HLS = 4
    HSV = 5
    RGB = 3
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
    _gray_matrix = ''  # the gray scale (cvmat) representation
    _equalize_gray_bitmap = ''  # the normalized bitmap
    _blob_label = ''  # the label image for blobbing
    _edge_map = ''  # holding reference for edge map
    _canny_param = ''  # parameters that created _edge_map
    _pilimage = ''  # holds a PIL object in buffer
    _numpy = ''  # numpy form buffer
    _gray_numpy = ''  # gray scale numpy for key point stuff
    _color_space = ColorSpace.UNKNOWN
    _surface = ''
    _cv2numpy = None  # numpy array for OpenCV >= 2.3
    _cv2gray_numpy = None  # grayscale numpy array for OpenCV >= 2.3
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
        '_numpy': '',
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
                self._numpy = src
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
        elif type(src) == str or src.__class__.__name__ == 'StringIO':
            if src == '':
                raise IOError("No filename provided to Image constructor")
            elif webp or src.split('.')[-1] == 'webp':
                try:
                    if src.__class__.__name__ == 'StringIO':
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

        elif (PIL_ENABLED and ((len(src.__class__.__bases__)
                                and src.__class__.__bases__[0].__name__ == "ImageFile")
                               or src.__class__.__name__ == "JpegImageFile"
                               or src.__class__.__name__ == "WebPPImageFile"
                               or src.__class__.__name__ == "Image")):
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