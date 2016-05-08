# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from .core import *


__all__ = [
    'Mode', 'Format7', 'mode_map'
]


class Mode(object):
    """
    Video mode for a DC1394 camera.
    Do not instantiate this class directly. Instead use one of the modes
    in 'Camera.modes' or 'Camera.modes_dict' and assign it to 'Camera.mode'.
    """
    def __init__(self, cam, mode_id):
        self._mode_id = mode_id
        self._cam = cam
        self._dtype_shape()

    def __repr__(self):
        pass

    def __eq__(self, other):
        pass

    def _dtype_shape(self):
        pass

    @property
    def mode_id(self):
        return

    @property
    def name(self):
        return

    @property
    def framerate(self):
        return

    @property
    def shape(self):
        return

    @property
    def color_coding(self):
        return

    @property
    def scalable(self):
        return


class Exif(Mode):
    pass


class Format7(Mode):
    @property
    def frame_inverval(self):
        return

    @property
    def max_image_size(self):
        return

    @property
    def image_size(self):
        return

    @image_size.setter
    def image_size(self, width, height):
        pass

    @property
    def image_position(self):
        return

    @image_position.setter
    def image_position(self, pos):
        pass
    
    @property
    def color_codings(self):
        return

    @property
    def color_coding(self):
        return

    @color_coding.setter
    def color_coding(self, color):
        pass

    @property
    def unit_position(self):
        return

    @property
    def unit_size(self):
        return

    @property
    def roi(self):
        return

    @roi.setter
    def roi(self, args):
        pass

    @property
    def dtype(self):
        return 
    
    @property
    def shape(self):
        return
    
    @property
    def packet_parameters(self):
        return 
    
    @property
    def packet_size(self):
        return

    @packet_size.setter
    def packet_size(self, pkt_size):
        pass

    @property
    def data_depth(self):
        return

    def setup(self, img_size=(QUERY_FROM_CAMERA, QUERY_FROM_CAMERA),
              img_pos=(QUERY_FROM_CAMERA, QUERY_FROM_CAMERA),
              color_coding=QUERY_FROM_CAMERA, pkt_size=USE_RECOMMANDED):
        pass


mode_map = {
    64: Mode,
    65: Mode,
    66: Mode,
    67: Mode,
    68: Mode,
    69: Mode,
    70: Mode,
    71: Mode,
    72: Mode,
    73: Mode,
    74: Mode,
    75: Mode,
    76: Mode,
    77: Mode,
    78: Mode,
    79: Mode,
    80: Mode,
    81: Mode,
    82: Mode,
    83: Mode,
    84: Mode,
    85: Mode,
    86: Mode,
    87: Exif,
    88: Format7,
    89: Format7,
    90: Format7,
    91: Format7,
    92: Format7,
    93: Format7,
    94: Format7,
    95: Format7,
}


def create_mode(cam, m):
    if isinstance(m, tuple):
        m = "%sx%s_%s" % m
    return Mode(cam, video_modes[m])
