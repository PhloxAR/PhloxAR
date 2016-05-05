# -----------------------------------------------------------------------------
#
# -*- coding: utf-8 -*-
#
# phlox-libdc1394/dc1394/core/_conversions.py
#
# Copyright (C) 2016, by Matthias Yang Chen <matthias_cy@outlook.com>
# All rights reserved.
#
# phlox-libdc1394 is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# phlox-libdc1394 is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with phlox-libdc1394. If not,
# see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
"""
Functions to convert video formats.
"""

from __future__ import unicode_literals
from ctypes import c_int

__all__ = [
    'bayer_method_t', 'bayer_methods', 'stereo_method_t', 'stereo_methods',
    'BAYER_METHOD_MAX', 'BAYER_METHOD_MIN', 'BAYER_METHOD_NUM',
    'STEREO_METHOD_MAX', 'STEREO_METHOD_MIN', 'STEREO_METHOD_NUM'
]

# A list of de-mosaicing techniques for Bayer-patterns.
# The speed of the techniques can vary greatly, as well as their quality.
bayer_methods = {
    'BAYER_METHOD_NEAREST': 0,
    'BAYER_METHOD_SIMPLE': 1,
    'BAYER_METHOD_BILINEAR': 2,
    'BAYER_METHOD_HQLINEAR': 3,
    'BAYER_METHOD_DOWNSAMPLE': 4,
    'BAYER_METHOD_EDGESENSE': 5,
    'BAYER_METHOD_VNG': 6,
    'BAYER_METHOD_AHD': 7,
}

bayer_method_t = c_int

BAYER_METHOD_MIN = bayer_methods['BAYER_METHOD_NEAREST']
BAYER_METHOD_MAX = bayer_methods['BAYER_METHOD_AHD']
BAYER_METHOD_NUM = BAYER_METHOD_MAX - BAYER_METHOD_MIN + 1

# A list of known stereo-in-normal-video modes used by manufacturers like
# Point Grey Research and Videre Design.
stereo_methods = {
    'STEREO_METHOD_INTERLACED': 0,
    'STEREO_METHOD_FIELD': 1,
}

stereo_method_t = c_int

STEREO_METHOD_MIN = stereo_methods['STEREO_METHOD_INTERLACED']
STEREO_METHOD_MAX = stereo_methods['STEREO_METHOD_FIELD']
STEREO_METHOD_NUM = STEREO_METHOD_MAX - STEREO_METHOD_MIN + 1
