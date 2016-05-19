# -----------------------------------------------------------------------------
#
# -*- coding: utf-8 -*-
#
# phlox-libdc1394/dc1394/core/_types.py
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
A few type definitions.
"""

from __future__ import unicode_literals
from ctypes import c_int, c_uint32, Structure

__all__ = [
    'video_modes', 'video_mode_t', 'video_modes_t', 'video_modes_detailed',
    'VIDEO_MODE_MAX', 'VIDEO_MODE_MIN', 'VIDEO_MODE_NUM',
    'VIDEO_MODE_FORMAT7_MAX', 'VIDEO_MODE_FORMAT7_MIN', 'VIDEO_MODE_FORMAT7_NUM',
    'color_coding_t', 'color_codings', 'COLOR_CODING_MAX', 'COLOR_CODING_MIN',
    'COLOR_CODING_NUM', 'color_filter_t', 'color_filters', 'COLOR_FILTER_MAX',
    'COLOR_FILTER_MIN', 'COLOR_FILTER_NUM', 'byte_order_t', 'byte_orders',
    'BYTE_ORDER_MAX', 'BYTE_ORDER_MIN', 'BYTE_ORDER_NUM', 'color_codings_t',
    'video_modes_t', 'bool_t', 'switch_t', 'invert'
]

# --------------------------------- enums ------------------------------------
# def enum(seq, start=0):
#     n = start
#     enums = {}
#     if isinstance(seq, list):
#         for elem in seq:
#             enums[elem] = n
#             n += 1
#     elif isinstance(seq, dict):
#         enums = seq
#     return type('Enum', (), enums)


def invert(to):
    tmp = {}
    for i, j in to.items():
        tmp[j] = i
    return tmp

# Enumeration of video modes.
# Note that the notion of IIDC "format" is not present here, except in the
# format_7 name.
video_modes = {
    '160x120_YUV444': 64,
    '320x240_YUV422': 65,
    '640x480_YUV411': 66,
    '640x480_YUV422': 67,
    '640x480_RGB8': 68,
    '640x480_MONO8': 69,
    '640x480_MONO16': 70,
    '800x600_YUV422': 71,
    '800x600_RGB8': 72,
    '800x600_MONO8': 73,
    '1024x768_YUV422': 74,
    '1024x768_RGB8': 75,
    '1024x768_MONO8': 76,
    '800x600_MONO16': 77,
    '1024x768_MONO16': 78,
    '1280x960_YUV422': 79,
    '1280x960_RGB8': 80,
    '1280x960_MONO8': 81,
    '1600x1200_YUV422': 82,
    '1600x1200_RGB8': 83,
    '1600x1200_MONO8': 84,
    '1280x960_MONO16': 85,
    '1600x1200_MONO16': 86,
    'EXIF': 87,
    'FORMAT7_0': 88,
    'FORMAT7_1': 89,
    'FORMAT7_2': 90,
    'FORMAT7_3': 91,
    'FORMAT7_4': 92,
    'FORMAT7_5': 93,
    'FORMAT7_6': 94,
    'FORMAT7_7': 95,
}

video_mode_t = c_int

video_modes['640x480_Y8'] = video_modes['640x480_MONO8']
video_modes['800x600_Y8'] = video_modes['800x600_MONO8']
video_modes['1024x768_Y8'] = video_modes['1024x768_MONO8']
video_modes['1280x960_Y8'] = video_modes['1280x960_MONO8']
video_modes['1600x1200_Y8'] = video_modes['1600x1200_MONO8']
video_modes['640x480_Y16'] = video_modes['640x480_MONO16']
video_modes['800x600_Y16'] = video_modes['800x600_MONO16']
video_modes['1024x768_Y16'] = video_modes['1024x768_MONO16']
video_modes['1280x960_Y16'] = video_modes['1280x960_MONO16']
video_modes['1600x1200_Y16'] = video_modes['1600x1200_MONO16']

VIDEO_MODE_MIN = video_modes['160x120_YUV444']
VIDEO_MODE_MAX = video_modes['FORMAT7_7']
VIDEO_MODE_NUM = VIDEO_MODE_MAX - VIDEO_MODE_MIN + 1


# Special min/max are defined for Format_7
VIDEO_MODE_FORMAT7_MIN = video_modes['FORMAT7_0']
VIDEO_MODE_FORMAT7_MAX = video_modes['FORMAT7_7']
VIDEO_MODE_FORMAT7_NUM = VIDEO_MODE_FORMAT7_MAX - VIDEO_MODE_FORMAT7_MIN + 1

video_modes_detailed = {
    64: (160, 120, 'YUV444'),
    65: (320, 240, 'YUV422'),
    66: (640, 480, 'YUV411'),
    67: (640, 480, 'YUV422'),
    68: (640, 480, 'RGB8'),
    69: (640, 480, 'MONO8'),
    70: (640, 480, 'MONO16'),
    71: (800, 600, 'YUV422'),
    72: (800, 600, 'RGB8'),
    73: (800, 600, 'MONO8'),
    74: (1024, 768, 'YUV422'),
    75: (1024, 768, 'RGB8'),
    76: (1024, 768, 'MONO8'),
    77: (800, 600, 'MONO16'),
    78: (1024, 768, 'MONO16'),
    79: (1280, 960, 'YUV422'),
    80: (1280, 960, 'RGB8'),
    81: (1280, 960, 'MONO8'),
    82: (1600, 1200, 'YUV422'),
    83: (1600, 1200, 'RGB8'),
    84: (1600, 1200, 'MONO8'),
    85: (1280, 960, 'MONO16'),
    86: (1600, 1200, 'MONO16'),
}

# Enumeration of _color codings.
color_codings = {
    'COLOR_CODING_MONO8': 352,
    'COLOR_CODING_YUV411': 353,
    'COLOR_CODING_YUV422': 354,
    'COLOR_CODING_YUV444': 355,
    'COLOR_CODING_RGB8': 356,
    'COLOR_CODING_MONO16': 357,
    'COLOR_CODING_RGB16': 358,
    'COLOR_CODING_MONO16S': 359,
    'COLOR_CODING_RGB16S': 360,
    'COLOR_CODING_RAW8': 361,
    'COLOR_CODING_RAW16': 362,
}

color_coding_t = c_int

color_codings['COLOR_CODING_Y8'] = color_codings['COLOR_CODING_MONO8']
color_codings['COLOR_CODING_Y16'] = color_codings['COLOR_CODING_MONO16']
color_codings['COLOR_CODING_Y16S'] = color_codings['COLOR_CODING_MONO16S']

COLOR_CODING_MIN = color_codings['COLOR_CODING_MONO8']
COLOR_CODING_MAX = color_codings['COLOR_CODING_RAW16']
COLOR_CODING_NUM = COLOR_CODING_MAX - COLOR_CODING_MIN + 1

# RAW sensor filters, these elementary tiles tesselate the image plane in RAW
# modes. RGGB should be interpreted in 2D as
#     RG
#     GB
# and similarly for other filters.
color_filters = {
    'COLOR_FILTER_RGGB': 512,
    'COLOR_FILTER_GBRG': 513,
    'COLOR_FILTER_GRBG': 514,
    'COLOR_FILTER_BGGR': 515,
}

color_filter_t = c_int

COLOR_FILTER_MIN = color_filters['COLOR_FILTER_RGGB']
COLOR_FILTER_MAX = color_filters['COLOR_FILTER_BGGR']
COLOR_FILTER_NUM = COLOR_FILTER_MAX - COLOR_FILTER_MIN + 1

# Byte order for YUV formats.
# IIDC cameras always return data in UYVY order, but conversion functions can
# change this if requested.
byte_orders = {
    'BYTE_ORDER_UYVY': 800,
    'BYTE_ORDER_YUYV': 801,
}

byte_order_t = c_int

BYTE_ORDER_MIN = byte_orders['BYTE_ORDER_UYVY']
BYTE_ORDER_MAX = byte_orders['BYTE_ORDER_YUYV']
BYTE_ORDER_NUM = BYTE_ORDER_MAX - BYTE_ORDER_MIN + 1


# A structure containing a list of _color codings.
class color_codings_t(Structure):
    _fields_ = [
        ('num', c_uint32),
        ('codings', color_coding_t * COLOR_CODING_NUM),
    ]


# A structure containing a list of video modes.
class video_modes_t(Structure):
    _fields_ = [
        ('num', c_uint32),
        ('modes', video_mode_t * VIDEO_MODE_NUM),
    ]

bool_t = c_int
switch_t = c_int
