# -----------------------------------------------------------------------------
#
# -*- coding: utf-8 -*-
#
# phlox-libdc1394/dc1394/core/_video.py
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
Functions related to video modes, formats, framerate and video flow.
"""

from __future__ import unicode_literals
from ctypes import c_int, c_uint32, c_void_p, c_uint64
from ctypes import POINTER, Structure
from ._types import color_coding_t, color_filter_t, video_mode_t, bool_t
from ._camera import camera_t

__all__ = [
    'iso_speeds', 'iso_speed_t', 'ISO_SPEED_MAX', 'ISO_SPEED_MIN', 'ISO_SPEED_NUM',
    'framerates', 'framerate_t', 'framerates_t', 'FRAMERATE_MAX', 'FRAMERATE_MIN',
    'FRAMERATE_NUM', 'operation_modes', 'operation_mode_t', 'OPERATION_MODE_MAX',
    'OPERATION_MODE_MIN', 'OPERATION_MODE_NUM', 'video_frame_t'
]

# Enumeration of iso data speeds.
# Most (if not all) cameras are compatible with 400Mbps speed. Only older
# cameras (pre-1999) may still only work at sub-400 speeds. However, speeds
# lower than 400Mbps are still useful: they can be used for longer distances
# (e.g. 10m cables). Speeds over 400Mbps are only available in "B" mode
# (OPERATION_MODE_1394B).
iso_speeds = {
    'ISO_SPEED_100': 0,
    'ISO_SPEED_200': 1,
    'ISO_SPEED_400': 2,
    'ISO_SPEED_800': 3,
    'ISO_SPEED_1600': 4,
    'ISO_SPEED_3200': 5,
}

iso_speed_t = c_int

ISO_SPEED_MIN = iso_speeds['ISO_SPEED_100']
ISO_SPEED_MAX = iso_speeds['ISO_SPEED_3200']
ISO_SPEED_NUM = ISO_SPEED_MAX - ISO_SPEED_MIN + 1

# Enumeration of video framerates.
# This enumeration is used for non-Format_7 modes. The framerate can be lower
# than expected if the exposure time is longer than the requested frame period.
# Framerate can be controlled in a number of other ways: framerate features,
# external trigger, software trigger, shutter throttling and packet
# size (Format_7)
framerates = {
    'FRAMERATE_1_875': 32,
    'FRAMERATE_3_75': 33,
    'FRAMERATE_7_5': 34,
    'FRAMERATE_15': 35,
    'FRAMERATE_30': 36,
    'FRAMERATE_60': 37,
    'FRAMERATE_120': 38,
    'FRAMERATE_240': 39,
}

framerate_t = c_int

FRAMERATE_MIN = framerates['FRAMERATE_1_875']
FRAMERATE_MAX = framerates['FRAMERATE_240']
FRAMERATE_NUM = FRAMERATE_MAX - FRAMERATE_MIN + 1

# Operation modes.
# Two operation modes exist: the legacy and most common 1394a, and the newer
# 1394B. The latter allows speeds over 400Mbps, but can also be used at other speeds.
operation_modes = {
    'OPERATION_MODE_LEGACY': 480,
    'OPERATION_MODE_1394B': 481,
}

operation_mode_t = c_int

OPERATION_MODE_MIN = operation_modes['OPERATION_MODE_LEGACY']
OPERATION_MODE_MAX = operation_modes['OPERATION_MODE_1394B']
OPERATION_MODE_NUM = OPERATION_MODE_MAX - OPERATION_MODE_MIN + 1


# List of framerates.
class framerates_t(Structure):
    _fields_ = [
        ('num', c_uint32),
        ('framerates', (framerate_t) * FRAMERATE_NUM)
    ]


# Video frame structure.
# video_frame_t is the structure returned by the capture functions.
# It contains the captured image as well as a number of information.
class video_frame_t(Structure):
    _fields_ = [
        ('image', c_void_p),  # unsigned char*
        ('size', c_uint32 * 2),
        ('color_coding', color_coding_t),
        ('color_filter', color_filter_t),
        ('yuv_byte_order', c_uint32),
        ('data_depth', c_uint32),
        ('stride', c_uint32),
        ('vidoe_mode', video_mode_t),
        ('total_bytes', c_uint64),
        ('padding_bytes', c_uint32),
        ('packet_size', c_uint32),
        ('packets_per_frame', c_uint32),
        ('timestamp', c_uint64),
        ('frames_behind', c_uint32),
        ('camera', POINTER(camera_t)),
        ('id', c_uint32),
        ('allocated_image_bytes', c_uint64),
        ('little_endian', bool_t),
        ('data_in_padding', bool_t),
    ]
