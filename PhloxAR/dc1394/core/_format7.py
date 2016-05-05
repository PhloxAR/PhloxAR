# -----------------------------------------------------------------------------
#
# -*- coding: utf-8 -*-
#
# phlox-libdc1394/dc1394/core/_format7.py
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
Functions to control Format_7 (aka scalable format, ROI)
"""

from __future__ import unicode_literals
from ctypes import c_uint32, c_uint64, Structure
from ._types import bool_t, color_coding_t, color_codings_t, color_filter_t
from ._types import VIDEO_MODE_FORMAT7_NUM

__all__ = [
    'QUERY_FROM_CAMERA', 'USE_MAX_AVAIL', 'USE_RECOMMANDED', 'format7mode_t',
    'format7modeset_t'
]

QUERY_FROM_CAMERA = -1
USE_MAX_AVAIL = -2
USE_RECOMMANDED = -3

# A struct containing information about a mode of format_7,
# the scalable image format.
class format7mode_t(Structure):
    _fields_ = [
        ('present', bool_t),

        ('size_x', c_uint32),
        ('size_y', c_uint32),
        ('max_size_x', c_uint32),
        ('max_size_y', c_uint32),

        ('pos_x', c_uint32),
        ('pos_y', c_uint32),

        ('unit_size_x', c_uint32),
        ('unit_size_y', c_uint32),
        ('unit_pos_x', c_uint32),
        ('unit_pos_y', c_uint32),

        ('color_codings', color_codings_t),
        ('color_coding', color_coding_t),

        ('pixnum', c_uint32),

        ('packet_size', c_uint32),
        ('unit_packet_size', c_uint32),
        ('max_packet_size', c_uint32),

        ('total_bytes', c_uint64),

        ('color_filter', color_filter_t)
    ]


# A struct containing the list of Format_7 modes.
class format7modeset_t(Structure):
    _fields_ = [
        ('mode', (format7mode_t) * VIDEO_MODE_FORMAT7_NUM),
    ]
