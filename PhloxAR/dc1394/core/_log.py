# -----------------------------------------------------------------------------
#
# -*- coding: utf-8 -*-
#
# phlox-libdc1394/dc1394/core/_log.py
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
Functions to log errors, warning and debug messages.
"""

from __future__ import unicode_literals
from ctypes import c_int
from ._types import invert

__all__ = [
    'QUERY_FROM_CAMERA', 'USE_MAX_AVAIL', 'USE_RECOMMANDED', 'format7mode_t',
    'format7modeset_t', 'error_t', 'errors', 'err_val', 'log_t', 'logs',
    'ERROR_NUM', 'ERROR_MIN', 'ERROR_MAX', 'LOG_NUM', 'LOG_MAX', 'LOG_MIN'
]

# Error codes.
# General rule: 0 is success, negative denotes a problem.
errors = {
    'SUCCESS': 0,
    'FAILURE': -1,
    'NOT_A_CAMERA': -2,
    'FUNCTION_NOT_SUPPORTED': -3,
    'CAMERA_NOT_INITIALIZED': -4,
    'MEMORY_ALLOCATION_FAILURE': -5,
    'TAGGED_REGISTER_NOT_FOUND': -6,
    'NO_ISO_CHANNEL': -7,
    'NO_BANDWIDTH': -8,
    'IOCTL_FAILURE': -9,
    'CAPTURE_IS_NOT_SET': -10,
    'CAPTURE_IS_RUNNING': -11,
    'RAW1394_FAILURE': -12,
    'FORMAT7_ERROR_FLAG_1': -13,
    'FORMAT7_ERROR_FLAG_2': -14,
    'INVALID_ARGUMENT_VALUE': -15,
    'REQ_VALUE_OUTSIDE_RANGE': -16,
    'INVALID_FEATURE': -17,
    'INVALID_VIDEO_FORMAT': -18,
    'INVALID_VIDEO_MODE': -19,
    'INVALID_FRAMERATE': -20,
    'INVALID_TRIGGER_MODE': -21,
    'INVALID_TRIGGER_SOURCE': -22,
    'INVALID_ISO_SPEED': -23,
    'INVALID_IIDC_VERSION': -24,
    'INVALID_COLOR_CODING': -25,
    'INVALID_COLOR_FILTER': -26,
    'INVALID_CAPTURE_POLICY': -27,
    'INVALID_ERROR_COD': -28,
    'INVALID_BAYER_METHOD': -29,
    'INVALID_VIDEO1394_DEVICE': -30,
    'INVALID_OPERATION_MODE': -31,
    'INVALID_TRIGGER_POLARITY': -32,
    'INVALID_FEATURE_MODE': -33,
    'INVALID_LOG_TYPE': -34,
    'INVALID_BYTE_ORDER': -35,
    'INVALID_STEREO_METHOD': -36,
    'BASLER_NO_MORE_SFF_CHUNKS': -37,
    'BASLER_CORRUPTED_SFF_CHUNK': -38,
    'BASLER_UNKNOWN_SFF_CHUNK': -39,
}

err_val = invert(errors)

error_t = c_int

ERROR_MIN = errors['BASLER_UNKNOWN_SFF_CHUNK']
ERROR_MAX = errors['SUCCESS']
ERROR_NUM = ERROR_MAX - ERROR_MIN + 1

# Types of logging messages
# - ERROR for real, hard, unrecoverable errors that will result in the program
#   terminating.
# - WARNING for things that have gone wrong, but are not requiring a
#   termination of the program.
# - DEBUG for debug messages that can be very verbose but may help the
#   developers to fix bugs.
logs = {
    'LOG_ERROR': 768,
    'LOG_WARNING': 769,
    'LOG_DEBUG': 770,
}

log_t = c_int

LOG_MIN = logs['LOG_ERROR']
LOG_MAX = logs['LOG_DEBUG']
LOG_NUM = LOG_MAX - LOG_MIN + 1
