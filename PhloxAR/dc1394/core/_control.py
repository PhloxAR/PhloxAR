# -----------------------------------------------------------------------------
#
# -*- coding: utf-8 -*-
#
# phlox-libdc1394/dc1394/core/_control.py
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
Libdc1394-control.h wrapper.
"""

from __future__ import unicode_literals
from ctypes import c_int, c_uint32, c_float, Structure
from ._types import bool_t, switch_t


__all__ = [
    'feature_info_t', 'feature_mode_t', 'feature_modes_t', 'feature_t',
    'featureset_t', 'feature_modes', 'features', 'FEATURE_MAX', 'FEATURE_MIN',
    'FEATURE_NUM', 'FEATURE_MODE_MAX', 'FEATURE_MODE_MIN', 'FEATURE_MODE_NUM',
    'trigger_mode_t', 'trigger_modes_t', 'trigger_polarity_t',
    'trigger_source_t', 'trigger_sources_t', 'trigger_polarity_t',
    'trigger_modes', 'trigger_sources', 'trigger_polarities',
    'TRIGGER_ACTIVE_MAX', 'TRIGGER_ACTIVE_MIN', 'TRIGGER_ACTIVE_NUM',
    'TRIGGER_MODE_MAX', 'TRIGGER_MODE_MIN', 'TRIGGER_MODE_NUM',
    'TRIGGER_SOURCE_MAX', 'TRIGGER_SOURCE_MIN', 'TRIGGER_SOURCE_NUM'
]

# Enumeration of trigger modes.
trigger_modes = {
    'TRIGGER_MODE_0': 384,
    'TRIGGER_MODE_1': 385,
    'TRIGGER_MODE_2': 386,
    'TRIGGER_MODE_3': 387,
    'TRIGGER_MODE_4': 388,
    'TRIGGER_MODE_5': 389,
    'TRIGGER_MODE_14': 390,
    'TRIGGER_MODE_15': 391
}

trigger_mode_t = c_int

TRIGGER_MODE_MIN = trigger_modes['TRIGGER_MODE_0']
TRIGGER_MODE_MAX = trigger_modes['TRIGGER_MODE_1']
TRIGGER_MODE_NUM = TRIGGER_MODE_MAX - TRIGGER_MODE_MIN + 1


# Enumeration of camera features.
features = {
    'FEATURE_BRIGHTNESS': 416,
    'FEATURE_EXPOSURE': 417,
    'FEATURE_SHARPNESS': 418,
    'FEATURE_WHITE_BALANCE': 419,
    'FEATURE_HUE': 420,
    'FEATURE_SATURATION': 421,
    'FEATURE_GAMMA': 422,
    'FEATURE_SHUTTER': 423,
    'FEATURE_GAIN': 424,
    'FEATURE_IRIS': 425,
    'FEATURE_FOCUS': 426,
    'FEATURE_TEMPERATURE': 427,
    'FEATURE_TRIGGER': 428,
    'FEATURE_TRIGGER_DELAY': 429,
    'FEATURE_WHITE_SHADING': 430,
    'FEATURE_FRAME_RATE': 431,
    'FEATURE_ZOOM': 432,
    'FEATURE_PAN': 433,
    'FEATURE_TILT': 434,
    'FEATURE_OPTICAL_FILTER': 435,
    'FEATURE_CAPTURE_SIZE': 436,
    'FEATURE_CAPTURE_QUALITY': 437,
}

feature_t = c_int

FEATURE_MIN = features['FEATURE_BRIGHTNESS']
FEATURE_MAX = features['FEATURE_CAPTURE_QUALITY']
FEATURE_NUM = FEATURE_MAX - FEATURE_MIN + 1

# Enumeration of trigger sources.
trigger_sources = {
    'TRIGGER_SOURCE_0': 576,
    'TRIGGER_SOURCE_1': 577,
    'TRIGGER_SOURCE_2': 578,
    'TRIGGER_SOURCE_3': 579,
    'TRIGGER_SOURCE_SOFTWARE': 580,
}

trigger_source_t = c_int

TRIGGER_SOURCE_MIN = trigger_sources['TRIGGER_SOURCE_0']
TRIGGER_SOURCE_MAX = trigger_sources['TRIGGER_SOURCE_SOFTWARE']
TRIGGER_SOURCE_NUM = TRIGGER_SOURCE_MAX - TRIGGER_SOURCE_MIN + 1

# External trigger polarity.
trigger_polarities = {
    'TRIGGER_ACTIVE_LOW': 0,
    'TRIGGER_ACTIVE_HIGH': 1,
}

trigger_polarity_t = c_int

TRIGGER_ACTIVE_MIN = trigger_polarities['TRIGGER_ACTIVE_LOW']
TRIGGER_ACTIVE_MAX = trigger_polarities['TRIGGER_ACTIVE_HIGH']
TRIGGER_ACTIVE_NUM = TRIGGER_ACTIVE_MAX - TRIGGER_ACTIVE_MIN + 1

# Control modes for features.
feature_modes = {
    'FEATURE_MODE_MANUAL': 736,
    'FEATURE_MODE_AUTO': 737,
    'FEATURE_MODE_ONE_PUSH_AUTO': 738,
}

feature_mode_t = c_int

FEATURE_MODE_MIN = feature_modes['FEATURE_MODE_MANUAL']
FEATURE_MODE_MAX = feature_modes['FEATURE_MODE_ONE_PUSH_AUTO']
FEATURE_MODE_NUM = FEATURE_MODE_MAX - FEATURE_MODE_MIN + 1


# List of features modes
class feature_modes_t(Structure):
    _fields_ = [
        ('num', c_uint32),
        ('modes', feature_mode_t),
    ]


# List of trigger modes
class trigger_modes_t(Structure):
    _fields_ = [
        ('num', c_uint32),
        ('modes', (trigger_mode_t) * TRIGGER_MODE_NUM),
    ]


# List of trigger sources
class trigger_sources_t(Structure):
    _fields_ = [
        ('num', c_uint32),
        ('sources', (trigger_mode_t) * TRIGGER_SOURCE_NUM)
    ]


# A structure containing all information about a features.
class feature_info_t(Structure):
    _fields_ = [
        ('id', feature_t),
        ('available', bool_t),
        ('readout_capable', bool_t),
        ('on_off_capable', bool_t),
        ('polarity_capable', bool_t),
        ('is_on', switch_t),
        ('current_mode', feature_mode_t),
        ('modes', feature_modes_t),
        ('trigger_modes', trigger_modes_t),
        ('trigger_mode', trigger_mode_t),
        ('trigger_polarity', trigger_polarity_t),
        ('trigger_sources', trigger_sources_t),
        ('trigger_source', trigger_source_t),
        ('min', c_uint32),
        ('max', c_uint32),
        ('value', c_uint32),
        ('BU_value', c_uint32),
        ('RV_value', c_uint32),
        ('B_value', c_uint32),
        ('R_value', c_uint32),
        ('G_value', c_uint32),
        ('target_value', c_uint32),
        ('abs_control', switch_t),
        ('abs_value', c_float),
        ('abs_max', c_float),
        ('abs_min', c_float),
    ]


# The list of features.
class featureset_t(Structure):
    _fields_ = [
        ('features', (feature_info_t) * FEATURE_NUM)
    ]
