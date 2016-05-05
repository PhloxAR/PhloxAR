# -----------------------------------------------------------------------------
#
# -*- coding: utf-8 -*-
#
# phlox-libdc1394/dc1394/_camera.py
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
from __future__ import unicode_literals
from ctypes import c_int, c_uint16, c_uint32, c_uint64, c_char_p
from ctypes import POINTER, Structure
from ._types import VIDEO_MODE_FORMAT7_NUM, bool_t

__all__ = [
    'iidc_version', 'iidc_version_t', 'IIDC_VERSION_MAX', 'IIDC_VERSION_MIN',
    'IIDC_VERSION_NUM', 'power_classes', 'power_class_t', 'POWER_CLASS_MAX',
    'POWER_CLASS_MIN', 'POWER_CLASS_NUM', 'phy_delays', 'phy_delay_t',
    'PHY_DELAY_MAX', 'PHY_DELAY_MIN', 'PHY_DELAY_NUM', 'camera_id_t',
    'camera_list_t', 'camera_t'
]

# List of IIDC versions.
# Currently, the following versions exist: 1.04, 1.20, PTGREY, 1.30 and 1.31
iidc_version = {
    'IIDC_VERSION_1_04': 544,
    'IIDC_VERSION_1_20': 545,
    'IIDC_VERSION_PTGREY': 546,
    'IIDC_VERSION_1_30': 547,
    'IIDC_VERSION_1_31': 548,
    'IIDC_VERSION_1_32': 549,
    'IIDC_VERSION_1_33': 550,
    'IIDC_VERSION_1_34': 551,
    'IIDC_VERSION_1_35': 552,
    'IIDC_VERSION_1_36': 553,
    'IIDC_VERSION_1_37': 554,
    'IIDC_VERSION_1_38': 555,
    'IIDC_VERSION_1_39': 556,
}

iidc_version_t = c_int

IIDC_VERSION_MIN = iidc_version['IIDC_VERSION_1_04']
IIDC_VERSION_MAX = iidc_version['IIDC_VERSION_1_39']
IIDC_VERSION_NUM = IIDC_VERSION_MAX - IIDC_VERSION_MIN + 1

# Enumeration of power classes.
# This is currently not used in libdc1394.
power_classes = {
    'POWER_CLASS_NONE': 608,
    'POWER_CLASS_PROV_MIN_15W': 609,
    'POWER_CLASS_PROV_MIN_30W': 610,
    'POWER_CLASS_PROV_MIN_45W': 611,
    'POWER_CLASS_USES_MAX_1W': 612,
    'POWER_CLASS_USES_MAX_3W': 613,
    'POWER_CLASS_USES_MAX_6W': 614,
    'POWER_CLASS_USES_MAX_10W': 615,
}

power_class_t = c_int

POWER_CLASS_MIN = power_classes['POWER_CLASS_NONE']
POWER_CLASS_MAX = power_classes['POWER_CLASS_USES_MAX_10W']
POWER_CLASS_NUM = POWER_CLASS_MAX - POWER_CLASS_MIN + 1

# Enumeration of PHY delays.
# This is currently not used in libdc1394.
phy_delays = {
    'PHY_DELAY_MAX_144_NS': 640,
    'PHY_DELAY_UNKNOWN_0': 641,
    'PHY_DELAY_UNKNOWN_1': 642,
    'PHY_DELAY_UNKNOWN_2': 643,
}

phy_delay_t = c_int

PHY_DELAY_MIN = phy_delays['PHY_DELAY_MAX_144_NS']
PHY_DELAY_MAX = phy_delays['PHY_DELAY_UNKNOWN_2']
PHY_DELAY_NUM = PHY_DELAY_MAX - PHY_DELAY_MIN + 1


# Camera structure.
class camera_t(Structure):
    _fields_ = [
        ('guid', c_uint64),
        ('uint', c_int),
        ('unit_spec_ID', c_uint32),
        ('unit_sw_version', c_uint32),
        ('unit_sub_sw_version', c_uint32),
        ('command_registers_base', c_uint32),
        ('unit_directory', c_uint32),
        ('unit_dependent_directory', c_uint32),
        ('advanced_features_csr', c_uint64),
        ('PIO_control_csr', c_uint64),
        ('SIO_control_csr', c_uint64),
        ('strobe_control_csr', c_uint64),
        ('format7_csr', c_uint64 * VIDEO_MODE_FORMAT7_NUM),
        ('iidc_version', iidc_version_t),
        ('vendor', c_char_p),
        ('model', c_char_p),
        ('vendor_id', c_uint32),
        ('model_id', c_uint32),
        ('bmode_capable', bool_t),
        ('one_shot_capable', bool_t),
        ('multi_shot_capable', bool_t),
        ('can_switch_on_off', bool_t),
        ('has_vmode_error_status', bool_t),
        ('has_feature_error_status', bool_t),
        ('max_men_channel', c_int),
        ('flags', c_uint32),
    ]


# A unique identifier for a functional camera unit.
# Since a single camera can contain several functional units (think stereo
# cameras), the GUID is not enough to identify an IIDC camera.
class camera_id_t(Structure):
    _fields_ = [
        ('unit', c_uint16),
        ('guid', c_uint64),
    ]


# A list of cameras.
class camera_list_t(Structure):
    _fields_ = [
        ('num', c_uint32),
        ('ids', POINTER(camera_id_t)),
    ]
