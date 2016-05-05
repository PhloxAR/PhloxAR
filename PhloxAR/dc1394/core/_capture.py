# -----------------------------------------------------------------------------
#
# -*- coding: utf-8 -*-
#
# phlox-libdc1394/dc1394/core/_capture.py
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
from ctypes import c_int

__all__ = [
    'capture_policies', 'capture_policy_t', 'capture_flags', 'CAPTURE_POLICY_MAX',
    'CAPTURE_POLICY_MIN', 'CAPTURE_POLICY_NUM'
]


# The capture policy.
# Can be blocking (wait for a frame forever) or polling (returns if no frames
# is in the ring buffer)
capture_policies = {
    'CAPTURE_POLICY_WAIT': 672,
    'CAPTURE_POLICY_POLL': 673,
}

capture_policy_t = c_int

CAPTURE_POLICY_MIN = capture_policies['CAPTURE_POLICY_WAIT']
CAPTURE_POLICY_MAX = capture_policies['CAPTURE_POLICY_POLL']
CAPTURE_POLICY_NUM = CAPTURE_POLICY_MAX - CAPTURE_POLICY_MIN + 1

# Capture flags. Currently limited to switching automatic functions on/off:
# channel allocation, bandwidth allocation and automatic.
capture_flags = {
    'CAPTURE_FLAGS_CHANNEL_ALLOC': 0x00000001,
    'CAPTURE_FLAGS_BANDWIDTH_ALLOC': 0x00000002,
    'CAPTURE_FLAGS_DEFAULT': 0x00000004,
    'CAPTURE_FLAGS_AUTO_ISO': 0x00000008
}
