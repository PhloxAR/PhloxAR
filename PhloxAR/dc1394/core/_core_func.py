# -----------------------------------------------------------------------------
#
# -*- coding: utf-8 -*-
#
# phlox-libdc1394/phloxar-dc1394/core/_core.py
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
Core functions of libdc1394.
"""

from __future__ import division, print_function, unicode_literals
from ctypes import *
from ctypes.util import find_library
from ctypes import POINTER as PTR
from ._camera import *
from ._capture import *
from ._conversions import *
from ._control import *
from ._format7 import *
from ._log import *
from ._types import *
from ._video import *

__all__ = [
    'dll'
]

# REMINDER:
#  By default the ctypes API does not know or care about how a dll function
#  should be called. This may lead a lot of crashes and memory corruption.
#  I think it is safer to declare all functions here, independent of usage,
#  so at least the API can handle the ingoing/returning parameters properly.

try:
    _dll = cdll.LoadLibrary(find_library('phloxar-dc1394'))
except Exception as e:
    raise RuntimeError("Fatal: libdc1394 could not be found or open: %s" % e)


# ---------------------------- python functions -------------------------------
# Global Error checking functions
def _errcheck(rtype, func, arg):
    """
    This function checks for the error types declared by the error_t.
    Use it for functions with restype = error_t to receive correct error
    messages from the library.
    """
    if rtype != 0:
        raise RuntimeError("Error in phloxar-dc1394 function call: %s" %
                           err_val(rtype))


# ------------------------ Startup functions: camera.h ------------------------
#   Creates a new context in which cameras can be searched and used. This
#   should be called before using any other libdc1394 functions.
_dll.dc1394_new.argtypes = None
_dll.dc1394_new.restype = c_void_p

#   Liberates a context. Last function to use in your program. After this, no
#   libdc1394 function can be used.
_dll.dc1394_free.argtypes = [c_void_p]
_dll.dc1394_free.restype = None

# Bus level functions:
#   Sets and gets the broadcast flag of a camera. If the broadcast flag is set,
#   all devices on the bus will execute the command. Useful to sync ISO start
#   commands or setting a bunch of cameras at the same time. Broadcast only
#   works with identical devices (brand/model). If the devices are not
#   identical your mileage may vary. Some cameras may not answer broadcast
#   commands at all. Also, this only works with cameras on the SAME bus
#   (IOW, the same port).
_dll.dc1394_camera_set_broadcast.argtypes = [PTR(camera_t), bool_t]
_dll.dc1394_camera_set_broadcast.restype = error_t
_dll.dc1394_camera_set_broadcast.errcheck = _errcheck

_dll.dc1394_camera_get_broadcast.argtypes = [PTR(camera_t), PTR(bool_t)]
_dll.dc1394_camera_get_broadcast.restype = error_t
_dll.dc1394_camera_get_broadcast.errcheck = _errcheck

# Resets the IEEE1394 bus which camera is attached to. Calling this function
# is "rude" to other devices because it causes them to re-enumerate on the bus
# may cause a temporary disruption in their current activities. Thus, use it
# sparingly. Its primary use is if a program shuts down uncleanly and needs
# to free leftover ISO channels or bandwidth. A bus reset will free those
# things as a side effect.
_dll.dc1394_reset_bus.argtypes = [PTR(camera_t)]
_dll.dc1394_reset_bus.restype = error_t
_dll.dc1394_reset_bus.errcheck = _errcheck

_dll.dc1394_read_cycle_timer.argtypes = [PTR(camera_t), PTR(c_uint32), PTR(c_uint64)]
_dll.dc1394_read_cycle_timer.restype = error_t
_dll.dc1394_read_cycle_timer.errcheck = _errcheck

# Gets the IEEE 1394 node ID of the camera.
_dll.dc1394_camera_get_node.argtypes = [PTR(camera_t), PTR(c_uint32), PTR(c_uint32)]
_dll.dc1394_camera_get_node.restype = error_t
_dll.dc1394_camera_get_node.errcheck = _errcheck


# ----------------------------- Camera functions ------------------------------
# Returns the list of cameras available on the computer. If present, multiple
# cards will be probed.
_dll.dc1394_camera_enumerate.argtypes = [c_void_p, PTR(PTR(camera_list_t))]
_dll.dc1394_camera_enumerate.restype = error_t
_dll.dc1394_camera_enumerate.errcheck = _errcheck

# Frees the memory allocated in dc1394_enumerate_cameras for the camera list
_dll.dc1394_camera_free_list.argtypes = [PTR(camera_list_t)]
_dll.dc1394_camera_free_list.restype = None

# Create a new camera on a GUID (Global Unique IDentifier)
_dll.dc1394_camera_new.argtypes = [c_void_p, c_uint64]
_dll.dc1394_camera_new.restype = PTR(camera_t)

# Create a new camera based on a GUID and a unit number(for multi-unit cameras)
_dll.dc1394_camera_new_unit.argtypes = [c_void_p, c_uint64, c_int]
_dll.dc1394_camera_new_unit.restype = PTR(camera_t)

# Free a camera structure
_dll.dc1394_camera_free.argtypes = [PTR(camera_t)]
_dll.dc1394_camera_free.restype = None

# Print various camera information, ushc as GUID, vendor, model, supported IIDC
# specs, etc...
# dc1394error_t dc1394_camera_print_info(dc1394camera_t *camera, FILE *fd);
_dll.dc1394_camera_print_info.argtypes = [PTR(camera_t), c_void_p]
_dll.dc1394_camera_print_info.restype = error_t
_dll.dc1394_camera_print_info.errcheck = _errcheck

# ------------------------ Feature control: control.h  ------------------------
# Collects the available features fro the camera described by node and stores
# in features.
_dll.dc1394_feature_get_all.argtypes = [PTR(camera_t), PTR(featureset_t)]
_dll.dc1394_feature_get_all.restype = error_t
_dll.dc1394_feature_get_all.errcheck = _errcheck

# Stores the bounds and options associated with the feature described by
# feature->feature_id
_dll.dc1394_feature_get.argtypes = [PTR(camera_t), PTR(feature_info_t)]
_dll.dc1394_feature_get.restype = error_t
_dll.dc1394_feature_get.errcheck = _errcheck

# Displays the bounds and options of the given feature
# dc1394error_t dc1394_feature_print(dc1394feature_info_t *feature, FILE *fd);
_dll.dc1394_feature_print.argtypes = [PTR(feature_info_t), c_void_p]
_dll.dc1394_feature_print.restype = error_t
_dll.dc1394_feature_print.errcheck = _errcheck

# Displays the bounds and options of every feature supported by the camera
_dll.dc1394_feature_print_all.argtypes = [PTR(featureset_t), c_void_p]
_dll.dc1394_feature_print_all.restype = error_t
_dll.dc1394_feature_print_all.errcheck = _errcheck

# White balance: get/set
_dll.dc1394_feature_whitebalance_get_value.argtypes = [PTR(camera_t), PTR(c_uint32), PTR(c_uint32)]
_dll.dc1394_feature_whitebalance_get_value.restype = error_t
_dll.dc1394_feature_whitebalance_get_value.errcheck = _errcheck

_dll.dc1394_feature_whitebalance_set_value.argtypes = [PTR(camera_t), c_uint32, c_uint32]
_dll.dc1394_feature_whitebalance_set_value.restype = error_t
_dll.dc1394_feature_whitebalance_set_value.errcheck = _errcheck

# Temperature: get/set
_dll.dc1394_feature_temperature_get_value.argtypes = [PTR(camera_t), PTR(c_uint32), PTR(c_uint32)]
_dll.dc1394_feature_temperature_get_value.restype = error_t
_dll.dc1394_feature_temperature_get_value.errcheck = _errcheck

_dll.dc1394_feature_temperature_set_value.argtypes = [PTR(camera_t), c_uint32]
_dll.dc1394_feature_temperature_set_value.restype = error_t
_dll.dc1394_feature_temperature_set_value.errcheck = _errcheck

# White shading: get/set
_dll.dc1394_feature_whiteshading_get_value.argtypes = [PTR(camera_t), PTR(c_uint32), PTR(c_uint32), PTR(c_uint32)]
_dll.dc1394_feature_whiteshading_get_value.restype = error_t
_dll.dc1394_feature_whiteshading_get_value.errcheck = _errcheck

_dll.dc1394_feature_whiteshading_set_value.argtypes = [PTR(camera_t), c_uint32, c_uint32, c_uint32]
_dll.dc1394_feature_whiteshading_set_value.restype = error_t
_dll.dc1394_feature_whiteshading_set_value.errcheck = _errcheck

# Feature value: get/set
_dll.dc1394_feature_get_value.argtypes = [PTR(camera_t), PTR(feature_t), PTR(c_uint32)]
_dll.dc1394_feature_get_value.restype = error_t
_dll.dc1394_feature_get_value.errcheck = _errcheck

_dll.dc1394_feature_set_value.argtypes = [PTR(camera_t), feature_t, c_uint32]
_dll.dc1394_feature_set_value.restype = error_t
_dll.dc1394_feature_set_value.errcheck = _errcheck

# Tells whether a feature is present or not
_dll.dc1394_feature_is_present.argtypes = [PTR(camera_t), feature_t, PTR(bool_t)]
_dll.dc1394_feature_is_present.restype = error_t
_dll.dc1394_feature_is_present.errcheck = _errcheck

# Tells whether a feature is readable or not
_dll.dc1394_feature_is_readable.argtypes = [PTR(camera_t), feature_t, PTR(bool_t)]
_dll.dc1394_feature_is_readable.restype = error_t
_dll.dc1394_feature_is_readable.errcheck = _errcheck

# Gets the boundaries of a feature
_dll.dc1394_feature_get_boundaries.argtypes = [PTR(camera_t), feature_t, PTR(c_uint32), PTR(c_uint32)]
_dll.dc1394_feature_get_boundaries.restype = error_t
_dll.dc1394_feature_get_boundaries.errcheck = _errcheck

# Tells whether a feature is switcheable or not (ON/OFF)
_dll.dc1394_feature_is_switchable.argtypes = [PTR(camera_t), feature_t, PTR(bool_t)]
_dll.dc1394_feature_is_switchable.restype = error_t
_dll.dc1394_feature_is_switchable.errcheck = _errcheck

# Power status of a feature (ON/OFF): get/set
_dll.dc1394_feature_get_power.argtypes = [PTR(camera_t), feature_t, PTR(switch_t)]
_dll.dc1394_feature_get_power.restype = error_t
_dll.dc1394_feature_get_power.errcheck = _errcheck

_dll.dc1394_feature_set_power.argtypes = [PTR(camera_t), feature_t, switch_t]
_dll.dc1394_feature_set_power.restype = error_t
_dll.dc1394_feature_set_power.errcheck = _errcheck

# Gets the list of control modes for a feature (manual, auto, etc...)
_dll.dc1394_feature_get_modes.argtypes = (PTR(camera_t), feature_t, PTR(feature_modes_t))
_dll.dc1394_feature_get_modes.restype = error_t
_dll.dc1394_feature_get_modes.errcheck = _errcheck

# Current control modes for a feature: get/set
_dll.dc1394_feature_get_mode.argtypes = [PTR(camera_t), feature_t, PTR(feature_mode_t)]
_dll.dc1394_feature_get_mode.restype = error_t
_dll.dc1394_feature_get_mode.errcheck = _errcheck

_dll.dc1394_feature_set_mode.argtypes = [PTR(camera_t), feature_t, feature_mode_t]
_dll.dc1394_feature_set_mode.restype = error_t
_dll.dc1394_feature_set_mode.errcheck = _errcheck

# Tells whether a feature can be controlled in absolute mode
_dll.dc1394_feature_has_absolute_control.argtypes = [PTR(camera_t), feature_t, PTR(bool_t)]
_dll.dc1394_feature_has_absolute_control.restype = error_t
_dll.dc1394_feature_has_absolute_control.errcheck = _errcheck

# Gets the absolute boundaries of a feature
_dll.dc1394_feature_get_absolute_boundaries.argtypes = [PTR(camera_t), feature_t, PTR(c_float), PTR(c_float)]
_dll.dc1394_feature_get_absolute_boundariesrestype = error_t
_dll.dc1394_feature_get_absolute_boundaries.errcheck = _errcheck

# Absolute value of a feature: get/set
_dll.dc1394_feature_get_absolute_value.argtypes = [PTR(camera_t), feature_t, PTR(c_float)]
_dll.dc1394_feature_get_absolute_value.restype = error_t
_dll.dc1394_feature_get_absolute_value.errcheck = _errcheck

_dll.dc1394_feature_set_absolute_value.argtypes = [PTR(camera_t), feature_t, c_float]
_dll.dc1394_feature_set_absolute_value.restype = error_t
_dll.dc1394_feature_set_absolute_value.errcheck = _errcheck

# The status of absolute control of a feature(ON/OFF): get/set
_dll.dc1394_feature_get_absolute_control.argtypes = [PTR(camera_t), feature_t, PTR(switch_t)]
_dll.dc1394_feature_get_absolute_control.restype = error_t
_dll.dc1394_feature_get_absolute_control.errcheck = _errcheck

_dll.dc1394_feature_set_absolute_control.argtypes = [PTR(camera_t), feature_t, switch_t]
_dll.dc1394_feature_set_absolute_control.restype = error_t
_dll.dc1394_feature_set_absolute_control.errcheck = _errcheck

# ----------------------------------- Trigger ---------------------------------
# The polarity of the external trigger: get/set
_dll.dc1394_external_trigger_get_polarity.argtypes = [PTR(camera_t), PTR(trigger_polarity_t)]
_dll.dc1394_external_trigger_get_polarity.restype = error_t
_dll.dc1394_external_trigger_get_polarity.errcheck = _errcheck

_dll.dc1394_external_trigger_set_polarity.argtypes = [PTR(camera_t), trigger_polarity_t]
_dll.dc1394_external_trigger_set_polarity.restype = error_t
_dll.dc1394_external_trigger_set_polarity.errcheck = _errcheck

# Tells whether the external trigger can change its polarity or not
_dll.dc1394_external_trigger_has_polarity.argtypes = [PTR(camera_t), PTR(bool_t)]
_dll.dc1394_external_trigger_has_polarity.restype = error_t
_dll.dc1394_external_trigger_has_polarity.errcheck = _errcheck

# Switch between internal and external trigger
_dll.dc1394_external_trigger_set_power.argtypes = [PTR(camera_t), switch_t]
_dll.dc1394_external_trigger_set_power.restype = error_t
_dll.dc1394_external_trigger_set_power.errcheck = _errcheck

# Gets the status of the external trigger
_dll.dc1394_external_trigger_get_power.argtypes = [PTR(camera_t), PTR(switch_t)]
_dll.dc1394_external_trigger_get_power.restype = error_t
_dll.dc1394_external_trigger_get_power.errcheck = _errcheck

# External trigger mode: get/set
_dll.dc1394_external_trigger_get_mode.argtypes = [PTR(camera_t), PTR(trigger_mode_t)]
_dll.dc1394_external_trigger_get_mode.restype = error_t
_dll.dc1394_external_trigger_get_mode.errcheck = _errcheck

_dll.dc1394_external_trigger_set_mode.argtypes = [PTR(camera_t), trigger_mode_t]
_dll.dc1394_external_trigger_set_mode.restype = error_t
_dll.dc1394_external_trigger_set_mode.errcheck = _errcheck

# External trigger source: get/set
_dll.dc1394_external_trigger_get_source.argtypes = [PTR(camera_t), PTR(trigger_source_t)]
_dll.dc1394_external_trigger_get_source.restype = error_t
_dll.dc1394_external_trigger_get_source.errcheck = _errcheck

_dll.dc1394_external_trigger_set_source.argtypes = [PTR(camera_t), trigger_source_t]
_dll.dc1394_external_trigger_set_source.restype = error_t
_dll.dc1394_external_trigger_set_source.errcheck = _errcheck

# Gets the list of available external trigger source
_dll.dc1394_external_trigger_get_supported_sources.argtypes = [PTR(camera_t), PTR(trigger_sources_t)]
_dll.dc1394_external_trigger_get_supported_sources.restype = error_t
_dll.dc1394_external_trigger_get_supported_sources.errcheck = _errcheck

# Turn software trigger on or off
_dll.dc1394_software_trigger_set_power.argtypes = [PTR(camera_t), switch_t]
_dll.dc1394_software_trigger_set_power.restype = error_t
_dll.dc1394_software_trigger_set_power.errcheck = _errcheck

# Gets the state of software trigger
_dll.dc1394_software_trigger_get_power.argtypes = [PTR(camera_t), PTR(switch_t)]
_dll.dc1394_software_trigger_get_power.restype = error_t
_dll.dc1394_software_trigger_get_power.errcheck = _errcheck


# ------------------------ PIO, SIO and Strobe Functions ----------------------
# Sends a quadlet on the PIO (output)
_dll.dc1394_pio_set.argtypes = [PTR(camera_t), c_uint32]
_dll.dc1394_pio_set.restype = error_t
_dll.dc1394_pio_set.errcheck = _errcheck

# Gets the current quadlet at the PIO (input)p
_dll.dc1394_pio_get.argtypes = [PTR(camera_t), PTR(c_uint32)]
_dll.dc1394_pio_get.restype = error_t
_dll.dc1394_pio_get.errcheck = _errcheck

# ------ other functionalities ------
# reset a camera to factory default settings
_dll.dc1394_camera_reset.argtypes = [PTR(camera_t)]
_dll.dc1394_camera_reset.restype = error_t
_dll.dc1394_camera_reset.errcheck = _errcheck

# Turn a camera on or off
_dll.dc1394_camera_set_power.argtypes = [PTR(camera_t), switch_t]
_dll.dc1394_camera_set_power.restype = error_t
_dll.dc1394_camera_set_power.errcheck = _errcheck

# Download a camera setup from the memory
_dll.dc1394_memory_busy.argtypes = [PTR(camera_t), PTR(bool_t)]
_dll.dc1394_memory_busy.restype = error_t
_dll.dc1394_memory_busy.errcheck = _errcheck

# Uploads a camera setup in the memory
# Note that this operation can only be performed a certain number of
# times for a given camera, as it requires reprogramming of an EEPROM.
_dll.dc1394_memory_save.argtypes = [PTR(camera_t), c_uint32]
_dll.dc1394_memory_save.restype = error_t
_dll.dc1394_memory_save.errcheck = _errcheck

# Tells whether the writing of the camera setup in memory is finished or not
_dll.dc1394_memory_load.argtypes = [PTR(camera_t), c_uint32]
_dll.dc1394_memory_load.restype = error_t
_dll.dc1394_memory_load.errcheck = _errcheck

# --------------------------- Video functions: video.h ------------------------
# Gets a list of video modes supported by the camera.
_dll.dc1394_video_get_supported_modes.argtypes = [PTR(camera_t), PTR(video_modes_t)]
_dll.dc1394_video_get_supported_modes.restype = error_t
_dll.dc1394_video_get_supported_modes.errcheck = _errcheck

# Gets a list of supported video framerates for a given video mode.
# Only works with non-scalable formats.
_dll.dc1394_video_get_supported_framerates.argtypes = [PTR(camera_t), video_mode_t, PTR(framerates_t)]
_dll.dc1394_video_get_supported_framerates.restype = error_t
_dll.dc1394_video_get_supported_framerates.errcheck = _errcheck

# Gets the current framerate. This is meaningful only if
# the video mode is not scalable.
_dll.dc1394_video_get_framerate.argtypes = [PTR(camera_t), video_mode_t, PTR(framerates_t)]
_dll.dc1394_video_get_framerate.restype = error_t
_dll.dc1394_video_get_framerate.errcheck = _errcheck

# Gets the current framerate. This is meaningful only if
# the video mode is not scalable
_dll.dc1394_video_get_framerate.argtypes = [PTR(camera_t), PTR(framerate_t)]
_dll.dc1394_video_get_framerate.restype = error_t
_dll.dc1394_video_get_framerate.errcheck = _errcheck

# Sets the current framerate. This is meaningful only if
# the video mode is not scalable
_dll.dc1394_video_set_framerate.argtypes = [PTR(camera_t), framerate_t]
_dll.dc1394_video_set_framerate.restype = error_t
_dll.dc1394_video_set_framerate.errcheck = _errcheck

# Gets the current vide mode
_dll.dc1394_video_get_mode.argtypes = [PTR(camera_t), PTR(video_mode_t)]
_dll.dc1394_video_get_mode.restype = error_t
_dll.dc1394_video_get_mode.errcheck = _errcheck

# Sets the current vide mode
_dll.dc1394_video_set_mode.argtypes = [PTR(camera_t), video_mode_t]
_dll.dc1394_video_set_mode.restype = error_t
_dll.dc1394_video_set_mode.errcheck = _errcheck

# Gets the current operation mode
_dll.dc1394_video_get_operation_mode.argtypes = [PTR(camera_t), PTR(operation_mode_t)]
_dll.dc1394_video_get_operation_mode.restype = error_t
_dll.dc1394_video_get_operation_mode.errcheck = _errcheck

# Sets the current operation mode
_dll.dc1394_video_set_operation_mode.argtypes = [PTR(camera_t), operation_mode_t]
_dll.dc1394_video_set_operation_mode.restype = error_t
_dll.dc1394_video_set_operation_mode.errcheck = _errcheck

# Gets the current ISO speed
_dll.dc1394_video_get_iso_speed.argtypes = [PTR(camera_t), PTR(iso_speed_t)]
_dll.dc1394_video_get_iso_speed.restype = error_t
_dll.dc1394_video_get_iso_speed.errcheck = _errcheck

# Sets the current ISO speed. Speeds over 400Mbps require 1394B
_dll.dc1394_video_set_iso_speed.argtypes = [PTR(camera_t), iso_speed_t]
_dll.dc1394_video_set_iso_speed.restype = error_t
_dll.dc1394_video_set_iso_speed.errcheck = _errcheck

# Gets the current ISO channel
_dll.dc1394_video_get_iso_channel.argtypes = [PTR(camera_t), PTR(c_uint32)]
_dll.dc1394_video_get_iso_channel.restype = error_t
_dll.dc1394_video_get_iso_channel.errcheck = _errcheck

# Sets the current ISO channel
_dll.dc1394_video_set_iso_channel.argtypes = [PTR(camera_t), c_uint32]
_dll.dc1394_video_set_iso_channel.restype = error_t
_dll.dc1394_video_set_iso_channel.errcheck = _errcheck

# Gets the current data depth, in bits. Only meaningful for
# 16bpp video modes (RAW16, RGB48, MONO16,...)
_dll.dc1394_video_get_data_depth.argtypes = [PTR(camera_t), PTR(c_uint32)]
_dll.dc1394_video_get_data_depth.restype = error_t
_dll.dc1394_video_get_data_depth.errcheck = _errcheck

# Starts/stops the isochronous data transmission. In other words,
# use this to control the image flow
_dll.dc1394_video_set_transmission.argtypes = [PTR(camera_t), switch_t]
_dll.dc1394_video_set_transmission.restypes = error_t
_dll.dc1394_video_set_transmission.errcheck = _errcheck

# Gets the status of the video transmission
_dll.dc1394_video_get_transmission.argtypes = [PTR(camera_t), PTR(switch_t)]
_dll.dc1394_video_get_transmission.restype = error_t
_dll.dc1394_video_get_transmission.errcheck = _errcheck

# Turns one-shot mode on or off
_dll.dc1394_video_set_one_shot.argtype = [PTR(camera_t), switch_t]
_dll.dc1394_video_set_one_shot.restype = error_t
_dll.dc1394_video_set_one_shot.errcheck = _errcheck

# Gets the status of the one-shot mode
_dll.dc1394_video_get_one_shot.restype = error_t
_dll.dc1394_video_get_one_shot.argtypes = [PTR(camera_t), PTR(bool_t)]
_dll.dc1394_video_get_one_shot.errcheck = _errcheck

# Turns multishot mode on or off
_dll.dc1394_video_set_multi_shot.argtypes = [PTR(camera_t), c_uint32, switch_t]
_dll.dc1394_video_set_multi_shot.restype = error_t
_dll.dc1394_video_set_multi_shot.errcheck = _errcheck

# Gets the status of the multi-shot mode
_dll.dc1394_video_get_multi_shot.argtypes = [PTR(camera_t), PTR(bool_t), PTR(c_uint32)]
_dll.dc1394_video_get_multi_shot.restype = error_t
_dll.dc1394_video_get_multi_shot.errcheck = _errcheck

# Gets the bandwidth usage of a camera.
# This function returns the bandwidth that is used by the
# camera *IF* ISO was ON. The returned value is in bandwidth units.
# The 1394 bus has 4915 bandwidth units available per cycle. Each unit
# corresponds to the time it takes to send one quadlet at ISO speed S1600.
# The bandwidth usage at S400 is thus four times the number of quadlets per
# packet. Thanks to Krisitian Hogsberg for clarifying this.
_dll.dc1394_video_get_bandwidth_usage.argtypes = [PTR(camera_t), PTR(c_uint32)]
_dll.dc1394_video_get_bandwidth_usage.restype = error_t
_dll.dc1394_video_get_bandwidth_usage.errcheck = _errcheck


# ----------------------- Capture functions: capture.h ------------------------
# Setup the capture, using a ring buffer of a certain size (num_dma_buffers)
# and certain options (flags)
_dll.dc1394_capture_setup.argtypes = [PTR(camera_t), c_uint32, c_uint32]
_dll.dc1394_capture_setup.restype = error_t
_dll.dc1394_capture_setup.errcheck = _errcheck

# Stop the capture
_dll.dc1394_capture_stop.argtypes = [PTR(camera_t)]
_dll.dc1394_capture_stop.restype = error_t
_dll.dc1394_capture_stop.errcheck = _errcheck

# Gets a file descriptor to be used for select(). Must be called
# after dc1394_capture_setup()
# Error check can do nothing with this one;
# we also do not really need this, since we do not want to dump files
# from the C library.p
_dll.dc1394_capture_get_fileno.argtypes = [PTR(camera_t)]
_dll.dc1394_capture_get_fileno.restype = c_int

# Captures a video frame. The returned struct contains the image buffer,
# among others. This image buffer SHALL NOT be freed, as it represents an area
# in the memory that belongs to the system.
_dll.dc1394_capture_dequeue.argtypes = [PTR(camera_t), capture_policy_t, PTR(PTR(video_frame_t))]
_dll.dc1394_capture_dequeue.restype = error_t
_dll.dc1394_capture_dequeue.errcheck = _errcheck

# Returns a frame to the ring buffer once it has been used.
_dll.dc1394_capture_enqueue.argtypes = [PTR(camera_t), PTR(video_frame_t)]
_dll.dc1394_capture_enqueue.restype = error_t
_dll.dc1394_capture_enqueue.errcheck = _errcheck

# Returns DC1394_TRUE if the given frame (previously dequeued) has been
# detected to be corrupt (missing data, corrupted data, overrun buffer, etc.).
# Note that certain types of corruption may go undetected in which case
# DC1394_FALSE will be returned.  The ability to detect corruption also varies
# between platforms.  Note that corrupt frames still need to be enqueued with
# dc1394_capture_enqueue() when no longer needed by the user.
_dll.dc1394_capture_is_frame_corrupt.argtypes = [PTR(camera_t), PTR(video_frame_t)]
_dll.dc1394_capture_is_frame_corrupt.restype = bool_t

# TODO: capture_callback_t
# Set a callback if supported by the platform (OS X only for now).
# _dll.dc1394_capture_set_callback.argtypes = [PTR(camera_t), capture_callback_t, c_void_p]
# _dll.dc1394_capture_set_callback.restype = c_void


# ----------------------- Conversion functions: conversions.h -----------------
# ---- Conversion functions to YUV422, MONO8 and RGB8 ----
# Converts an image buffer to YUV422
# parameters: *src, *dest, width, height, byte_order, source_coding, bits
_dll.dc1394_convert_to_YUV422.argtypes = [PTR(c_uint8), PTR(c_uint8), c_uint32,
                                          c_uint32, c_uint32, color_coding_t, c_uint32]
_dll.dc1394_convert_to_YUV422.restype = error_t
_dll.dc1394_convert_to_YUV422.errcheck = _errcheck

# Converts an image buffer to MONO8
_dll.dc1394_convert_to_MONO8.argtypes = [PTR(c_uint8), PTR( c_uint8), c_uint32,
                                        c_uint32, c_uint32, color_coding_t, c_uint32]
_dll.dc1394_convert_to_MONO8.restype = error_t
_dll.dc1394_convert_to_MONO8.errcheck = _errcheck

# Converts an image buffer to RGB8
_dll.dc1394_convert_to_RGB8.argtypes = [PTR(c_uint8), PTR(c_uint8), c_uint32,
                                        c_uint32, c_uint32, color_coding_t, c_uint32 ]
_dll.dc1394_convert_to_RGB8.restype = error_t
_dll.dc1394_convert_to_RGB8.errcheck = _errcheck


# ---- Conversion functions for stereo images
# changes a 16bit stereo image (8bit/channel) into two 8bit images on top
# of each other
_dll.dc1394_deinterlace_stereo.argtypes = [PTR(c_uint8), PTR(c_uint8), c_uint32, c_uint32]
_dll.dc1394_deinterlace_stereo.restype = error_t
_dll.dc1394_deinterlace_stereo.errcheck = _errcheck


# Color conversion functions for cameras that can output raw Bayer pattern
# images(color codings DC1394_COLOR_CODING_RAW8 and DC1394_COLOR_CODING_RAW16).
#
# Credits and sources:
# - Nearest Neighbor: OpenCV library
# - Bilinear: OpenCV library
# - HQLinear: High-Quality Linear Interpolation For Demosaicing Of
#             Bayer-Patterned Color Images, by Henrique S. Malvar, Li-wei He,
#             and Ross Cutler, in Proceedings of the ICASSP'04 Conference.
# - Edge Sense II: Laroche, Claude A. "Apparatus and method for adaptively
#                  interpolating a full color image utilizing chrominance
#                  gradients" U.S. Patent 5,373,322. Based on the code found
#                  on the website http://www-ise.stanford.edu/~tingchen/
#                  Converted to C and adapted to all four elementary patterns.
# - Downsample: "Known to the Ancients"
# - Simple: Implemented from the information found in the manual of
#           Allied Vision Technologies (AVT) cameras.
# - VNG: Variable Number of Gradients, a method described in
#        http://www-ise.stanford.edu/~tingchen/algodep/vargra.html
#        Sources import from DCRAW by Frederic Devernay. DCRAW is a RAW
#        converter program by Dave Coffin. URL:
#        http://www.cybercom.net/~dcoffin/dcraw/
# - AHD: Adaptive Homogeneity-Directed Demosaicing Algorithm, by K. Hirakawa
#        and T.W. Parks, IEEE Transactions on Image Processing, Vol. 14, Nr. 3,
#        March 2005, pp. 360 - 369.

# Perform de-mosaicing on an 8-bit image buffer
# parameters: uint16_t *bayer, uint16_t *rgb, uint32_t width, uint32_t height,
# color_filter_t tile, bayer_method_t method
_dll.dc1394_bayer_decoding_8bit.argtypes = [PTR(c_uint8), PTR(c_uint8), c_uint32,
                                            c_uint32, color_filter_t, bayer_method_t]
_dll.dc1394_bayer_decoding_8bit.restype = error_t
_dll.dc1394_bayer_decoding_8bit.errcheck = _errcheck


# Perform de-mosaicing on an 16-bit image buffer
# parameters: uint16_t *bayer, uint16_t *rgb, uint32_t width, uint32_t height,
# color_filter_t tile, bayer_method_t method, uint32_t bits
_dll.dc1394_bayer_decoding_16bit.argtypes = [PTR(c_uint8), PTR(c_uint8), c_uint32,
                                             c_uint32, color_filter_t, bayer_method_t, c_uint32 ]
_dll.dc1394_bayer_decoding_16bit.restype = error_t
_dll.dc1394_bayer_decoding_16bit.errcheck = _errcheck

# ---- Frame based conversions ----
# Converts the format of a video frame.
# To set the format of the output, simply set the values of the corresponding
# fields in the output frame
# parameters: inframe and outframe
_dll.dc1394_convert_frames.argtypes = [PTR(video_frame_t), PTR(video_frame_t)]
_dll.dc1394_convert_frames.restype = error_t
_dll.dc1394_convert_frames.errcheck = _errcheck

# De-mosaicing of a Bayer-encoded video frame
# To set the format of the output, simply set the values of the corresponding
# fields in the output frame
_dll.dc1394_debayer_frames.argtypes = [PTR(video_frame_t), PTR(video_frame_t), bayer_method_t]
_dll.dc1394_debayer_frames.restype = error_t
_dll.dc1394_debayer_frames.errcheck = _errcheck

# De-interlacing of stereo data for cideo frames
# To set the format of the output, simply set the values of the corresponding
# fields in the output frame
_dll.dc1394_deinterlace_stereo_frames.argtypes = [PTR(video_frame_t), PTR(video_frame_t), stereo_method_t]
_dll.dc1394_deinterlace_stereo_frames.restype = error_t
_dll.dc1394_deinterlace_stereo_frames.errcheck = _errcheck


# ----------------------- Register functions: register.h ----------------------
# parameters: *camera, offset, *value, num_register
_dll.dc1394_get_registers.argtypes = [PTR(camera_t), c_uint64, PTR(c_uint32), c_uint32]
_dll.dc1394_get_registers.restype = error_t
_dll.dc1394_get_registers.errcheck = _errcheck

# _dll.dc1394_get_register.argtypes = [PTR(camera_t), c_uint64, PTR(c_uint32)]
# _dll.dc1394_get_register.restype = error_t
# _dll.dc1394_get_register.errcheck = _errcheck

_dll.dc1394_set_registers.argtypes = [PTR(camera_t), c_uint64, PTR(c_uint32), c_uint32]
_dll.dc1394_set_registers.restype = error_t
_dll.dc1394_set_registers.errcheck = _errcheck

# _dll.dc1394_set_register.argtypes = [PTR(camera_t), c_uint64, c_uint32]
# _dll.dc1394_set_register.restype = error_t
# _dll.dc1394_set_register.errcheck = _errcheck

_dll.dc1394_get_adv_control_registers.argtypes = [PTR(camera_t), c_uint64, PTR(c_uint32), c_uint32]
_dll.dc1394_get_adv_control_registers.restype = error_t
_dll.dc1394_get_adv_control_registers.errcheck = _errcheck

# _dll.dc1394_get_adv_control_register.argtypes = [PTR(camera_t), c_uint64, PTR(c_uint32)]
# _dll.dc1394_get_adv_control_register.restype = error_t
# _dll.dc1394_get_adv_control_register = _errcheck

_dll.dc1394_set_adv_control_registers.argtypes = [PTR(camera_t), c_uint64, PTR(c_uint32), c_uint32]
_dll.dc1394_set_adv_control_registers.restype = error_t
_dll.dc1394_set_adv_control_registers.errcheck = _errcheck

# _dll.dc1394_set_adv_control_register.argtypes = [PTR(camera_t), c_uint64, c_uint32]
# _dll.dc1394_set_adv_control_register.restype = error_t
# _dll.dc1394_set_adv_control_register.errcheck = _errcheck

# ---- Get/set format_7 registers ----
# parameters: *camera, mode, offset, *value:
_dll.dc1394_get_format7_register.argtypes = [PTR(camera_t), c_uint, c_uint64, PTR(c_uint32)]
_dll.dc1394_get_format7_register.restype = error_t
_dll.dc1394_get_format7_register.errcheck = _errcheck

_dll.dc1394_set_format7_register.argtypes = [PTR(camera_t), c_uint, c_uint64, c_uint32]
_dll.dc1394_set_format7_register.restype = error_t
_dll.dc1394_set_format7_register.errcheck = _errcheck

# ---- Get/set absolute control registers ----
# parameters *camera, feature, offset, *value
_dll.dc1394_get_absolute_register.argtypes = [PTR(camera_t), c_uint, c_uint64, PTR(c_uint32)]
_dll.dc1394_get_absolute_register.restype = error_t
_dll.dc1394_get_absolute_register.errcheck = _errcheck

_dll.dc1394_set_absolute_register.restype = error_t
_dll.dc1394_set_absolute_register.argtypes = [PTR(camera_t), c_uint, c_uint64, c_uint32]
_dll.dc1394_set_absolute_register.errcheck = _errcheck

# ---- Get/set PIO feature registers ----
# parameters: *camera, offset, *value
_dll.dc1394_get_PIO_register.argtypes = [PTR(camera_t), c_uint64, PTR(c_uint32)]
_dll.dc1394_get_PIO_register.restype = error_t
_dll.dc1394_get_PIO_register.errcheck = _errcheck

_dll.dc1394_set_PIO_register.argtypes = [PTR(camera_t), c_uint64, c_uint32]
_dll.dc1394_set_PIO_register.restype = error_t
_dll.dc1394_set_PIO_register.errcheck = _errcheck

# Get/Set Strobe Feature Registers
# parameters: *camera, offset, *value
_dll.dc1394_get_strobe_register.argtypes = [PTR(camera_t), c_uint64, PTR(c_uint32)]
_dll.dc1394_get_strobe_register.restype = error_t
_dll.dc1394_get_strobe_register.errcheck = _errcheck

_dll.dc1394_set_strobe_register.argtypes = [PTR(camera_t), c_uint64, c_uint32]
_dll.dc1394_set_strobe_register.restype = error_t
_dll.dc1394_set_strobe_register.errcheck = _errcheck


# ----------------- Format_7 (scalable image format) functions ----------------

# Gets the maximal image size for a given mode.
# parameters: *camera, video_mode, *h_size, *v_size:
_dll. dc1394_format7_get_max_image_size.argtypes = [PTR(camera_t), video_mode_t, PTR(c_uint32), PTR(c_uint32)]
_dll. dc1394_format7_get_max_image_size.restype = error_t
_dll. dc1394_format7_get_max_image_size.errcheck = _errcheck

# Gets the unit sizes for a given mode. The image size can only be a multiple
# of the unit size, and cannot be smaller than it.
# parameters: *camera, video_mode, *h_unit, *v_unit
_dll.dc1394_format7_get_unit_size.argtypes = [PTR(camera_t), video_mode_t, PTR(c_uint32), PTR(c_uint32)]
_dll.dc1394_format7_get_unit_size.restype = error_t
_dll.dc1394_format7_get_unit_size.errcheck = _errcheck

# Gets the current image size
_dll.dc1394_format7_get_image_size.argtypes = [PTR(camera_t), video_mode_t, PTR(c_uint32), PTR(c_uint32)]
_dll.dc1394_format7_get_image_size.restype = error_t
_dll.dc1394_format7_get_image_size.errcheck = _errcheck

# Sets the current image size
_dll.dc1394_format7_set_image_size.argtypes = [PTR(camera_t), video_mode_t, c_uint32, c_uint32]
_dll.dc1394_format7_set_image_size.restype = error_t
_dll.dc1394_format7_set_image_size.errcheck = _errcheck

# Gets the current image position
# parameters: *camera, video_mode, *left, *top
_dll.dc1394_format7_get_image_position.argtypes = [PTR(camera_t), video_mode_t, PTR(c_uint32), PTR(c_uint32)]
_dll.dc1394_format7_get_image_position.restype = error_t
_dll.dc1394_format7_get_image_position.errcheck = _errcheck

# Sets the current image position
_dll.dc1394_format7_set_image_position.argtypes = [PTR(camera_t), video_mode_t, c_uint32, c_uint32]
_dll.dc1394_format7_set_image_position.restype = error_t
_dll.dc1394_format7_set_image_position.errcheck = _errcheck

# Gets the unit positions for a given mode. The image position can
# only be a multiple of the unit position (zero is acceptable).
# parameters: *camera, video_mode, *h_unit, *v_unit
_dll.dc1394_format7_get_unit_position.argtypes = [PTR(camera_t), video_mode_t, PTR(c_uint32), PTR(c_uint32)]
_dll.dc1394_format7_get_unit_position.restype = error_t
_dll.dc1394_format7_get_unit_position.errcheck = _errcheck

# Gets the current color coding
_dll.dc1394_format7_get_color_coding.argtypes = [PTR(camera_t), video_mode_t, PTR(color_coding_t)]
_dll.dc1394_format7_get_color_coding.restype = error_t
_dll.dc1394_format7_get_color_coding.errcheck = _errcheck

# Gets the list of color codings available for this mode
_dll.dc1394_format7_get_color_codings.argtypes = [PTR(camera_t), video_mode_t, PTR(color_codings_t)]
_dll.dc1394_format7_get_color_codings.restype = error_t
_dll.dc1394_format7_get_color_codings.errcheck = _errcheck

# Sets the current color coding
_dll.dc1394_format7_set_color_coding.argtypes = [PTR(camera_t), video_mode_t, color_coding_t]
_dll.dc1394_format7_set_color_coding.restype = error_t
_dll.dc1394_format7_set_color_coding.errcheck = _errcheck

# Gets the current color filter
_dll.dc1394_format7_get_color_filter.argtypes = [PTR(camera_t), video_mode_t, PTR(color_filter_t)]
_dll.dc1394_format7_get_color_filter.restype = error_t
_dll.dc1394_format7_get_color_filter.errcheck = _errcheck

# Get the parameters of the packet size: its maximal size and its unit size.
# The packet size is always a multiple of the unit bytes and cannot be zero.
# parameters: *camera, video_mode, *unit_bytes, *max_bytes
_dll.dc1394_format7_get_packet_parameters.argtypes = [PTR(camera_t), video_mode_t, PTR(c_uint32), PTR(c_uint32)]
_dll.dc1394_format7_get_packet_parameters.restype = error_t
_dll.dc1394_format7_get_packet_parameters.errcheck = _errcheck

# Gets the current packet size
# parameters: *camera, video_mode, *packet_size
_dll.dc1394_format7_get_packet_size.argtypes = [PTR(camera_t), video_mode_t, PTR(c_uint32)]
_dll.dc1394_format7_get_packet_size.restype = error_t
_dll.dc1394_format7_get_packet_size.errcheck = _errcheck

# Sets the current packet size
_dll.dc1394_format7_set_packet_size.argtypes = [PTR(camera_t), video_mode_t, c_uint32]
_dll.dc1394_format7_set_packet_size.restype = error_t
_dll.dc1394_format7_set_packet_size.errcheck = _errcheck

# Gets the recommended packet size. Ignore if zero.
# parameters: &camera, video_mode, &packet size
_dll.dc1394_format7_get_recommended_packet_size.argtypes = [PTR(camera_t), video_mode_t, PTR(c_uint32)]
_dll.dc1394_format7_get_recommended_packet_size.restype = error_t
_dll.dc1394_format7_get_recommended_packet_size.errcheck = _errcheck

# Gets the number of packets per frame.
# parameters: &camera, video_mode, &packets per frame
_dll.dc1394_format7_get_packets_per_frame.argtypes = [PTR(camera_t), video_mode_t, PTR(c_uint32)]
_dll.dc1394_format7_get_packets_per_frame.restype = error_t
_dll.dc1394_format7_get_packets_per_frame.errcheck = _errcheck

# Gets the data depth (e.g. 12, 13, 14 bits/pixel)
# parameters: &camera, video_mode, &data_depth
_dll.dc1394_format7_get_data_depth.argtypes = [PTR(camera_t), video_mode_t, PTR(c_uint32)]
_dll.dc1394_format7_get_data_depth.restype = error_t
_dll.dc1394_format7_get_data_depth.errcheck = _errcheck

# Gets the frame interval in float format
# parameters: &camera, video_mode, &interval
_dll.dc1394_format7_get_frame_interval.argtypes = [PTR(camera_t), video_mode_t, PTR(c_float)]
_dll.dc1394_format7_get_frame_interval.restype = error_t
_dll.dc1394_format7_get_frame_interval.errcheck = _errcheck

# Gets the number of pixels per image frame
# parameters: &camera, video_mode, &pixnum
_dll.dc1394_format7_get_pixel_number.restype = error_t
_dll.dc1394_format7_get_pixel_number.argtypes = [PTR(camera_t), video_mode_t, PTR(c_uint32)]
_dll.dc1394_format7_get_pixel_number.errcheck = _errcheck

# Get the total number of bytes per frame. This includes padding
# (to reach an entire number of packets)
# parameters: &camera, video_mode, &total_bytes
_dll.dc1394_format7_get_total_bytes.argtypes = [PTR(camera_t), video_mode_t, PTR(c_uint64)]
_dll.dc1394_format7_get_total_bytes.restype = error_t
_dll.dc1394_format7_get_total_bytes.errcheck = _errcheck

# These functions get the properties of (one or all) format7 mode(s)
# Gets the properties of all Format_7 modes supported by the camera.
_dll.dc1394_format7_get_modeset.argtypes = [PTR(camera_t), PTR(format7modeset_t)]
_dll.dc1394_format7_get_modeset.restype = error_t
_dll.dc1394_format7_get_modeset.errcheck = _errcheck

# Gets the properties of a Format_7 mode
_dll.dc1394_format7_get_mode_info.argtypes = [PTR(camera_t), video_mode_t, PTR(format7mode_t)]
_dll.dc1394_format7_get_mode_info.restype = error_t
_dll.dc1394_format7_get_mode_info.errcheck = _errcheck

# Joint function that fully sets a certain ROI taking all parameters
# into account. Note that this function does not SWITCH to the video mode
# passed as argument, it mearly sets it
# parameters: &camera, video_mode, color_coding, packet_size,
#             left, top, width, height
_dll.dc1394_format7_set_roi.argtypes = [PTR(camera_t), video_mode_t,
                                        color_coding_t, c_int32, c_int32,
                                        c_int32, c_int32, c_int32]
_dll.dc1394_format7_set_roi.restype = error_t
_dll.dc1394_format7_set_roi.errcheck = _errcheck

_dll.dc1394_format7_get_roi.argtypes = [PTR(camera_t), video_mode_t,
                                        PTR(color_coding_t), PTR(c_uint32),
                                        PTR(c_uint32), PTR(c_uint32),
                                        PTR(c_uint32), PTR(c_uint32)]
_dll.dc1394_format7_get_roi.restype = error_t
_dll.dc1394_format7_get_roi.errcheck = _errcheck


# ----------------------------- utilities: utils.h ----------------------------

# Returns the image width and height (in pixels) corresponding to a video mode.
# Works for scalable and non-scalable video modes.
# parameters: &camera, video_mode, &width, &height
_dll.dc1394_get_image_size_from_video_mode.argtypes = [PTR(camera_t),
                                                       video_mode_t,
                                                       PTR(c_int32),
                                                       PTR(c_int32)]
_dll.dc1394_get_image_size_from_video_mode.restype = error_t
_dll.dc1394_get_image_size_from_video_mode.errcheck = _errcheck

# Returns the given framerate as a float
_dll.dc1394_framerate_as_float.argtypes = [framerate_t, PTR(c_float)]
_dll.dc1394_framerate_as_float.restype = error_t
_dll.dc1394_framerate_as_float.errcheck = _errcheck

# Returns the number of bits per pixel for a certain color coding. This is
# the size of the data sent on the bus, the effective data depth may vary.
# Example: RGB16 is 16, YUV411 is 8, YUV422 is 8
_dll.dc1394_get_color_coding_data_depth.argtypes = [color_coding_t, PTR(c_uint32)]
_dll.dc1394_get_color_coding_data_depth.restype = error_t
_dll.dc1394_get_color_coding_data_depth.errcheck = _errcheck

# Returns the bit-space used by a pixel. This is different from the data depth!
# For instance, RGB16 has a bit space of 48 bits, YUV422 is 16bits and
# YU411 is 12bits.
_dll.dc1394_get_color_coding_bit_size.argtypes = [color_coding_t, PTR(c_uint32)]
_dll.dc1394_get_color_coding_bit_size.restype = error_t
_dll.dc1394_get_color_coding_bit_size.errcheck = _errcheck

# Returns the color coding from the video mode. Works with scalable
# image formats too.
_dll.dc1394_get_color_coding_from_video_mode.argtypes = [PTR(camera_t),
                                                         video_mode_t,
                                                         PTR(color_coding_t)]
_dll.dc1394_get_color_coding_from_video_mode.restype = error_t
_dll.dc1394_get_color_coding_from_video_mode.errcheck = _errcheck

# Tells whether the color mode is color or monochrome
_dll.dc1394_is_color.argtypes = [color_coding_t, PTR(bool_t)]
_dll.dc1394_is_color.restype = error_t
_dll.dc1394_is_color.errcheck = _errcheck

# Tells whether the video mode is scalable or not.
_dll.dc1394_is_video_mode_scalable.argtypes = [video_mode_t]
_dll.dc1394_is_video_mode_scalable.restype = bool_t

# Tells whether the video mode is "still image" or not ("still image" is
# currently not supported by any cameras on the market)
_dll.dc1394_is_video_mode_still_image.argtypes = [video_mode_t]
_dll.dc1394_is_video_mode_still_image.restype = bool_t

# Tells whether two IDs refer to the same physical camera unit.
_dll.dc1394_is_same_camera.argtypes = [camera_id_t, camera_id_t]
_dll.dc1394_is_same_camera.restype = bool_t

# Returns a descriptive name for a feature
_dll.dc1394_feature_get_string.argtypes = [feature_t]
_dll.dc1394_feature_get_string.restype = c_char_p

# Returns a descriptive string for an error code
_dll.dc1394_error_get_string.argtypes = [error_t]
_dll.dc1394_error_get_string.restype = c_char_p

# Calculates the CRC16 checksum of a memory region. Useful to verify the CRC of
# an image buffer, for instance.
# parameters: &buffer, buffer_size
_dll.dc1394_checksum_crc16.argtypes = [PTR(c_uint8), c_uint32]
_dll.dc1394_checksum_crc16.restype = c_uint16


# -------------- ISO resources (channels and bandwidth)functions --------------

# dc1394_iso_set_persist
# param camera A camera handle.
# Calling this function will cause isochronous channel and bandwidth
# allocations to persist beyond the lifetime of this dc1394camera_t instance.
# Normally (when this function is not called), any allocations would be
# automatically released upon freeing this camera or a premature shutdown of
# the application (if possible).  For this function to be used, it
# must be called prior to any allocations or an error will be returned.
_dll.dc1394_iso_set_persist.argtypes = [PTR(camera_t)]
_dll.dc1394_iso_set_persist.restype = error_t
_dll.dc1394_iso_set_persist.errcheck = _errcheck

# dc1394_iso_allocate_channel:
# param &camera , channels_allowed, &channel
# channels_allowed: A bitmask of acceptable channels for the allocation.
# The LSB corresponds to channel 0 and the MSB corresponds to channe 63.
# Only channels whose bit is set will be considered for the allocation
# If \a channels_allowed = 0, the complete set of channels supported by this
# camera will be considered for the allocation.
# Allocates an isochronous channel.  This function may be called multiple
# times, each time allocating an additional channel.  The channel is
# automatically re-allocated if there is a bus reset.  The channel is
# automatically released when this dc1394camera_t is freed or if
# the application shuts down prematurely.  If the channel needs to persist
# beyond the lifetime of this application, call \a dc1394_iso_set_persist()
# first.  Note that this function does _NOT_ automatically program @a camera
# to use the allocated channel for isochronous streaming.
# You must do that manually using \a dc1394_video_set_iso_channel().
_dll.dc1394_iso_allocate_channel.argtypes = [PTR(camera_t), c_uint64, PTR(c_int)]
_dll.dc1394_iso_allocate_channel.restype = error_t
_dll.dc1394_iso_allocate_channel.errcheck = _errcheck

# dc1394_iso_release_channel:
# param &camera, channel_to_release
# Releases a previously allocated channel.  It is acceptable to release
# channels that were allocated by a different process or host.  If attempting
# to release a channel that is already released, the function will succeed.
_dll.dc1394_iso_release_channel.argtypes = [PTR(camera_t), c_int]
_dll.dc1394_iso_release_channel.restype = error_t
_dll.dc1394_iso_release_channel.errcheck = _errcheck

# dc1394_iso_allocate_bandwidth
# param &camera, bandwidth_units
# bandwidth_units: the number of isochronous bandwidth units to allocate
# Allocates isochronous bandwidth.  This functions allocates bandwidth in
# addition_ to any previous allocations.  It may be called multiple times.
# The bandwidth is automatically re-allocated if there is a bus reset.
# The bandwidth is automatically released if this camera is freed or the
# application shuts down prematurely.  If the bandwidth needs to persist
# beyond the lifetime of this application, call a
# dc1394_iso_set_persist() first.
_dll.dc1394_iso_allocate_bandwidth.argtypes = [PTR(camera_t), c_int]
_dll.dc1394_iso_allocate_bandwidth.restype = error_t
_dll.dc1394_iso_allocate_bandwidth.errcheck = _errcheck

# dc1394_iso_release_bandwidth:
# param &camera, bandwidth_units
# Releases previously allocated isochronous bandwidth.  Each \a dc1394camera_t
# keeps track of a running total of bandwidth that has been allocated.
# Released bandwidth is subtracted from this total for the sake of automatic
# re-allocation and automatic release on shutdown. It is also acceptable for a
# camera to release more bandwidth than it has allocated (to clean up for
# another process for example).  In this case, the running total of bandwidth
# is not affected. It is acceptable to release more bandwidth than is
# allocated in total for the bus.  In this case, all bandwidth is released and
# the function succeeds.
_dll.dc1394_iso_release_bandwidth.argtypes = [PTR(camera_t), c_int]
_dll.dc1394_iso_release_bandwidth.restype = error_t
_dll.dc1394_iso_release_bandwidth.errcheck = _errcheck

# dc1394_iso_release_all:
# Releases all channels and bandwidth that have been previously allocated for
# this dc1394camera_t.  Note that this information can only be tracked per
# process, and there is no knowledge of allocations for this camera by
# previous processes.  To release resources in such a case, the manual release
# functions \a dc1394_iso_release_channel() and a
# dc1394_iso_release_bandwidth() must be used.
_dll.dc1394_iso_release_all.argtypes = [PTR(camera_t)]
_dll.dc1394_iso_release_all.restype = error_t
_dll.dc1394_iso_release_all.errcheck = _errcheck


# ---------------------------- Log functions: log.h ---------------------------
# dc1394_log_register_handler: register log handler for reporting error,
# warning or debug statements. Passing NULL as argument turns off this log
# level.
# params: &log_handler, type_of_the_log, message_type, log_message
_dll.dc1394_log_register_handler.argtypes = [log_t, c_void_p, c_void_p]
_dll.dc1394_log_register_handler.restype = error_t
_dll.dc1394_log_register_handler.errcheck = _errcheck

# dc1394_log_set_default_handler: set the log handler to the default handler
# At boot time, debug logging is OFF (handler is NULL). Using this function
# for the debug statements will start logging of debug statements using the
# default handler.
_dll.dc1394_log_set_default_handler.argtypes = [log_t]
_dll.dc1394_log_set_default_handler.restype = error_t
_dll.dc1394_log_set_default_handler.errcheck = _errcheck

# dc1394_log_error: logs a fatal error condition to the registered facility
# This function shall be invoked if a fatal error condition is encountered.
# The message passed as argument is delivered to the registered error
# reporting function registered before.
# param [in] format,...: error message to be logged, multiple arguments
# allowed (printf style)
_dll.dc1394_log_error.restype = None
_dll.dc1394_log_error.argtypes = [c_char_p]

# dc1394_log_warning: logs a nonfatal error condition to the registered
# facility This function shall be invoked if a nonfatal error condition is
# encountered. The message passed as argument is delivered to the registered
# warning reporting function registered before.
_dll.dc1394_log_warning.restype = None
_dll.dc1394_log_warning.argtypes = [c_char_p]

# dc1394_log_debug: logs a debug statement to the registered facility
# This function shall be invoked if a debug statement is to be logged.
# The message passed as argument is delivered to the registered debug
# reporting function registered before ONLY IF the environment variable
# DC1394_DEBUG has been set before the program starts.
_dll.dc1394_log_debug.restype = None
_dll.dc1394_log_debug.argtypes = [c_char_p]

dll = _dll
