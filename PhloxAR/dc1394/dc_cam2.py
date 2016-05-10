# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from ctypes import byref, POINTER, c_uint32, c_int32, c_float

from .core import *
from .frame import *
from .dc_cam import DCError, DCCameraError


class Context(object):
    """
    The DC1394 context.
    Each application should maintain one of these, especially if it
    wants to access several cameras. But as the Camera objects
    will create a Context themselves if not supplied with one, it is not
    strictly necessary.
    Additionally, since the context needs to stay alive for the lifespan
    of the Camera objects, their health is enforced by the Camera objects.
    The available camera GUIDs can be obtained from the cameras
    list. To obtain a Camera object for a certain camera,
    either the camera method of a Context object
    can be used or the context can be passed to the Camera constructor.
    """
    _handle = None

    def __init__(self):
        self._handle = dll.dc1394_new()

    def __del__(self):
        self.close()

    def close(self):
        """
        Free the library and the dc 1394 context
        After calling this, all cameras in this context are invalid.
        """
        if self._handle is not None:
            dll.dc1394_free(self._handle)
        self._handle = None

    @property
    def cameras(self):
        """
        The list of cameras attached to the system. Read-only.
        Each item contains the GUID of the camera and the unit number.
        Pass a (GUID, unit) tuple of the list to camera_handle() to
        obtain a handle. Since a single camera can contain several
        functional units (think stereo cameras), the GUID is not enough
        to identify an IIDC camera.

        If present, multiple cards will be probed.
        """
        cam_list = POINTER(camera_list_t)()
        dll.dc1394_camera_enumerate(self._handle, byref(cam_list))
        cams = [(cam.guid, cam.uint) for cam in
                cam_list.contents.ids[:cam_list.contents.num]]
        dll.dc1394_free_list(cam_list)
        return cams

    def camera_handle(self, guid, unit=None):
        """
        Obtain a camera handle given the GUID and optionally the unit number
        of the camera.
        Pass this handle to Camera or to camera.
        DC1394Exception will be throw if the requested camera is inaccessible.
        """
        if unit is None:
            handle = dll.dc1394_camera_new(self._handle, guid)
        else:
            handle = dll.dc1394_camera_new_unit(self._handle, guid, unit)

        if not handle:
            raise DCError("Couldn't access camera ({}, {})!".format(guid, unit))

    def camera(self, guid, unit=None, **kwargs):
        """
        Obtain a Camera instance for a given camera GUID
        """
        handle = self.camera_handle(guid, unit)
        return DCCamera2(context=self, handle=handle, **kwargs)


class Feature(object):
    pass


class Trigger(Feature):
    pass


class WhiteBalance(Feature):
    pass


class Temperature(Feature):
    pass


class WhiteShading(Feature):
    pass


class Mode(object):
    def __init__(self, cam, mode_id):
        pass

    @property
    def mode_id(self):
        pass

    @property
    def name(self):
        pass

    @property
    def rates(self):
        pass

    @property
    def image_size(self):
        pass

    @property
    def color_coding(self):
        pass

    @property
    def scalable(self):
        pass

    @property
    def dtype(self):
        pass

    def __str__(self):
        pass


class Exif(Mode):
    pass


class Format7(Mode):
    @property
    def max_image_size(self):
        pass

    @property
    def image_size(self):
        pass

    @image_size.setter
    def image_size(self, value):
        pass

    @property
    def image_position(self):
        pass

    @image_position.setter
    def image_position(self, value):
        pass

    @property
    def color_codings(self):
        pass

    @property
    def color_coding(self):
        pass

    @color_coding.setter
    def color_coding(self, color):
        pass

    @property
    def unit_position(self):
        pass

    @property
    def unit_size(self):
        pass

    @property
    def roi(self):
        pass

    @roi.setter
    def roi(self, args):
        pass

    @property
    def recommanded_packet_size(self):
        pass

    @property
    def packet_params(self):
        pass

    @property
    def packet_size(self):
        pass

    @packet_size.setter
    def packet_size(self, size):
        pass

    @property
    def total_bytes(self):
        pass

    @property
    def data_depth(self):
        pass

    @property
    def pixel_number(self):
        pass

    def setup(self, image_size=(QUERY_FROM_CAMERA, QUERY_FROM_CAMERA),
              image_position=(QUERY_FROM_CAMERA, QUERY_FROM_CAMERA),
              color_coding=QUERY_FROM_CAMERA, packet_size=USE_RECOMMANDED):
        pass


class DCCamera2(object):
    _cam = None
    _context = None

    def __init__(self, guid, context=None, handle=None, iso_speed=None,
                 mode=None, rate=None, **features):
        pass

    def __del__(self):
        self.close()

    def close(self):
        pass

    def power(self, on=True):
        pass

    def reset_bus(self):
        pass

    def reset_camera(self):
        pass

    def memory_save(self, channel):
        pass

    def memory_load(self, channel):
        pass

    @property
    def memory_busy(self):
        pass

    def flush(self):
        pass

    def dequeue(self, poll=False):
        pass

    def start_capture(self, buf_size=4, capture_flags='DEFAULT'):
        pass

    def stop_capture(self):
        pass

    def start_video(self):
        pass

    def stop_video(self):
        pass

    def start_one_shot(self):
        pass

    def stop_one_shot(self):
        pass

    def start_multi_shot(self, n):
        pass

    def stop_multi_shot(self):
        pass

    @property
    def fileno(self):
        pass

    def _load_features(self):
        pass

    @property
    def features(self):
        pass

    def setup(self, active=True, mode='manual', absolute=True, **features):
        pass

    def _load_modes(self):
        pass

    @property
    def modes(self):
        pass

    @property
    def modes_dict(self):
        pass

    def get_register(self, offset):
        pass

    def set_register(self, offset, value):
        pass

    __getitem__ = get_register
    __setitem__ = set_register

    @property
    def broadcast(self):
        pass

    @broadcast.setter
    def broadcast(self, value):
        pass

    @property
    def model(self):
        pass

    @property
    def guid(self):
        pass

    @property
    def vendor(self):
        pass

    def __str__(self):
        pass

    @property
    def mode(self):
        pass

    @mode.setter
    def mode(self, mode):
        pass

    @property
    def rate(self):
        pass

    @rate.setter
    def rate(self, framerate):
        pass

    @property
    def iso_speed(self):
        pass

    @iso_speed.setter
    def iso_speed(self, speed):
        pass

    @property
    def operation_mode(self):
        pass

    @operation_mode.setter
    def operation_mode(self, mode):
        pass

    @property
    def iso_channel(self):
        pass

    @iso_channel.setter
    def iso_channel(self, channel):
        pass

    @property
    def data_depth(self):
        pass

    @property
    def bandwidth_usage(self):
        pass

    def get_strobe(self, offset):
        pass

    def set_strobe(self, offset, value):
        pass

    def __eq__(self, other):
        pass

    @property
    def node(self):
        pass


class ThreadedCamera(DCCamera2):
    def start(self, queue=0, mark_corrupt=True):
        pass

    def run(self):
        pass

    def next_image(self):
        pass

    def current_image(self, new=False):
        pass

    def stop(self):
        pass
