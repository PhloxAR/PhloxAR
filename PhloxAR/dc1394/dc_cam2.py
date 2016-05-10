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
