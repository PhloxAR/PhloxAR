# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from .core import *
from ctypes import c_int32, c_uint32, c_uint64, c_float, byref, pointer


__all__ = [
    'Mode', 'Format7', 'mode_map'
]


class Mode(object):
    """
    Video mode for a DC1394 camera.
    Do not instantiate this class directly. Instead use one of the modes
    in 'Camera.modes' or 'Camera.modes_dict' and assign it to 'Camera.mode'.
    """
    _mode_id = None
    _cam = None
    _color_coding = None
    _dtype = None

    def __init__(self, cam, mode_id):
        self._mode_id = mode_id
        self._cam = cam
        self._dtype_shape()

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self._mode_id == other.mode_id

    def _dtype_shape(self):
        """
        Data type and shape.
        """
        import numpy
        w = c_int32()
        h = c_int32()
        dll.dc1394_get_image_size_from_video_mode(self._cam, self._mode_id,
                                                  byref(w), byref(h))
        self._shape = (h.value, w.value)

        cc = color_coding_t()
        dll.dc1394_get_color_coding_from_video_mode(
                self._cam, self._mode_id, byref(cc))

        self._color_coding = color_codings[cc.value]

        self._dtype = '<u1'
        if '8' in self._color_coding:
            self._dtype = '>u1'
        elif '16' in self._color_coding:
            self._dtype = '>u2'
        elif 'YUV' in self._color_coding:
            print("Warning: YUV image format!")
            # the data depth is 8 bit in the buffer,
            # but 12 or 16 bit in a _color pixel.
            self._dtype = ">u1"
        else:
            print("Nonstandard image format: %s" % mode[-1])
            self._dtype = ">u1"

        if "RGB" in self._color_coding:
            self._shape.append(3)

    @property
    def mode_id(self):
        return self._mode_id

    @property
    def name(self):
        """
        A descriptive name for this mode.
        """
        return video_modes[self._mode_id]

    @property
    def framerate(self):
        """
        Allowed framerate if the camera is in this mode.
        """
        return video_modes[self._mode_id]

    @property
    def shape(self):
        """
        The size in pixels of frames acquired in this mode.
        """
        return self._shape

    @property
    def color_coding(self):
        """
        The type of _color coding of pixels.
        """
        return self._color_coding

    @property
    def scalable(self):
        """
        Is this video scalable?
        """
        return bool(dll.dc1394_is_video_mode_scalable(self._mode_id))

    @property
    def dtype(self):
        """
        The numpy data type of an image of this mode.
        """
        return self._dtype


class Exif(Mode):
    pass


class Format7(Mode):
    """
    Format7 modes are flexible modes that support:

    * acquiring and transferring only a subsection of the frame for
      faster acquisition: regio-of-interes (ROI)
    * binning the pixels of the sensor for faster acquisition and
      reduced readout noise. The binning strategy in the different
      Format7 modes is defined by the vendor.

    Many aspects of Format7 modes can be altered while an acquisition is
    in progress. A notable exception from this is the size of thepacket.

    Use 'max_image_size', 'unit_size', 'unit_position, 'color_codings',
    and 'data_depth' to obtain information about the mode and then set
    its parameters via the attributes 'image_size', 'image_position',
    'color_coding', and 'packet_size' or all of them via the 'roi'
    attribute or with a call to 'setup.
    All settings are sent to the hardware right away.
    """
    @property
    def frame_interval(self):
        """
        The current frame interval in this format7 mode in seconds.
        Read-only.
        Use the 'Camera.framerate' and 'Camera.shutter'
        features (if present) to influence the framerate.
        """
        fi = c_float()
        dll.dc1394_get_frame_interval(self._cam, self._mode_id, byref(fi))

        return fi.value

    @property
    def max_image_size(self):
        """
        The maximum size (horizontal and vertical) of the ROI in pixels.
        Read-only.
        """
        hsize = c_uint32()
        vsize = c_uint32()
        dll.dc1394_format7_get_max_image_size(self._cam, self._mode_id,
                                              byref(hsize), byref(vsize))
        return hsize.value, vsize.value

    @property
    def image_size(self):
        """
        The current size (horizontal and vertical) of the ROI in pixels.
        The image size can only be a multiple of the :attr:`unit_size`, and
        cannot be smaller than it.
        """
        hsize = c_uint32()
        vsize = c_uint32()
        dll.dc1394_format7_get_image_size(self._cam, self._mode_id,
                                          byref(hsize), byref(vsize))
        return hsize.value, vsize.value

    @image_size.setter
    def image_size(self, width, height):
        x = c_uint32()
        y = c_uint32()
        dll.dc1394_format7_get_image_position(self._cam, self._mode_id,
                                              byref(x), byref(y))
        return x.value, y.value

    @property
    def image_position(self):
        """
        The start position of the upper left corner of the ROI in
        pixels (horizontal and vertical).
        The image position can only be a multiple of the unit position
        (zero is acceptable).
        """
        x = c_uint32()
        y = c_uint32()
        dll.dc1394_format7_get_image_position(self._cam, self._mode_id,
                                              byref(x), byref(y))
        return x.value, y.value

    @image_position.setter
    def image_position(self, pos):
        x, y = pos
        dll.dc1394_format7_set_image_position(self._cam, self._mode_id, x, y)
    
    @property
    def color_codings(self):
        """
        Allowed _color codings in this mode. Read-only.
        """
        pos_codings = color_codings_t()
        dll.dc1394_format7_get_color_codings(self._cam, self._mode_id,
                                             byref(pos_codings))
        return [color_codings[i] for i in pos_codings.codings[:pos_codings.num]]

    @property
    def color_coding(self):
        """
        The current _color coding.
        """
        cc = color_coding_t()
        dll.dc1394_format7_get_color_coding(self._cam, self._mode_id, byref(cc))
        return color_codings[cc.value]

    @color_coding.setter
    def color_coding(self, color):
        code = color_codings[color]
        dll.dc1394_format7_set_color_coding(self._cam, self._mode_id, code)

    @property
    def unit_position(self):
        """
        Horizontal and vertical 'image_position' multiples.
        Read-only.
        """
        h_unit = c_uint32()
        v_unit = c_uint32()
        dll.dc1394_format7_get_unit_position(self._cam, self._mode_id,
                                             byref(h_unit), byref(v_unit))
        return h_unit.value, v_unit.value

    @property
    def unit_size(self):
        """
        Horizontal and vertical :attr:`image_size` multiples. Read-only.
        """
        h_unit = c_uint32()
        v_unit = c_uint32()
        dll.dc1394_format7_get_unit_size(self._cam, self._mode_id,
                                         byref(h_unit), byref(v_unit))
        return h_unit.value, v_unit.value

    @property
    def roi(self):
        """
        Get and set all Format7 parameters at once.
        The following definitions can be used to set ROI of Format7 in
        a simpler fashion:
        * QUERY_FROM_CAMERA (-1) will use the current value used by the
          camera,
        * USE_MAX_AVAIL will (-2) set the value to its maximum and
        * USE_RECOMMENDED (-3) can be used for the bytes-per-packet
          setting.
        """
        w, h, x, y = c_int32(), c_int32(), c_int32(), c_int32()
        cco, packet_size = color_coding_t(), c_int32()
        dll.dc1394_format7_get_roi(self._cam, self._mode_id, pointer(cco),
                                   byref(packet_size),
            byref(x), byref(y), byref(w), byref(h))
        return ((w.value, h.value), (x.value, y.value),
                color_codings[cco.value], packet_size.value)

    @roi.setter
    def roi(self, args):
        size, position, color, packet_size = args
        dll.dc1394_format7_set_roi(self._cam, self._mode_id,
                                   color_codings[color], packet_size,
                                   position[0], position[1], size[0], size[1])

    @property
    def dtype(self):
        self._dtype_shape()
        return self._dtype
    
    @property
    def shape(self):
        self._dtype_shape()
        return self._shape

    @property
    def recommended_packet_size(self):
        """
        Recommended number of bytes per packet. Read-only.
        """
        packet_size = c_uint32()
        dll.dc1394_format7_get_recommended_packet_size(self._cam, self._mode_id,
                                                       byref(packet_size))
        return packet_size.value
    
    @property
    def packet_parameters(self):
        """
        Maximum number and unit size of bytes per packet. Read-only.
        Get the parameters of the packet size: its maximal size and its
        unit size. The packet size is always a multiple of the unit
        bytes and cannot be zero.
        """
        packet_size_max = c_uint32()
        packet_size_unit = c_uint32()
        dll.dc1394_format7_get_packet_parameters(self._cam, self._mode_id,
                                                 byref(packet_size_unit),
                                                 byref(packet_size_max))
        return packet_size_unit.value, packet_size_max.value
    
    @property
    def packet_size(self):
        """
        Current number of bytes per packet.
        """
        packet_size = c_uint32()
        dll.dc1394_format7_get_packet_size(self._cam, self._mode_id,
                                           byref(packet_size))
        return packet_size.value

    @packet_size.setter
    def packet_size(self, packet_size):
        dll.dc1394_format7_set_packet_size(self._cam, self._mode_id,
                                           int(packet_size))

    @property
    def total_bytes(self):
        """
        Current total number of bytes per frame. Read-only.
        This includes padding (to reach an entire number of packets).
        Use :attr:`packet_size` to influence its value.
        """
        ppf = c_uint64()
        dll.dc1394_format7_get_total_bytes(self._cam, self._mode_id, byref(ppf))
        return ppf.value

    @property
    def pixel_number(self):
        """
        The number of pixels per frame. Read-only.
        """
        px = c_uint32()
        dll.dc1394_format7_get_pixel_number(self._cam, self._mode_id, byref(px))
        return px.value

    @property
    def data_depth(self):
        """
        The number of bits per pixel. Read-only.
        Need not be a multiple of 8.
        """
        dd = c_uint32()
        dll.dc1394_format7_get_data_depth(self._cam, self._mode_id, byref(dd))
        return dd.value

    def setup(self, image_size=(QUERY_FROM_CAMERA, QUERY_FROM_CAMERA),
              image_position=(QUERY_FROM_CAMERA, QUERY_FROM_CAMERA),
              color_coding=QUERY_FROM_CAMERA, packet_size=USE_RECOMMANDED):
        """
        Setup this Format7 mode.
        Similar to setting :attr:`roi` but size and positiogn are made
        multiples of :attr:`unit_size` and :attr:`unit_position`. All
        arguments are optional and default to not changing the current
        value. :attr:`packet_size` is set to the recommended value.
        """
        wu, hu = self.unit_size
        xu, yu = self.unit_position
        position = xu*int(image_position[0]/xu), yu*int(image_position[1]/yu)
        size = wu*int(image_size[0]/wu), hu*int(image_size[1]/hu)
        self.roi = size, position, color_coding, packet_size
        return self.roi


mode_map = {
    64: Mode,
    65: Mode,
    66: Mode,
    67: Mode,
    68: Mode,
    69: Mode,
    70: Mode,
    71: Mode,
    72: Mode,
    73: Mode,
    74: Mode,
    75: Mode,
    76: Mode,
    77: Mode,
    78: Mode,
    79: Mode,
    80: Mode,
    81: Mode,
    82: Mode,
    83: Mode,
    84: Mode,
    85: Mode,
    86: Mode,
    87: Exif,
    88: Format7,
    89: Format7,
    90: Format7,
    91: Format7,
    92: Format7,
    93: Format7,
    94: Format7,
    95: Format7,
}


def create_mode(cam, m):
    if isinstance(m, tuple):
        m = "%sx%s_%s" % m
    return Mode(cam, video_modes[m])
