# -----------------------------------------------------------------------------
#
# -*- coding: utf-8 -*-
#
# phlox-libdc1394/phloxar-dc1394/frame.py
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
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from ctypes import ARRAY, c_byte
from numpy import ndarray
from .core import *

__all__ = ['Frame']


class Frame(ndarray):
    """
    A frame returned by the camera.
    All metadata are retained as attributes of the resulting image.
    """
    _cam = None
    _frame = None

    def __new__(cls, camera, frame):
        """
        Convert a phloxar-dc1394 frame into a Frame instance.
        :param camera:
        :param frame:
        :return:
        """
        dtype = ARRAY(c_byte, frame.contents.image_bytes)
        buf = dtype.from_address(frame.contents.image)
        width, height = frame.contents.size
        pixels = width * height
        endian = frame.contents.little_endian and '<' or '>'
        type_str = '%su%i' % (endian, frame.contents.image_bytes / pixels)
        img = ndarray.__new__(cls, shape=(height, width), dtype=type_str, buffer=buf)

        img.frame_id = frame.contents.id
        img.frames_behind = frame.contents.frames_behind
        img.position = frame.contents.position
        img.packet_size = frame.contents.packet_size
        img.packets_per_frame = frame.contents.packet_per_frame
        img.timestamp = frame.contents.timestamp
        img.video_mode = video_modes[frame.contents.video_mode]
        img.data_depth = frame.contents.data_depth
        img.color_coding = color_codings[frame.contents.color_coding]
        img.color_filter = frame.contents.color_filter
        img.yuv_byte_order = frame.contents.yuv_byte_order
        img.stride = frame.contents.stride
        # save camera and frame for enqueue()
        img._frame = frame
        img._cam = camera

        return img

    def __array_finalize__(self, img):
        """
            Finalize the new Image class array.
            If called with an image object, inherit the properties of that image.
            """
        if img is None:
            return
        # do not inherit _frame and _cam since we also get called on copy()
        # and should not hold references to the frame in this case
        for key in ["position", "color_coding", "color_filter",
                    "yuv_byte_order", "stride", "packet_size",
                    "packets_per_frame", "timestamp", "frames_behind",
                    "frame_id", "data_depth", "video_mode"]:
            setattr(self, key, getattr(img, key, None))

    def enqueue(self):
        """
            Returns a frame to the ring buffer once it has been used.
            This method is also called implicitly on ``del``.
            Only call this method on the original frame obtained from
            Camera.dequeue` and not on its views, new-from-templates or
            copies. Otherwise an AttributeError will be raised.
            """
        if not hasattr(self, "_frame"):  # or self.base is not None:
            raise AttributeError("can only enqueue the original frame")
        if self._frame is not None:
            dll.dc1394_capture_enqueue(self._cam, self._frame)
            self._frame = None
            self._cam = None

    # from contextlib iport closing
    # with closing(camera.dequeue()) as im:
    #   do stuff with im
    close = enqueue

    def __del__(self):
        try:
            self.enqueue()
        except AttributeError:
            pass

    @property
    def corrupt(self):
        """
        Whether this frame corrupt.

        Returns ``True`` if the given frame has been detected to be
        corrupt (missing data, corrupted data, overrun buffer, etc.) and
        ``False`` otherwise.

        .. note::
           Certain types of corruption may go undetected in which case
           ``False`` will be returned erroneously.  The ability to
           detect corruption also varies between platforms.

        .. note::
           Corrupt frames still need to be enqueued with `enqueue`
           when no longer needed by the user.
        """
        return bool(dll.dc1394_capture_is_frame_corrupt(self._cam, self._frame))

    def to_rgb(self):
        """
        Convert the image to an RGB image.

        Array shape is: (image.shape[0], image.shape[1], 3)
        Uses the dc1394_convert_to_RGB() function for the conversion.
        """
        res = ndarray(3 * self.size, dtype='u1')
        shape = self.shape
        inp = ndarray(shape=len(self.data), buffer=self.data, dtype='u1')
        dll.dc1394_convert_to_RGB8(inp, res, shape[1], shape[0],
                                   self.yuv_byte_order, self.color_coding,
                                   self.data_depth)
        res.shape = shape[0], shape[1], 3
        return res

    def to_mono8(self):
        """
        Convert he image to 8 bit gray scale.
        Uses the dc1394_convert_to_MONO8() function
        """
        res = ndarray(self.size, dtype='u1')
        shape = self.shape
        inp = ndarray(shape=len(self.data), buffer=self.data, dtype='u1')
        dll.dc1394_convert_to_MONO8(inp, res, shape[1], shape[0],
                                    self.yuv_byte_order, self.color_coding,
                                    self.data_depth)
        res.shape = shape
        return res

    def to_yuv422(self):
        """
        Convert he image to YUV422 color format.
        Uses the dc1394_convert_to_YUV422() function
        """
        res = ndarray(self.size, dtype='u1')
        shape = self.shape
        inp = ndarray(shape=len(self.data), buffer=self.data, dtype='u1')
        dll.dc1394_convert_to_YUV422(inp, res, shape[1], shape[0],
                                     self.yuv_byte_order, self.color_coding,
                                     self.data_depth)
        return ndarray(shape=shape, buffer=res.data, dtype='u2')
