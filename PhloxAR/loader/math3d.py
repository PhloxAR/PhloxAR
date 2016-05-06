#!/usr/bin/env python
#
# Copyright (c) 2006 Alex Holkner
# Alex.Holkner@mail.google.com
#
# Copyright (c) 2016 Matthias Y. Chen
#
# This library is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2.1 of the License, or (at your
# option) any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals
import math
import operator
import types


class Point2(object):
    pass


class Point3(object):
    pass


class Vector2(object):
    def __init__(self, x=0, y=0):
        self._x = x
        self._y = y

    def __copy__(self):
        return self.__class__(self._x, self._y)

    copy = __copy__

    def __repr__(self):
        return '<Vector2(%.2f, %.2f)>'% (self._x, self._y)

    def __eq__(self, other):
        if isinstance(other, Vector2):
            return self._x == other._x and self._y == other._y

    def __ne__(self, other):
        return not self.__eq__(other)

    def __nonzero__(self):
        return self._x != 0 or self._y != 0

    def __len__(self):
        return 2

    def __getitem__(self, key):
        return (self._x, self._y)[key]

    def __setitem__(self, key, value):
        l = [self._x, self._y]
        l[key] = value
        self._x, self._y = l

    def __iter__(self):
        return iter((self._x, self._y))

    def __getattr__(self, key):
        try:
            return tuple([(self._x, self._y)['xy'.index(c)] for c in key])
        except ValueError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if len(key):
            object.__setattr__(self, key, value)
        else:
            try:
                l = [self._x, self._y]
                for k, v in map(None, key, value):
                    l['xy'.index(k)] = v
                self._x, self._y = l
            except ValueError:
                raise AttributeError(key)

    def __add__(self, other):
        pass

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._x = val

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, val):
        self._y = val


class Vector3(object):
    pass


class Matrix3(object):
    pass


class Matrix4(object):
    pass


class Quaternion(object):
    pass


class Geometry(object):
    pass


class Line2(Geometry):
    pass


class Line3(object):
    pass


class Ray2(Line2):
    pass


class Ray3(Line3):
    pass


class LineSegment2(Line2):
    pass


class LineSegment3(Line3):
    pass


class Circle(Geometry):
    pass


class Sphere(object):
    pass


class Plane(object):
    pass