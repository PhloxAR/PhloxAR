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


class Vector2(object):
    def __init__(self, x=0, y=0):
        self._x = x
        self._y = y

    def __repr__(self):
        return '<Vector2(%.2f, %.2f)>'% (self._x, self._y)

    def __eq__(self, other):
        if isinstance(other, Vector2):
            return self._x == other.x and self._y == other.y

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
        if isinstance(other, Vector2):
            if self.__class__ is other.__class__:
                _class = Vector2
            else:
                _class = Point2

            return _class(self._x + other.x, self._y + other.y)
        else:
            assert hasattr(other, '__len__') and len(other) == 2
            return Vector2(self._x + other[0], self._y + other[1])

    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, Vector2):
            self._x += other.x
            self._y += other.y
        else:
            self.x += other[0]
            self.y += other[1]

        return self

    def __sub__(self, other):
        if isinstance(object, Vector2):
            if self.__class__ is other.__class__:
                _class = Vector2
            else:
                _class = Point2
            return _class(self._x - other.x, self._y - other.y)
        else:
            assert hasattr(other, '__len__') and len(other) == 2
            return Vector2(self._x - other[0], self._y - other[1])

    def __rsub__(self, other):
        if isinstance(other, Vector2):
            return Vector2(other.x - self._x, other.y - self._y)

        else:
            assert hasattr(other, '__len__') and len(other) == 2
            return Vector2(other.x - self[0], other.y - self[1])

    def __mul__(self, other):
        assert type(other) in (int, long, float)
        return Vector2(self._x * other, self._y * other)

    __rmul__ = __mul__

    def __imul__(self, other):
        assert type(other) in (int, long, float)
        self._x *= other
        self._y *= other

        return self

    def __div__(self, other):
        assert type(other) in (int, long, float)
        return Vector2(operator.div(self._x, other), operator.div(self._y, other))

    def __rdiv__(self, other):
        assert type(other) in (int, long, float)
        return Vector2(operator.div(other, self._x), operator.div(other, self._y))

    def __floordiv__(self, other):
        assert type(other) in (int, long, float)
        return Vector2(operator.floordiv(self._x, other),
                       operator.floordiv(self._y, other))

    def __rfloordiv__(self, other):
        assert type(other) in (int, long, float)
        return Vector2(operator.floordiv(other, self._x),
                       operator.floordiv(other, self._y))

    def __truediv__(self, other):
        assert type(other) in (int, long, float)
        return Vector2(operator.truediv(self._x, other),
                       operator.truediv(self._y, other))

    def __rtruediv__(self, other):
        assert type(other) in (int, long, float)
        return Vector2(operator.truediv(other, self._x),
                       operator.truediv(other, self._y))

    def __neg__(self):
        return Vector2(-self._x, -self._y)

    @property
    def copy(self):
        return self.__class__(self._x, self._y)

    @property
    def magnitude(self):
        return math.sqrt(self._x**2 + self._y**2)

    def magnitude_squared(self):
        return self._x**2 + self._y**2

    def normalize(self):
        d = self.magnitude
        if d:
            self._x /= d
            self._y /= d
        return self

    def normalized(self):
        d = self.magnitude
        if d:
            return Vector2(self._x / d, self._y / d)
        return self.copy

    def dot(self, other):
        assert isinstance(other, Vector2)
        return self._x * other.x + self._y * other.y

    # to improve
    def cross(self, other):
        return Vector3(0, 0, self._x * other.y - self._y * other.x)

    def reflect(self, normal):
        # assume normal is normalized
        assert isinstance(normal, Vector2)
        d = 2 * (self.x * normal.x + self.y * normal.y)
        return Vector2(self.x - d * normal.x, self.y - d * normal.y)

    def angle(self, other):
        return math.acos(self.dot(other) / self.magnitude)

    def project(self, other):
        n = other.normalized()
        return self.dot(n) * n

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
    def __init__(self, x=0, y=0, z=0):
        self._x = x
        self._y = y
        self._z = z

    def __repr__(self):
        return '<Vector3 (%.2f, %.2f, %.2f)>' % (self._x, self._y, self._z)

    def __eq__(self, other):
        if isinstance(other, Vector3):
            return (self._x == other.x and self.y == other.y and
                    self.z == other.z)
        else:
            assert hasattr(other, '__len__') and len(other) == 3
            return (self.x == other[0] and self.y == other[1] and
                   self.z == other[2])

    def __ne__(self, other):
        return not self.__eq__(other)

    def __nonzero__(self):
        return self._x !=0 or self._y != 0 or self._z != 0

    def __len__(self):
        return 3

    def __getitem__(self, key):
        return (self._x, self._y, self._z)[key]

    def __setitem__(self, key, value):
        l = [self._x, self._y, self._z]
        l[key] = value
        self.x, self.y, self.z = l

    def __iter__(self):
        return iter((self._x, self._y, self._z))

    def __getattr__(self, item):
        try:
            return tuple([(self._x, self._y, self._z)['xyz'.index(c)] for c in item])
        except ValueError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        if len(key) == 1:
            object.__setattr__(self, key, value)
        else:
            try:
                l = [self._x, self._y, self._z]
                for k, v in map(None, key, value):
                    l['xyz'.index(k)] = v
                self._x, self._y, self._z = l
            except ValueError:
                raise AttributeError(key)

    def __add__(self, other):
        if isinstance(other, Vector3):
            if self.__class__ is other.__class__:
                _class = Vector3
            else:
                _class = Point3
            return _class(self._x + other.x, self._y + other.y, self._z + other.z)
        else:
            assert hasattr(other, '__len__') and len(other) == 3
        return Vector3(self._x + other[0], self._y + other[1], self._z + other[2])

    def __iadd__(self, other):
        if isinstance(other, Vector3):
            self._x += other.x
            self._y += other.y
            self._z += other.z
        else:
            self._x += other[0]
            self._y += other[1]
            self._z += other[2]
        return self

    def __sub__(self, other):
        if isinstance(other, Vector3):
            if self.__class__ is other.__class__:
                _class = Vector3
            else:
                _class = Point3
            return Vector3(self._x - other.x, self._y - other.y, self._z - other.z)
        else:
            assert hasattr(other, '__len__') and len(other) == 3
            return Vector3(self._x - other[0], self._y - other[1], self._z - other[2])

    def __rsub__(self, other):
        if isinstance(other, Vector3):
            return Vector3(other.x - self._x, other.y - self._y, other.z - self._z)
        else:
            assert hasattr(other, '__len__') and len(other) == 3
            return Vector3(other._x - self[0], other._y - self[1], other._z - self[2])

    def __mul__(self, other):
        if isinstance(other, Vector3):
            # TODO component-wise mul/div in-place and on Vector2; docs.
            if self.__class__ is Point3 or other.__class__ is Point3:
                _class = Point3
            else:
                _class = Vector3
            return _class(self._x * other.x, self._y * other.y, self._z * other.z)
        else:
            assert type(other) in (int, long, float)
            return Vector3(self._x * other,
                           self._y * other,
                           self._z * other)

    __rmul__ = __mul__

    def __imul__(self, other):
        assert type(other) in (int, long, float)
        self._x *= other
        self._y *= other
        self._z *= other
        return self

    def __div__(self, other):
        assert type(other) in (int, long, float)
        return Vector3(operator.div(self._x, other), operator.div(self._y, other),
                       operator.div(self._z, other))

    def __rdiv__(self, other):
        assert type(other) in (int, long, float)
        return Vector3(operator.div(other, self._x), operator.div(other, self._y),
                       operator.div(other, self._z))

    def __floordiv__(self, other):
        assert type(other) in (int, long, float)
        return Vector3(operator.floordiv(self._x, other),
                       operator.floordiv(self._y, other),
                       operator.floordiv(self._z, other))

    def __rfloordiv__(self, other):
        assert type(other) in (int, long, float)
        return Vector3(operator.floordiv(other, self._x),
                       operator.floordiv(other, self._y),
                       operator.floordiv(other, self._z))

    def __truediv__(self, other):
        assert type(other) in (int, long, float)
        return Vector3(operator.truediv(self._x, other),
                       operator.truediv(self._y, other),
                       operator.truediv(self._z, other))

    def __rtruediv__(self, other):
        assert type(other) in (int, long, float)
        return Vector3(operator.truediv(other, self._x),
                       operator.truediv(other, self._y),
                       operator.truediv(other, self._z))

    def __neg__(self):
        return Vector3(-self._x, -self._y, -self._z)

    @property
    def copy(self):
        return self.__class__(self._x, self._y, self._z)

    @property
    def magnitude(self):
        return math.sqrt(self._x ** 2 + self._y ** 2 + self._z ** 2)

    @property
    def magnitude_squared(self):
        return self._x ** 2 + self._y ** 2 + self._z ** 2

    def normalize(self):
        d = self.magnitude
        if d:
            self._x /= d
            self._y /= d
            self._z /=d
        return self

    def normalized(self):
        d = self.magnitude
        if d:
            Vector3(self._x / d, self._y / d, self._z / d)
        return self.copy

    def dot(self, other):
        assert isinstance(other, Vector3)
        return self._x * other.x + self._y * other.y + self._z * other.z

    def cross(self, other):
        assert isinstance(other, Vector3)
        return Vector3(self._y * other.z - self._z * other.y,
                       self._z * other.x - self._x * other.z,
                       self._x * other.y - self._y * other.x)

    def reflect(self, normal):
        # assume normal is normalized
        assert isinstance(normal, Vector3)
        d = 2 * (self.x * normal.x + self.y * normal.y + self.z * normal.z)
        return Vector3(self.x - d * normal.x,
                       self.y - d * normal.y,
                       self.z - d * normal.z)

    def rotate_around(self, axis, theta):
        """
        Return the vector rotated around axis through angle theta. Right hand rule applies"""

        # Adapted from equations published by Glenn Murray.
        # http://inside.mines.edu/~gmurray/ArbitraryAxisRotation/ArbitraryAxisRotation.html
        x, y, z = self._x, self._y, self._z
        u, v, w = axis.x, axis.y, axis.z

        # Extracted common factors for simplicity and efficiency
        r2 = u ** 2 + v ** 2 + w ** 2
        r = math.sqrt(r2)
        ct = math.cos(theta)
        st = math.sin(theta) / r
        dt = (u * x + v * y + w * z) * (1 - ct) / r2
        return Vector3((u * dt + x * ct + (-w * y + v * z) * st),
                       (v * dt + y * ct + (w * x - u * z) * st),
                       (w * dt + z * ct + (-v * x + u * y) * st))

    def angle(self, other):
        return math.acos(self.dot(other) / (self.magnitude * other.magnitude))

    def project(self, other):
        n = other.normalized()
        return self.dot(n) * n

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

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, val):
        self._z = val


class Matrix3(object):
    def __init__(self):
        pass

    def __repr__(self):
        pass

    def __getattr__(self, key):
        pass

    def __setitem__(self, key, value):
        pass

    def __mul__(self, other):
        pass

    def __imul__(self, other):
        pass

    @property
    def copy(self):
        pass

    @property
    def identity(self):
        pass

    def scale(self, x, y):
        pass

    def translate(self, x, y):
        pass

    def rotate(self, angle):
        pass

    @classmethod
    def new_identity(cls):
        pass

    @classmethod
    def new_scale(cls):
        pass

    @classmethod
    def new_translate(cls):
        pass

    @classmethod
    def new_rotate(cls):
        pass

    @property
    def determinant(self):
        pass

    @property
    def inverse(self):
        pass



class Matrix4(object):
    pass


class Quaternion(object):
    pass


class Geometry(object):
    pass


class Point2(Vector2, Geometry):
    pass


class Point3(object):
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

vec2 = Vector2
vec3 = Vector3
mat3 = Matrix3
mat4 = Matrix4
ray2 = Ray2
ray3 = Ray3