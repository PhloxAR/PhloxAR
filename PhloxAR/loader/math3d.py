#!/usr/bin/env python
#
# Copyright (c) 2006 alex Holkner
# alex.Holkner@mail.google.com
#
# Copyright (c) 2016 Matthias Y. Chen
#
# This library is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2.1 of the License, or (at your
# option) any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTabILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, boston, MA  02110-1301 USA

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

        # adapted from equations published by Glenn Murray.
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
    """
    a b c
    e f g
    i j k
    """
    a = b = c = e = f = g = i = j = k = 0

    def __init__(self):
        self.a = self.f = self.k = 1
        self.b = self.c = self.e = self.g = self.i = self.j = 0

    def __repr__(self):
        return ('<Matrix3>\n[% 8.2f % 8.2f % 8.2f\n % 8.2f % 8.2f % 8.2f\n'
                ' % 8.2f % 8.2f % 8.2f]') % (self.a, self.b, self.c, self.e,
                                             self.f, self.g, self.i, self.j,
                                             self.k)

    def __getitem__(self, key):
        return [self.a, self.e, self.i,
                self.b, self.f, self.j,
                self.c, self.g, self.k][key]

    def __setitem__(self, key, value):
        l = self[:]
        l[key] = value
        (self.a, self.e, self.i, self.b, self.f, self.j,
         self.c, self.g, self.k) = l

    def __mul__(self, other):
        if isinstance(other, Matrix3):
            # Caching repeatedly accessed attributes in local variables
            # apparently increases performance by 20%.  Attrib: Will McGugan.
            aa = self.a
            ab = self.b
            ac = self.c
            ae = self.e
            af = self.f
            ag = self.g
            ai = self.i
            aj = self.j
            ak = self.k
            ba = other.a
            bb = other.b
            bc = other.c
            be = other.e
            bf = other.f
            bg = other.g
            bi = other.i
            bj = other.j
            bk = other.k
            c = Matrix3()
            c.a = aa * ba + ab * be + ac * bi
            c.b = aa * bb + ab * bf + ac * bj
            c.c = aa * bc + ab * bg + ac * bk
            c.e = ae * ba + af * be + ag * bi
            c.f = ae * bb + af * bf + ag * bj
            c.g = ae * bc + af * bg + ag * bk
            c.i = ai * ba + aj * be + ak * bi
            c.j = ai * bb + aj * bf + ak * bj
            c.k = ai * bc + aj * bg + ak * bk
            return c
        elif isinstance(other, Point2):
            a = self
            b = other
            p = Point2(0, 0)
            p.x = a.a * b.x + a.b * b.y + a.c
            p.y = a.e * b.x + a.f * b.y + a.g
            return p
        elif isinstance(other, Vector2):
            a = self
            b = other
            v = Vector2(0, 0)
            v.x = a.a * b.x + a.b * b.y
            v.y = a.e * b.x + a.f * b.y
            return v
        else:
            other = other.copy()
            other._apply_transform(self)
            return other

    def __imul__(self, other):
        assert isinstance(other, Matrix3)
        # Cache attributes in local vars (see Matrix3.__mul__).
        aa = self.a
        ab = self.b
        ac = self.c
        ae = self.e
        af = self.f
        ag = self.g
        ai = self.i
        aj = self.j
        ak = self.k
        ba = other.a
        bb = other.b
        bc = other.c
        be = other.e
        bf = other.f
        bg = other.g
        bi = other.i
        bj = other.j
        bk = other.k
        self.a = aa * ba + ab * be + ac * bi
        self.b = aa * bb + ab * bf + ac * bj
        self.c = aa * bc + ab * bg + ac * bk
        self.e = ae * ba + af * be + ag * bi
        self.f = ae * bb + af * bf + ag * bj
        self.g = ae * bc + af * bg + ag * bk
        self.i = ai * ba + aj * be + ak * bi
        self.j = ai * bb + aj * bf + ak * bj
        self.k = ai * bc + aj * bg + ak * bk
        return self

    @property
    def copy(self):
        m = Matrix3()
        m.a = self.a
        m.b = self.b
        m.c = self.c
        m.e = self.e
        m.f = self.f
        m.g = self.g
        m.i = self.i
        m.j = self.j
        m.k = self.k

        return m

    def identity(self):
        self.a = self.f = self.k = 1
        self.b = self.c = self.e = self.g = self.i = self.j = 0
        return self

    def scale(self, x, y):
        self *= Matrix3.new_scale(x, y)
        return self

    def translate(self, x, y):
        self *= Matrix3.new_translate(x, y)
        return self

    def rotate(self, angle):
        self *= Matrix3.new_rotate(angle)
        return self

    @classmethod
    def new_identity(cls):
        return cls()

    @classmethod
    def new_scale(cls, x, y):
        c = cls()
        c.a = x
        c.f = y
        return c

    @classmethod
    def new_translate(cls, x, y):
        c = cls()
        c.c = x
        c.g = y
        return c

    @classmethod
    def new_rotate(cls, angle):
        cl = cls()
        s = math.sin(angle)
        c = math.cos(angle)

        cl.a = cl.f = c
        cl.b = -s
        cl.e = s

        return cl

    @property
    def determinant(self):
        return (self.a * self.f * self.k + self.b * self.g * self.i +
                self.c * self.e * self.j - self.a * self.g * self.j -
                self.b * self.e * self.k - self.c * self.f * self.i)

    @property
    def inverse(self):
        tmp = Matrix3()
        d = self.determinant

        if abs(d) < 0.001:
            # No inverse, return identity
            return tmp
        else:
            d = 1.0 / d

            tmp.a = d * (self.f * self.k - self.g * self.j)
            tmp.b = d * (self.c * self.j - self.b * self.k)
            tmp.c = d * (self.b * self.g - self.c * self.f)
            tmp.e = d * (self.g * self.i - self.e * self.k)
            tmp.f = d * (self.a * self.k - self.c * self.i)
            tmp.g = d * (self.c * self.e - self.a * self.g)
            tmp.i = d * (self.e * self.j - self.f * self.i)
            tmp.j = d * (self.b * self.i - self.a * self.j)
            tmp.k = d * (self.a * self.f - self.b * self.e)

            return tmp


class Matrix4(object):
    """
    a b c d
    e f g h
    i j k l
    m n o p
    """
    a = b = c = d = 0
    e = f = g = h = 0
    i = j = k = l = 0
    m = n = o = p = 0

    def __init__(self):
        self.a = self.f = self.k = self.p = 1.
        self.b = self.c = self.d = self.e = 0
        self.g = self.h = self.i = self.j = 0
        self.l = self.m = self.n = self.o = 0

    def __repr__(self):
        return ('<Matrix4>\n'
                '% 8.2f % 8.2f % 8.2f % 8.2f\n'
                '% 8.2f % 8.2f % 8.2f % 8.2f\n'
                '% 8.2f % 8.2f % 8.2f % 8.2f\n'
                '% 8.2f % 8.2f % 8.2f % 8.2f]') % (self.a, self.b, self.c, self.d,
                                                   self.e, self.f, self.g, self.h,
                                                   self.i, self.j, self.k, self.l,
                                                   self.m, self.n, self.o, self.p)

    def __getitem__(self, key):
        return [self.a, self.e, self.i, self.m,
                self.b, self.f, self.j, self.n,
                self.c, self.g, self.k, self.o,
                self.d, self.h, self.l, self.p][key]

    def __setitem__(self, key, value):
        l = self[:]
        l[key] = value
        (self.a, self.e, self.i, self.m,
         self.b, self.f, self.j, self.n,
         self.c, self.g, self.k, self.o,
         self.d, self.h, self.l, self.p) = l

    def __mul__(self, other):
        if isinstance(other, Matrix4):
            # Cache attributes in local vars (see Matrix3.__mul__).
            aa = self.a
            ab = self.b
            ac = self.c
            ad = self.d
            ae = self.e
            af = self.f
            ag = self.g
            ah = self.h
            ai = self.i
            aj = self.j
            ak = self.k
            al = self.l
            am = self.m
            an = self.n
            ao = self.o
            ap = self.p
            ba = other.a
            bb = other.b
            bc = other.c
            bd = other.d
            be = other.e
            bf = other.f
            bg = other.g
            bh = other.h
            bi = other.i
            bj = other.j
            bk = other.k
            bl = other.l
            bm = other.m
            bn = other.n
            bo = other.o
            bp = other.p
            c = Matrix4()
            c.a = aa * ba + ab * be + ac * bi + ad * bm
            c.b = aa * bb + ab * bf + ac * bj + ad * bn
            c.c = aa * bc + ab * bg + ac * bk + ad * bo
            c.d = aa * bd + ab * bh + ac * bl + ad * bp
            c.e = ae * ba + af * be + ag * bi + ah * bm
            c.f = ae * bb + af * bf + ag * bj + ah * bn
            c.g = ae * bc + af * bg + ag * bk + ah * bo
            c.h = ae * bd + af * bh + ag * bl + ah * bp
            c.i = ai * ba + aj * be + ak * bi + al * bm
            c.j = ai * bb + aj * bf + ak * bj + al * bn
            c.k = ai * bc + aj * bg + ak * bk + al * bo
            c.l = ai * bd + aj * bh + ak * bl + al * bp
            c.m = am * ba + an * be + ao * bi + ap * bm
            c.n = am * bb + an * bf + ao * bj + ap * bn
            c.o = am * bc + an * bg + ao * bk + ap * bo
            c.p = am * bd + an * bh + ao * bl + ap * bp
            return c
        elif isinstance(other, Point3):
            a = self
            b = other
            p = Point3(0, 0, 0)
            p.x = a.a * b.x + a.b * b.y + a.c * b.z + a.d
            p.y = a.e * b.x + a.f * b.y + a.g * b.z + a.h
            p.z = a.i * b.x + a.j * b.y + a.k * b.z + a.l
            return p
        elif isinstance(other, Vector3):
            a = self
            b = other
            v = Vector3(0, 0, 0)
            v.x = a.a * b.x + a.b * b.y + a.c * b.z
            v.y = a.e * b.x + a.f * b.y + a.g * b.z
            v.z = a.i * b.x + a.j * b.y + a.k * b.z
            return v
        else:
            other = other.copy()
            other._apply_transform(self)
            return other

    def __imul__(self, other):
        aa = self.a
        ab = self.b
        ac = self.c
        ad = self.d
        ae = self.e
        af = self.f
        ag = self.g
        ah = self.h
        ai = self.i
        aj = self.j
        ak = self.k
        al = self.l
        am = self.m
        an = self.n
        ao = self.o
        ap = self.p
        ba = other.a
        bb = other.b
        bc = other.c
        bd = other.d
        be = other.e
        bf = other.f
        bg = other.g
        bh = other.h
        bi = other.i
        bj = other.j
        bk = other.k
        bl = other.l
        bm = other.m
        bn = other.n
        bo = other.o
        bp = other.p
        self.a = aa * ba + ab * be + ac * bi + ad * bm
        self.b = aa * bb + ab * bf + ac * bj + ad * bn
        self.c = aa * bc + ab * bg + ac * bk + ad * bo
        self.d = aa * bd + ab * bh + ac * bl + ad * bp
        self.e = ae * ba + af * be + ag * bi + ah * bm
        self.f = ae * bb + af * bf + ag * bj + ah * bn
        self.g = ae * bc + af * bg + ag * bk + ah * bo
        self.h = ae * bd + af * bh + ag * bl + ah * bp
        self.i = ai * ba + aj * be + ak * bi + al * bm
        self.j = ai * bb + aj * bf + ak * bj + al * bn
        self.k = ai * bc + aj * bg + ak * bk + al * bo
        self.l = ai * bd + aj * bh + ak * bl + al * bp
        self.m = am * ba + an * be + ao * bi + ap * bm
        self.n = am * bb + an * bf + ao * bj + ap * bn
        self.o = am * bc + an * bg + ao * bk + ap * bo
        self.p = am * bd + an * bh + ao * bl + ap * bp

    @property
    def copy(self):
        m = Matrix4()
        m.a = self.a
        m.b = self.b
        m.c = self.c
        m.d = self.d
        m.e = self.e
        m.f = self.f
        m.g = self.g
        m.h = self.h
        m.i = self.i
        m.j = self.j
        m.k = self.k
        m.l = self.l
        m.m = self.m
        m.n = self.n
        m.o = self.o
        m.p = self.p
        return m

    def transform(self, other):
        a = self
        b = other
        p = Point3(0, 0, 0)
        p.x = a.a * b.x + a.b * b.y + a.c * b.z + a.d
        p.y = a.e * b.x + a.f * b.y + a.g * b.z + a.h
        p.z = a.i * b.x + a.j * b.y + a.k * b.z + a.l
        w = a.m * b.x + a.n * b.y + a.o * b.z + a.p
        if w != 0:
            p.x /= w
            p.y /= w
            p.z /= w
        return p

    def identity(self):
        self.a = self.f = self.k = self.p = 1.
        self.b = self.c = self.d = self.e = 0
        self.g = self.h = self.i = self.j = 0
        self.l = self.m = self.n = self.o = 0

        return self

    def scale(self, x, y, z):
        self *= Matrix4.new_scale(x, y, z)
        return self

    def translate(self, x, y, z):
        self *= Matrix4.new_translate(x, y, z)
        return self

    def rotate_x(self, angle):
        self *= Matrix4.new_rotate_x(angle)
        return self

    def rotate_y(self, angle):
        self *= Matrix4.new_rotate_y(angle)
        return self

    def rotate_z(self, angle):
        self *= Matrix4.new_rotate_z(angle)
        return self

    def rotate_axis(self, angle, axis):
        self *= Matrix4.new_rotate_axis(angle, axis)
        return self

    def rotate_euler(self, heading, attitude, bank):
        self *= Matrix4.new_rotate_euler(heading, attitude, bank)
        return self

    def rotate_triple_axis(self, x, y, z):
        self *= Matrix4.new_rotate_triple_axis(x, y, z)
        return self

    def transpose(self):
        (self.a, self.e, self.i, self.m,
         self.b, self.f, self.j, self.n,
         self.c, self.g, self.k, self.o,
         self.d, self.h, self.l, self.p) = \
            (self.a, self.b, self.c, self.d,
             self.e, self.f, self.g, self.h,
             self.i, self.j, self.k, self.l,
             self.m, self.n, self.o, self.p)

    def transposed(self):
        m = self.copy
        m.transpose()
        return m

    @classmethod
    def new(cls, *vals):
        m = cls()
        m[:] = vals
        return m

    @classmethod
    def new_identity(cls):
        c = cls()
        return c

    @classmethod
    def new_scale(cls, x, y, z):
        c = cls()
        c.a = x
        c.f = y
        c.k = z
        return c

    @classmethod
    def new_translate(cls, x, y, z):
        c = cls()
        c.d = x
        c.h = y
        c.l = z
        return c

    @classmethod
    def new_rotate_x(cls, angle):
        cl = cls()
        s = math.sin(angle)
        c = math.cos(angle)
        cl.f = cl.k = c
        cl.g = -s
        cl.j = s
        return cl

    @classmethod
    def new_rotate_y(cls, angle):
        cl = cls()
        s = math.sin(angle)
        c = math.cos(angle)
        cl.a = cl.k = c
        cl.c = s
        cl.i = -s
        return cl

    @classmethod
    def new_rotate_z(cls, angle):
        cl = cls()

        s = math.sin(angle)
        c = math.cos(angle)
        cl.a = cl.f = c
        cl.b = -s
        cl.e = s

        return cl

    @classmethod
    def new_rotate_axis(cls, angle, axis):
        assert (isinstance(axis, Vector3))
        vector = axis.normalized()
        x = vector.x
        y = vector.y
        z = vector.z

        cl = cls()
        s = math.sin(angle)
        c = math.cos(angle)
        c1 = 1. - c

        # from the glRotate man page
        cl.a = x * x * c1 + c
        cl.b = x * y * c1 - z * s
        cl.c = x * z * c1 + y * s
        cl.e = y * x * c1 + z * s
        cl.f = y * y * c1 + c
        cl.g = y * z * c1 - x * s
        cl.i = x * z * c1 - y * s
        cl.j = y * z * c1 + x * s
        cl.k = z * z * c1 + c
        return cl

    @classmethod
    def new_rotate_euler(cls, heading, attitude, bank):
        # from http://www.euclideanspace.com/
        ch = math.cos(heading)
        sh = math.sin(heading)
        ca = math.cos(attitude)
        sa = math.sin(attitude)
        cb = math.cos(bank)
        sb = math.sin(bank)

        cl = cls()
        cl.a = ch * ca
        cl.b = sh * sb - ch * sa * cb
        cl.c = ch * sa * sb + sh * cb
        cl.e = sa
        cl.f = ca * cb
        cl.g = -ca * sb
        cl.i = -sh * ca
        cl.j = sh * sa * cb + ch * sb
        cl.k = -sh * sa * sb + ch * cb
        return cl

    @classmethod
    def new_rotate_triple_axis(cls, x, y, z):
        m = cls()

        m.a, m.b, m.c = x.x, y.x, z.x
        m.e, m.f, m.g = x.y, y.y, z.y
        m.i, m.j, m.k = x.z, y.z, z.z

        return m

    @classmethod
    def new_look_at(cls, eye, at, up):
        z = (eye - at).normalized()
        x = up.cross(z).normalized()
        y = z.cross(x)

        m = cls.new_rotate_triple_axis(x, y, z)
        m.d, m.h, m.l = eye.x, eye.y, eye.z
        return m

    @classmethod
    def new_perspective(cls, fov_y, aspect, near, far):
        # from the gluPerspective man page
        f = 1 / math.tan(fov_y / 2)
        m = cls()
        assert near != 0.0 and near != far
        m.a = f / aspect
        m.f = f
        m.k = (far + near) / (near - far)
        m.l = 2 * far * near / (near - far)
        m.o = -1
        m.p = 0
        return m

    @property
    def determinant(self):
        return ((self.a * self.f - self.e * self.b)
                * (self.k * self.p - self.o * self.l)
                - (self.a * self.j - self.i * self.b)
                * (self.g * self.p - self.o * self.h)
                + (self.a * self.n - self.m * self.b)
                * (self.g * self.l - self.k * self.h)
                + (self.e * self.j - self.i * self.f)
                * (self.c * self.p - self.o * self.d)
                - (self.e * self.n - self.m * self.f)
                * (self.c * self.l - self.k * self.d)
                + (self.i * self.n - self.m * self.j)
                * (self.c * self.h - self.g * self.d))

    @property
    def inverse(self):
        tmp = Matrix4()
        d = self.determinant

        if abs(d) < 0.001:
            # No inverse, return identity
            return tmp
        else:
            d = 1.0 / d

            tmp.a = d * (self.f * (self.k * self.p - self.o * self.l) +
                         self.j * (self.o * self.h - self.g * self.p) +
                         self.n * (self.g * self.l - self.k * self.h))
            tmp.e = d * (self.g * (self.i * self.p - self.m * self.l) +
                         self.k * (self.m * self.h - self.e * self.p) +
                         self.o * (self.e * self.l - self.i * self.h))
            tmp.i = d * (self.h * (self.i * self.n - self.m * self.j) +
                         self.l * (self.m * self.f - self.e * self.n) +
                         self.p * (self.e * self.j - self.i * self.f))
            tmp.m = d * (self.e * (self.n * self.k - self.j * self.o) +
                         self.i * (self.f * self.o - self.n * self.g) +
                         self.m * (self.j * self.g - self.f * self.k))

            tmp.b = d * (self.j * (self.c * self.p - self.o * self.d) +
                         self.n * (self.k * self.d - self.c * self.l) +
                         self.b * (self.o * self.l - self.k * self.p))
            tmp.f = d * (self.k * (self.a * self.p - self.m * self.d) +
                         self.o * (self.i * self.d - self.a * self.l) +
                         self.c * (self.m * self.l - self.i * self.p))
            tmp.j = d * (self.l * (self.a * self.n - self.m * self.b) +
                         self.p * (self.i * self.b - self.a * self.j) +
                         self.d * (self.m * self.j - self.i * self.n))
            tmp.n = d * (self.i * (self.n * self.c - self.b * self.o) +
                         self.m * (self.b * self.k - self.j * self.c) +
                         self.a * (self.j * self.o - self.n * self.k))

            tmp.c = d * (self.n * (self.c * self.h - self.g * self.d) +
                         self.b * (self.g * self.p - self.o * self.h) +
                         self.f * (self.o * self.d - self.c * self.p))
            tmp.g = d * (self.o * (self.a * self.h - self.e * self.d) +
                         self.c * (self.e * self.p - self.m * self.h) +
                         self.g * (self.m * self.d - self.a * self.p))
            tmp.k = d * (self.p * (self.a * self.f - self.e * self.b) +
                         self.d * (self.e * self.n - self.m * self.f) +
                         self.h * (self.m * self.b - self.a * self.n))
            tmp.o = d * (self.m * (self.f * self.c - self.b * self.g) +
                         self.a * (self.n * self.g - self.f * self.o) +
                         self.e * (self.b * self.o - self.n * self.c))

            tmp.d = d * (self.b * (self.k * self.h - self.g * self.l) +
                         self.f * (self.c * self.l - self.k * self.d) +
                         self.j * (self.g * self.d - self.c * self.h))
            tmp.h = d * (self.c * (self.i * self.h - self.e * self.l) +
                         self.g * (self.a * self.l - self.i * self.d) +
                         self.k * (self.e * self.d - self.a * self.h))
            tmp.l = d * (self.d * (self.i * self.f - self.e * self.j) +
                         self.h * (self.a * self.j - self.i * self.b) +
                         self.l * (self.e * self.b - self.a * self.f))
            tmp.p = d * (self.a * (self.f * self.k - self.j * self.g) +
                         self.e * (self.j * self.c - self.b * self.k) +
                         self.i * (self.b * self.g - self.f * self.c))

        return tmp


class Quaternion(object):
    def __init__(self, w=1, x=0, y=0, z=0):
        pass

    def __repr__(self):
        pass

    def __mul__(self, other):
        pass

    def __imul__(self, other):
        pass

    @property
    def copy(self):
        pass

    @property
    def magnitude(self):
        pass

    @property
    def identity(self):
        pass

    def rotate_axis(self, angle, axis):
        pass

    def rotate_euler(self, heading, attitude, bank):
        pass

    def rotate_matrix(self, mat):
        pass

    def conjugate(self):
        pass

    def normalize(self):
        pass

    def normalized(self):
        pass

    @property
    def angel_axis(self):
        pass

    @property
    def euler(self):
        pass

    @property
    def matrix(self):
        pass

    @classmethod
    def new_identity(cls):
        pass

    @classmethod
    def new_rotate_axis(cls, angle, axis):
        pass

    @classmethod
    def new_rotate_euler(cls, heading, attitude, bank):
        pass

    @classmethod
    def new_rotate_matrix(cls, mat):
        pass

    @classmethod
    def new_interpolate(cls, q1, q2, t):
        pass


class Geometry(object):
    pass


class Point2(Vector2, Geometry):
    pass


class Point3(Vector3, Geometry):
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