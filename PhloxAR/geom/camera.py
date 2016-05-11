# -*- coding: utf-8 -*-
from __future__ import division, print_function
from __future__ import unicode_literals, absolute_import
from scipy import linalg
import numpy as npy


class Camera(object):
    """
    Class for representing pin-hole cameras.
    """
    def __init__(self, mat_p):
        """
        Initialize P = K [R|T] camera model.
        """
        self._mat_p = mat_p
        self._mat_k = None  # calibration matrix
        self._mat_r = None  # rotation
        self._mat_t = None  # translation
        self._c = None  # camera center

    def project(self, x):
        """
        Project points in x (4 * n array) and normalize coordinates.
        """
        dx = npy.dot(self._mat_p, x)
        for i in range(3):
            dx[i] /= dx[2]

        return dx

    def factor(self):
        """
        Factorize the camera matrix into k, r, t as P = K [R|T]
        """
        # factor first 3*3 part
        k, r = linalg.rq(self._mat_p[:, :3])

        # make diagonal of K positive
        t = npy.diag(npy.sign(npy.diag(k)))
        if linalg.det(t) < 0:
            t[1, 1] *= -1

        self._mat_k = npy.dot(k, t)
        self._mat_r = npy.dot(t, r)
        self._mat_t = npy.dot(linalg.inv(self._mat_k), self._mat_p[:, 3])

        return self._mat_k, self._mat_r, self._mat_t

    def center(self):
        """
        Compute and return the camera center.
        """
        if self._c is not None:
            return self._c
        else:
            self.factor()
            self._c = -npy.dot(self._mat_r.T, self._mat_t)
            return self._c


def rotation_matrix(v):
    """
    Create a 3D rotation matrix for rotation around the axis of the vector v
    """
    r = npy.eye(4)
    r[:3, :3] = linalg.expm([[0, -v[2], v[1]],
                             [v[2], 0, -v[0]],
                             [-v[1], v[0], 0]])


def rq(a):
    q, r = linalg.qr(npy.flipud(a).T)
    r = npy.flipud(r.T)
    q = q.T

    return r[:, ::-1], q[::-1, :]

