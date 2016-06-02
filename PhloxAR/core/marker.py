# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function

import numpy as np
import cv2


class MarkerDetector(object):
    BLACK_THRESH = 100
    WHITE_THRESH = 155
    MARKER_IDX = 3

    def __init__(self):
        with np.load('calib/mine_cam.npz') as calib:
            self.mtx, self.dist = [calib[i] for i in ('mtx', 'dist')]

    def detect(self, image, marker):
        markers = []

        # stage 1: convert the image into gray scale an threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 3, 3)

        # stage 2: find contours
        _, contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # sort contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        for cnt in contours:
            # stage 3: shape check
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05*perimeter, True)

            if len(approx) == 4:
                # stage 4: perspective warping
                warped = self.warping_transform(gray, approx.rehspae(4, 2))

                # stage 5: border check
                if warped[(warped.shape[0] / 100.0) * 5,
                          (warped.shape[1] / 100.0) * 5] > self.BLACK_THRESH:
                    continue

                marker_pattern = self.get_marker_pattern(warped,
                                                         self.BLACK_THRESH,
                                                         self.WHITE_THRESH)

                if not marker_pattern:
                    continue

                # stage 7: match marker pattern
                found, rotation, name = marker.match_marker_pattern(
                    marker_pattern
                )

                if found:
                    # stage 8: duplicate marker check
                    if name in [m[self.MARKER_IDX] for m in markers]:
                        continue

                    # stage 9: get rotation and translation vectors
                    rvecs, tvecs = self.get_vectors(image, approx.reshape(4, 2),
                                                    self.mtx, self.dist)
                    markers.append([rvecs, tvecs, rotation, name])

            return markers

    def _order_points(self, points):
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)

        ordered = np.zeros((4, 2), dtype=np.float32)

        ordered[0] = points[np.argmin(s)]
        ordered[2] = points[np.argmax(s)]
        ordered[1] = points[np.argmin(diff)]
        ordered[3] = points[np.argmax(diff)]

        return ordered

    def _contour_width_height(self, contour):
        """
        Calculate a contour's width and height, the contour is consists of
        four points.

        Args:
            contour:

        Returns:

        """
        # top-left, top-right, bottom-right, bottom-left
        tl, tr, br, bl = contour

        top_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        bottom_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))

        left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))

        width = max(int(top_width), int(bottom_width))
        height = max(int(left_height), int(right_height))

        return width, height

    def _marker_coords(self, width, height):
        return np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], np.float32)

    def warping_transform(self, image, src):
        # src and dst points
        src = self._order_points(src)

        (max_width, max_height) = self._contour_width_height(src)
        dst = self._marker_coords(max_width, max_height)

        # warp perspective
        matrix = cv2.getPerspectiveTransform(src, dst)

        warped = cv2.warpPerspective(image, matrix, self._contour_width_height(src))

        _, warped = cv2.threshold(warped, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return warped

    def get_marker_pattern(self, image, black_thresh, white_thresh):
        # collect pixel from each cell (left to right, top to bottom)
        cells = []
        rows = []
        cols = []

        cell_half_width = int(round(image.shape[1] / 10.0))
        cell_half_height = int(round(image.shape[0] / 10.0))

        rows.append(cell_half_height * 3)
        rows.append(cell_half_height * 5)
        rows.append(cell_half_height * 7)
        cols.append(cell_half_width * 3)
        cols.append(cell_half_width * 5)
        cols.append(cell_half_width * 7)

        for i in rows:
            for j in cols:
                cells.append(image[i, j])

        # threshold pixels to either black or white
        for idx, val in enumerate(cells):
            if val < black_thresh:
                cells[idx] = 0
            elif val > white_thresh:
                cells[idx] = 1
            else:
                return None

        return cells

    def get_vectors(self, image, points, mtx, dist):
        # order points
        points = self._order_points(points)

        # set up criteria, image, points and axis
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        imgp = np.array(points, dtype='float32')

        objp = np.array([[0., 0., 0.], [1., 0., 0.],
                         [1., 1., 0.], [0., 1., 0.]], dtype='float32')

        # calculate rotation and translation vectors
        cv2.cornerSubPix(gray, imgp, (11, 11), (-1, -1), criteria)

        # rvecs, tvecs, _ = cv2.solvePnPRansac(objp, imgp, mtx, dist)
        ret, rvecs, tvecs = cv2.solvePnP(objp, imgp, mtx, dist)

        return rvecs, tvecs

    def threshold(self, gray_img):
        pass

    def find_contours(self, thresh_img):
        pass

    def find_candidates(self, contours):
        pass

    def recognize(self, gray_img):
        pass

    def esimate(self, marker):
        pass


class Marker(object):
    MARKER_TABLE = [
        [
            [
                [0, 1, 0, 1, 0, 0, 0, 1, 1],
                [0, 0, 1, 1, 0, 1, 0, 1, 0],
                [1, 1, 0, 0, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 1, 0, 0]
            ],
            'ROCKY'
        ],
        [
            [
                [1, 0, 0, 0, 1, 0, 1, 0, 1],
                [0, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 0]
            ],
            'SPORTY'
        ]
    ]

    @classmethod
    def match_marker_pattern(self, pattern):
        found = False
        rotation = None
        name = None
        val = 0

        for record in self.MARKER_TABLE:
            for idx, val in enumerate(record[0]):
                if pattern == val:
                    found = True
                    rotation = idx
                    name = record[1]

            if found:
                for i, x in enumerate(pattern):
                    val += x * np.power(2, i)

        return found, rotation, name

