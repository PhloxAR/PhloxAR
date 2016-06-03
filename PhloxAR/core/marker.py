# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function

import numpy as np
import cv2


__all__ = [
    'MarkerDetector'
]


class MarkerDetector(object):
    QUADRILATERAL_POINTS = 4
    BLACK_THRESHOLD = 100
    WHITE_THRESHOLD = 155
    MARKER_NAME_INDEX = 3

    def __init__(self):
        with np.load('calib/mine_cam.npz') as calib:
            self.mtx, self.dist, = [calib[i] for i in ('mtx', 'dist')]

    def detect(self, image):

        markers = []

        # Stage 1: Convert the image into gray scale on threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 3, 3)

        # Stage 2: Find contours
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_SIMPLE)

        # sort contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        for cnt in contours:

            # Stage 3: Shape check
            perimeter = cv2.arcLength(cnt, True)
            # approx = cv2.approxPolyDP(contour, 0.01*perimeter, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * perimeter, True)

            if len(approx) == self.QUADRILATERAL_POINTS:

                # Stage 4: Perspective warping
                topdown_quad = self.warping_transform(gray, approx.reshape(4, 2))

                # Stage 5: Border check
                if topdown_quad[(topdown_quad.shape[0] / 100.0) * 5,
                                (topdown_quad.shape[
                                     1] / 100.0) * 5] > self.BLACK_THRESHOLD:
                    continue

                # Stage 6: Get marker pattern
                marker_pattern = None

                try:
                    marker_pattern = self.get_marker_pattern(topdown_quad,
                                                             self.BLACK_THRESHOLD,
                                                             self.WHITE_THRESHOLD)
                except:
                    continue

                if not marker_pattern:
                    continue

                # Stage 7: Match marker pattern
                marker_found, marker_rotation, marker_name = self.match_marker_pattern(
                    marker_pattern)

                if marker_found:

                    # Stage 8: Duplicate marker check
                    if marker_name in [marker[self.MARKER_NAME_INDEX] for marker
                                       in markers]:
                        continue

                    # Stage 9: Get rotation and translation vectors
                    print('----------------- 0.o -----------------------------')
                    rvecs, tvecs = self.get_vectors(image, approx.reshape(4, 2),
                                                    self.mtx, self.dist)
                    markers.append([rvecs, tvecs, marker_rotation, marker_name])

        return markers

    def _marker_coords(self, max_width, max_height):
        print('max_width and max_height is', max_width, max_height)
        return np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype='float32')

    def warping_transform(self, image, src):

        # src and dst points
        src = self._order_points(src)

        (max_width, max_height) = self._width_height(src)
        dst = self._marker_coords(max_width, max_height)

        # warp perspective
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, matrix, self._width_height(src))
        _, warped = cv2.threshold(warped, 125, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return warped

    def _width_height(self, points):

        tl, tr, br, bl = points

        top_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        bottom_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))

        width = max(int(top_width), int(bottom_width))

        left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))

        height = max(int(left_height), int(right_height))

        return width, height

    def _order_points(self, points):
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)

        ordered_points = np.zeros((4, 2), dtype=np.float32)

        ordered_points[0] = points[np.argmin(s)]
        ordered_points[2] = points[np.argmax(s)]
        ordered_points[1] = points[np.argmin(diff)]
        ordered_points[3] = points[np.argmax(diff)]

        return ordered_points

    def get_marker_pattern(self, image, black_threshold, white_threshold):
        # collect pixel from each cell (left to right, top to bottom)
        cells = []

        cell_half_width = int(round(image.shape[1] / 10.0))
        cell_half_height = int(round(image.shape[0] / 10.0))

        for i in range(3, 9, 2):
            for j in range(3, 9, 2):
                cells.append(image[cell_half_height * i, cell_half_width * j])

        # threshold pixels to either black or white
        for idx, val in enumerate(cells):
            if val < black_threshold:
                cells[idx] = 0
            elif val > white_threshold:
                cells[idx] = 1
            else:
                return None
        return cells

    def get_vectors(self, image, points, mtx, dist):

        # order points
        points = self._order_points(points)

        # set up criteria, image, points and axis
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        imgp = np.array(points, dtype=np.float32)

        objp = np.array([[0., 0., 0.], [1., 0., 0.],
                         [1., 1., 0.], [0., 1., 0.]], dtype=np.float32)

        # calculate rotation and translation vectors
        cv2.cornerSubPix(gray, imgp, (11, 11), (-1, -1), criteria)

        ret, rvecs, tvecs = cv2.solvePnP(objp, imgp, mtx, dist)

        return rvecs, tvecs

    # match marker pattern to database record
    def match_marker_pattern(self, marker_pattern):
        marker_found = False
        marker_rotation = None
        marker_name = None

        for marker_record in MARKER_TABLE:
            for idx, val in enumerate(marker_record[0]):
                if marker_pattern == val:
                    marker_found = True
                    marker_rotation = idx
                    marker_name = marker_record[1]
                    break
            if marker_found:
                val = 0
                for i, x in enumerate(marker_pattern):
                    val += x * np.power(2, i)

                break

        return marker_found, marker_rotation, marker_name