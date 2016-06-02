# -*- coding: utf-8 -*-

import cv2


class MarkerDetector(object):

    def __init__(self):
        pass

    def process_frame(self):
        pass

    def get_transformations(self):
        pass

    def find_markers(self):
        pass

    def prepare_image(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def perform_threshold(self):
        pass

    def find_contours(self):
        pass

    def find_marker_candidates(self):
        pass

    def detect_markers(self):
        pass

    def estimate_position(self):
        pass
