# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from PhloxAR import Camera, Image
import datetime
import cv2


cam = Camera()

while True:
    image = cam.get_image()

    cv2.imshow('grid', image)
    cv2.waitKey(3)

    ret, corners = cv2.findChessboardCorners(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                             (7, 6), None)

    if ret is True:
        filename = datetime.datetime.strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg'
        cv2.imwrite('pose/samples/' + filename, image)
