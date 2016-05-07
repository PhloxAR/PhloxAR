# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PIL import Image
from numpy import *
import os

from . import sift


def process_image_dsift(image, result, size=20, steps=10,
                        force_orientation=False, resize=None):
    """
    Process an image with densely sampled SIFT descriptors
    and save the results in a file. Optional input: size of features,
    steps between locations, forcing computation of descriptor orientation
    (False means all are oriented upwards), tuple for resizing the image.
    """

    im = Image.open(image).convert('L')
    if resize is not None:
        im = im.resize(resize)
    m,n = im.size

    if image[-3:] != 'pgm':
        # create a pgm file
        im.save('tmp.pgm')
        image = 'tmp.pgm'

    # create frames and save to temporary file
    scale = size / 3.0
    x, y = meshgrid(range(steps,m,steps),range(steps,n,steps))
    xx,yy = x.flatten(),y.flatten()
    frame = array([xx,yy,scale*ones(xx.shape[0]),zeros(xx.shape[0])])
    savetxt('tmp.frame',frame.T,fmt='%03.3f')

    if force_orientation:
        cmd = str("sift " + image + " --output=" + result +
                  " --read-frames=tmp.frame --orientations")
    else:
        cmd = str("sift " + image + " --output=" + result +
                  " --read-frames=tmp.frame")
    os.system(cmd)
    print("processed " + image + "to " + result)