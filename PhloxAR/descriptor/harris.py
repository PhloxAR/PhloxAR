# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

import matplotlib.pylab as pylab
import numpy
from scipy.ndimage import filters


def compute_harris_response(img, sigma=3):
    """
    Compute the Harris corner detector response function for each pixel in
    a gray level image.
    :param img: the image to compute
    :param sigma: for Gaussian kernel
    :return:
    """
    # derivatives
    ix = numpy.zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), (0, 1), ix)
    iy = numpy.zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), (1, 0), iy)

    # compute components of the Harris matrix
    wxx = filters.gaussian_filter(ix * ix, sigma)
    wxy = filters.gaussian_filter(ix * iy, sigma)
    wyy = filters.gaussian_filter(iy * iy, sigma)

    # determinant and trace
    wdet = wxx * wyy - wxy**2
    wtr = wxx + wyy

    return wdet / wtr


def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    """
    Return corners from a Harris response image
    :param harrisim:
    :param min_dist: minimum number of pixels separating corner
                      and image boundary.
    :param threshold:
    :return:
    """
    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    # get coordinates of candidates
    coords = numpy.array(harrisim_t.nonzero())

    # and their values
    candidate_values = [harrisim[c[0], c[1]] for c in coords]

    # sort candidates (reverse to get descending order)
    index = numpy.argsort(candidate_values)[::-1]

    # store allowed point locations in array
    allowed_locations = numpy.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i, 0] - min_dist):(coords[i, 0] + min_dist),
            (coords[i, 1] - min_dist):(coords[i, 1] + min_dist)] = 0

    return filtered_coords


def plot_harris_points(img, filtered_coords):
    """
    Plots corners found in image.
    :param img:
    :param filtered_coords:
    :return:
    """
    pylab.figure()
    pylab.gray()
    pylab.imshow(img)
    pylab.plot([p[1] for p in filtered_coords],
               [p[0] for p in filtered_coords],
               '*')
    pylab.axis('off')
    pylab.show()


def get_descriptors(img, filtered_coords, wid=5):
    pass