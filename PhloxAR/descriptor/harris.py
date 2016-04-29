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
    1. Compute x and y derivatives of image
       Ix = Gx*I  Iy = Gy*I
    2. Compute products of derivatives at every pixel
       Ix2 = Ix.Ix  Iy2 = Iy.Iy  Ixy = Ix.Iy
    3. Compute the sums of the products of derivatives at each pixel
       Sx2 = G*Ix2  Sy2 = G*Iy2  Sxy = G*Ixy
    4. Define at each pixel (x, y) the matrix M
       M = [ Sx2  Sxy]
           [ Sxy  Sy2]
    5. Compute the response of the detector at each pixel
       R = Det(M) - k(Trace(M))^2
       k is an empirically determined constant; k = 0.04 ~ 0.06
       Det(M) = eigenvalue1 x eigenvalue2
       Trace(M) = eigenvalue1 + eigenvalue2
    6. Threshold on value of R. Compute nonmax suppression

    :param img: the image to compute
    :param sigma: for Gaussian kernel
    :return: an image with each pixel containing the value of the Harris
              response function.
    """
    # derivatives
    ix = numpy.zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), (0, 1), ix)
    iy = numpy.zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), (1, 0), iy)

    # compute components of the Harris matrix, gaussian function is used
    # as window function
    wxx = filters.gaussian_filter(ix * ix, sigma)
    wxy = filters.gaussian_filter(ix * iy, sigma)
    wyy = filters.gaussian_filter(iy * iy, sigma)

    # determinant and trace
    wdet = wxx * wyy - wxy**2
    wtr = wxx + wyy

    return wdet / wtr


def get_harris_points(img, min_dist=10, threshold=0.1):
    """
    Return corners from a Harris response image
    :param img: the image after compute harris response
    :param min_dist: minimum number of pixels separating corner
                      and image boundary.
    :param threshold: decides whether a pixel should be choose or not
    :return:
    """
    # find top corner candidates above a threshold
    corner_threshold = img.max() * threshold
    harris_img = (img > corner_threshold) * 1

    # get coordinates of candidates
    coords = numpy.array(harris_img.nonzero()).T

    # and their values
    candidate_values = [img[c[0], c[1]] for c in coords]

    # sort candidates (reverse to get descending order)
    index = numpy.argsort(candidate_values)[::-1]

    # store allowed point locations in array
    allowed_locations = numpy.zeros(img.shape)
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
    """
    For each point return pixel values around the point using a
    neighbourhood of width 2*wid + 1. (Assume points are extracted
    with minimum distance > wid).
    :param img:
    :param filtered_coords:
    :param wid:
    :return: descriptors
    """
    desc = []

    for coords in filtered_coords:
        patch = img[coords[0] - wid:coords[0] + wid + 1,
                coords[1] - wid:coords[1] + wid + 1].flatten()
        desc.append(patch)

    return desc


def match(desc1, desc2, threshold=0.5):
    """
    For each corner point descriptor in the first image,
    select its match to second image using normalized
    cross correlation.
    :param desc1:
    :param desc2:
    :param threshold:
    :return:
    """
    n = len(desc1[0])

    # pair-wise distances
    d = -numpy.ones((len(desc1), len(desc2)))

    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - numpy.mean(desc1[i])) / numpy.std(desc1[i])
            d2 = (desc2[j] - numpy.mean(desc2[j])) / numpy.std(desc2[j])
            ncc = sum(d1 * d2) / (n - 1)
            if ncc > threshold:
                d[i, j] = ncc

    ndx = numpy.argsort(-d)
    match_scores = ndx[:, 0]

    return match_scores


def match2(desc1, desc2, threshold=0.5):
    """
    Two-sided symmetric version of match
    :param desc1:
    :param desc2:
    :param threshold:
    :return:
    """
    matches12 = match(desc1, desc2, threshold)
    matches21 = match(desc2, desc1, threshold)

    ndx12 = numpy.where(matches12 >= 0)[0]

    # remove matches that are not symmetric
    for n in ndx12:
        if matches21[matches12[n]] != n:
            matches12[n] = -1

    return matches12


def append_image(img1, img2):
    """
    Return a new image that appends the two images side-by-side.
    :param img1:
    :param img2:
    :return:
    """
    row1 = img1.shape[0]
    row2 = img2.shape[0]

    if row1 < row2:
        img1 = numpy.concatenate((img1, numpy.zeros((row2-row1, img1.shape[1]))),
                                 axis=0)
    elif row1 > row2:
        img2 = numpy.concatenate((img2, numpy.zeros((row1-row2, img2.shape[1]))),
                                 axis=0)
    # if none of these cases, no filling needed.
    return numpy.concatenate((img1, img2), axis=1)


def plot_matches(img1, img2, loc1, loc2, match_scores, show_below=True):
    """
    Show a figure with lines joining the accepted matches input.
    :param img1: first image
    :param img2: second image
    :param loc1: feature locations of first image
    :param loc2: feature locations of second image
    :param match_scores: output from 'match()'
    :param show_below: if images should be shown below matches
    :return: None
    """
    img3 = append_image(img1, img2)

    if show_below:
        img3 = numpy.vstack((img3, img3))

    pylab.imshow(img3)

    col1 = img1.shape[1]

    for i, m in enumerate(match_scores):
        if m > 0:
            pylab.plot([loc1[i][1], loc2[m][1] + col1], [loc1[i][0], loc2[m][0]],
                       'c')
    pylab.axis('off')