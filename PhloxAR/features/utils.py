# -*- coding: utf-8 -*-
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PhloxAR.base import *
from PhloxAR.image import *
from PhloxAR.color import *
from PhloxAR.features.feature import Feature, FeatureSet
from PhloxAR.features.detection import *

"""
So this is a place holder for some routines that should live in
featureset if we can make it specific to a type of features
"""


def GetParallelSets(line_fs, parallel_thresh=2):
    result = []
    sz = len(line_fs)
    # construct the pairwise cross product ignoring dupes
    for i in range(0, sz):
        for j in range(0, sz):
            if j <= i:
                result.append(npy.Inf)
            else:
                result.append(npy.abs(line_fs[i].cross(line_fs[j])))

    result = npy.array(result)
    # reshape it
    result = result.reshape(sz, sz)
    # find the lines that are less than our thresh
    l1, l2 = npy.where(result < parallel_thresh)
    idxs = zip(l1, l2)
    ret = []
    # now construct the line pairs
    for idx in idxs:
        ret.append((line_fs[idx[0]], line_fs[idx[1]]))
    return ret


def ParallelDistance(line1, line2):
    pass
