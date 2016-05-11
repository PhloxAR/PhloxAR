# -*- coding: utf-8 -*-
from __future__ import division, print_function
from __future__ import unicode_literals, absolute_import
import numpy as npy


class KnnClassifier(object):
    def __init__(self, labels, samples):
        """
        Initialize classifier with training data.
        """
        self.labels = labels
        self.samples = samples

    def classify(self, point, k=3):
        """
        Classify a point against k nearest
        in the training data, return label.
        """

        # compute distance to all training points
        dist = npy.array([_l2dist(point, s) for s in self.samples])

        # sort them
        ndx = dist.argsort()

        # use dictionary to store the k nearest
        votes = {}
        for i in range(k):
            label = self.labels[ndx[i]]
            votes.setdefault(label, 0)
            votes[label] += 1

        return max(votes)


def _l2dist(p1, p2):
    return npy.sqrt(sum((p1 - p2) ** 2))


def _l1dist(v1, v2):
    return sum(abs(v1 - v2))