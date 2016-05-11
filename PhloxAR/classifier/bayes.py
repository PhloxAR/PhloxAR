# -*- coding: utf-8 -*-
from __future__ import division, print_function
from __future__ import unicode_literals, absolute_import
import numpy as npy


class BayesClassifier(object):
    def __init__(self):
        """
        Initialize classifier with training data.
        """
        self._labels = []  # class labels
        self._mean = []  # class mean
        self._var = []  # class variance
        self._n = 0  # nbr of classes

    def train(self, data, labels=None):
        """
        Train on data (list of arrays n * dim).
        Labels are optional, default is 0...n-1
        """
        if labels is None:
            labels = range(len(data))

        self._labels = labels
        self._n = len(labels)

        for c in data:
            self._mean.append(npy.mean(c, axis=0))
            self._var.append(npy.var(c, axis=0))

    def classify(self, points):
        """
        Classify the points by computing probabilities
        for each class and return most probable label.
        """
        # compute probabilities for each class
        est_prob = npy.array(
                [_gauss(m, v, points) for m, v in zip(self._mean, self._var)])

        print('est prob', est_prob.shape, self._labels)
        # get index of highest probability, this gives class label
        ndx = est_prob.argmax(axis=0)

        est_labels = npy.array([self._labels[n] for n in ndx])

        return est_labels, est_prob


def _gauss(m, v, x):
    """
    Evaluate Gaussian in d-dimensions with independent
    mean m and variance v at the points in (the rows of) x.
    http://en.wikipedia.org/wiki/Multivariate_normal_distribution
    """
    if len(x.shape) == 1:
        n, d = 1, x.shape[0]
    else:
        n, d = x.shape

    # covariance matrix, subtract mean
    S = npy.diag(1/v)
    x = x-m
    # product of probabilities
    y = npy.exp(-0.5 * npy.diag(npy.dot(x, npy.dot(S, x.T))))

    # normalize and return
    return y * (2*npy.pi)**(-d/2.0) / (npy.sqrt(npy.prod(v)) + 1e-6)
