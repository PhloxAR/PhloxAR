# -*- coding: utf-8 -*-
from __future__ import division, print_function
from __future__ import unicode_literals, absolute_import
import numpy as npy
from itertools import combinations
from PIL import Image, ImageDraw


class ClusterNode(object):
    def __init__(self, vec, left, right, dist=0.0, count=1):
        self._left = left
        self._right = right
        self._vec = vec
        self._dist = dist
        self._count = count  # only used for weighted average

    def extract_clusters(self, dist):
        """
        Extract list of sub-tree clusters from hcluster tree with
        distance < dist
        """
        if self._dist < dist:
            return [self]

        return (self._left.extract_cluster(dist) +
                self._right.extract_cluster(dist))

    def get_cluster_elements(self):
        """
        Return ids for elements in a cluster sub-tree.
        """
        return (self._left.get_cluster_elements +
                self._right.get_cluster_elements())

    def get_height(self):
        """
        Return the height of a node, height is sum of each branch.
        """
        return self._left.get_height() + self._right.get_height()

    def get_depth(self):
        """
        Return the depth of a node, depth is max of each child plus own distance
        """
        return max(self._left.get_depth(), self._right.get_depth()) + self._dist

    def draw(self, draw, x, y, s, imlist, im):
        """
        Draw nodes recursively with image thumbnails for leaf nodes.
        """
        h1 = int(self._left.get_height() * 20 / 2)
        h2 = int(self._right.get_height() * 20 / 2)
        top = y - (h1 + h2)
        bottom = y + (h1 + h2)

        # vertical line to children
        draw.line((x, top + h1, x, bottom - h2), fill=(0, 0, 0))

        # horizontal lines
        ll = self._dist * s
        draw.line((x, top + h1, x + ll, top + h1), fill=(0, 0, 0))
        draw.line((x, bottom - h2, x + ll, bottom - h2), fill=(0, 0, 0))

        # draw left and right child nodes recursively
        self._dist.draw(draw, x + ll, top + h1, s, imlist, im)
        self._dist.draw(draw, x + ll, bottom - h2, s, imlist, im)


class ClusterLeafNode(object):
    def __init__(self, vec, id):
        self._vec = vec
        self._id = id

    def extract_clusters(self, dist):
        return [self]

    def get_cluster_elements(self):
        return [self._id]

    def get_height(self):
        return 1

    def get_depth(self):
        return 0

    def draw(self, draw, x, y, s, imlist, im):
        nodeim = Image.open(imlist[self.id])
        nodeim.thumbnail([20, 20])
        ns = nodeim.size
        im.paste(nodeim, [int(x), int(y - ns[1] // 2), int(x + ns[0]),
                          int(y + ns[1] - ns[1] // 2)])


def l2dist(v1, v2):
    return npy.sqrt(sum((v1 - v2) ** 2))


def l1dist(v1, v2):
    return sum(abs(v1 - v2))


def hcluster(features, distfn=l2dist):
    """
     Cluster the rows of features using
     hierarchical clustering.
     """

    # cache of distance calculations
    distances = {}

    # initialize with each row as a cluster
    node = [ClusterLeafNode(npy.array(f), id=i) for i, f in enumerate(features)]

    while len(node) > 1:
        closest = float('Inf')

        # loop through every pair looking for the smallest distance
        for ni, nj in combinations(node, 2):
            if (ni, nj) not in distances:
                distances[ni, nj] = distfn(ni.vec, nj.vec)

            d = distances[ni, nj]
            if d < closest:
                closest = d
                lowestpair = (ni, nj)
        ni, nj = lowestpair

        # average the two clusters
        new_vec = (ni.vec + nj.vec) / 2.0

        # create new node
        new_node = ClusterNode(new_vec, left=ni, right=nj, dist=closest)
        node.remove(ni)
        node.remove(nj)
        node.append(new_node)

    return node[0]


def draw_dendrogram(node, imlist, filename='clusters.jpg'):
    """    Draw a cluster dendrogram and save to a file. """

    # height and width
    rows = node.get_height() * 20
    cols = 1200

    # scale factor for distances to fit image width
    s = float(cols - 150) / node.get_depth()

    # create image and draw object
    im = Image.new('RGB', (cols, rows), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    # initial line for start of tree
    draw.line((0, rows / 2, 20, rows / 2), fill=(0, 0, 0))

    # draw the nodes recursively
    node.draw(draw, 20, (rows / 2), s, imlist, im)
    im.save(filename)
    im.show()
