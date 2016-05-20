# -*- coding: utf-8 -*-
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

import itertools
from ..base import np, cv2
from .track import SURFTrack

__all__ = [
    'surf_tracker'
]


def surf_tracker(img, bb, ts, **kwargs):
    """
    **DESCRIPTION**

    (Dev Zone)
    Tracking the object surrounded by the bounding box in the given
    image using SURF keypoints.
    Warning: Use this if you know what you are doing. Better have a
    look at Image.track()
    **PARAMETERS**
    * *img* - Image - Image to be tracked.
    * *bb*  - tuple - Bounding Box tuple (x, y, w, h)
    * *ts*  - TrackSet - 
    Optional PARAMETERS:
    eps_val     - eps for DBSCAN
                  The maximum distance between two samples for them
                  to be considered as in the same neighborhood.

    min_samples - min number of samples in DBSCAN
                  The number of samples in a neighborhood for a point
                  to be considered as a core point.

    distance    - thresholding KNN distance of each feature
                  if KNN distance > distance, point is discarded.
    **RETURNS**
    
    **HOW TO USE**
    >>> cam = Camera()
    >>> ts = []
    >>> img = cam.getImage()
    >>> bb = (100, 100, 300, 300) # get BB from somewhere
    >>> ts = surf_tracker(img, bb, ts, eps_val=0.7, distance=150)
    >>> while (some_condition_here):
        ... img = cam.getImage()
        ... bb = ts[-1].bb
        ... ts = surf_tracker(img, bb, ts, eps_val=0.7, distance=150)
        ... ts[-1].drawBB()
        ... img.show()
    This is too much confusing. Better use
    Image.track() method.
    READ MORE:
    SURF based Tracker:
    Matches keypoints from the template image and the current frame.
    flann based matcher is used to match the keypoints.
    Density based clustering is used classify points as in-region (of bounding box)
    and out-region points. Using in-region points, new bounding box is predicted using
    k-means.
    """
    eps_val = 0.69
    min_samples = 5
    distance = 100

    for key in kwargs:
        if key == 'eps_val':
            eps_val = kwargs[key]
        elif key == 'min_samples':
            min_samples = kwargs[key]
        elif key == 'dist':
            distance = kwargs[key]

    from scipy.spatial import distance as Dis
    from sklearn.cluster import DBSCAN

    if len(ts) == 0:
        # Get template keypoints
        bb = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
        template = img

        surf = cv2.xfeatures2d.SURF_create()

        template_region = template.gray_narray[bb[1]:bb[1] + bb[3],
                          bb[0]:bb[0] + bb[2]]

        tkps, tdescs = surf.detectAndCompute(template_region, None)

    else:
        template = ts[-1].template
        tkps = ts[-1].kps
        tdescs = ts[-1].descs
        surf = ts[-1].sift

    newimg = img.narray

    # Get image keypoints
    skps, sdescs = surf.detectAndCompute(newimg, None)

    if tdescs is None:
        print("Descriptors are Empty")
        return None

    if sdescs is None:
        track = SURFTrack(img, skps, surf, template, skps, sdescs, tkps, tdescs)
        return track

    # flann based matcher
    flann_params = dict(algorithm=1, trees=4)
    flann = cv2.flann.Index(sdescs, flann_params)
    idx, dist = flann.knnSearch(tdescs, 1, params={})
    del flann

    # filter points using distance criteria
    dist = (dist[:, 0] / 2500.0).reshape(-1, ).tolist()
    idx = idx.reshape(-1).tolist()
    indices = sorted(range(len(dist)), key=lambda x: dist[x])

    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    skp_final = []
    skp_final_labelled = []
    data_cluster = []

    for i, dis in itertools.izip(idx, dist):
        if dis < distance:
            skp_final.append(skps[i])
            data_cluster.append((skps[i].pt[0], skps[i].pt[1]))

    # Use Density based clustering to further filter out keypoints
    n_data = np.asarray(data_cluster)
    D = Dis.squareform(Dis.pdist(n_data))
    S = 1 - (D / np.max(D))

    db = DBSCAN(eps=eps_val, min_samples=min_samples).fit(S)
    core_samples = db.core_sample_indices_
    labels = db.labels_
    for label, i in zip(labels, range(len(labels))):
        if label == 0:
            skp_final_labelled.append(skp_final[i])

    track = SURFTrack(img, skp_final_labelled, surf, template, skps, sdescs,
                      tkps, tdescs)

    return track
