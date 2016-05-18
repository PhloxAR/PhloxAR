# -*- coding:utf-8 -*-
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PhloxAR.image import Image
from PhloxAR.features.feature import FeatureSet
from PhloxAR.features.blob import Blob
from PhloxAR.base import *
import cv2


__all__ = [
    'BlobMaker'
]


class BlobMaker(object):
    """
    Blob maker encapsulates all of the contour extraction process and data, so
    it can be used inside the image class, or extended and used outside the
    image class. The general idea is that the blob maker provides the utilities
    that one would use for blob extraction. Later implementations may include
    tracking and other features.
    """
    _mem_storage = None

    def __init__(self):
        self._mem_storage = cv.CreateMemStorage()

    def extract_with_model(self, img, colormodel, minsize=10, maxsize=0):
        """
        Extract blobs using a color model
        :param img: the input image
        :param colormodel: the color model to use.
        :param minsize: the minimum size of the returned features.
        :param maxsize: the maximum size of the returned features 0=uses the default value.

        Parameters:
            img - Image
            colormodel - ColorModel object
            minsize - Int
            maxsize - Int
        """
        if maxsize <= 0:
            maxsize = img.width * img.height
        gray = colormodel.threshold(img)
        blobs = self.extract_from_binary(gray, img, minArea=minsize,
                                         maxArea=maxsize)
        ret = sorted(blobs, key=lambda x: x.mArea, reverse=True)
        return FeatureSet(ret)

    def extract(self, img, threshval=127, minsize=10, maxsize=0,
                threshblocksize=3, threshconstant=5):
        """
        This method performs a threshold operation on the input image and then
        extracts and returns the blobs.
        img       - The input image (color or b&w)
        threshval - The threshold value for the binarize operation. If threshval = -1 adaptive thresholding is used
        minsize   - The minimum blob size in pixels.
        maxsize   - The maximum blob size in pixels. 0=uses the default value.
        threshblocksize - The adaptive threhold block size.
        threshconstant  - The minimum to subtract off the adaptive threshold
        """
        if maxsize <= 0:
            maxsize = img.width * img.height

        # create a single channel image, thresholded to parameters

        blobs = self.extract_from_binary(
            img.binarize(threshval, 255, threshblocksize,
                         threshconstant).invert(), img, minsize, maxsize)
        ret = sorted(blobs, key=lambda x: x.mArea, reverse=True)
        return FeatureSet(ret)

    def extract_from_binary(self, binaryImg, colorImg, minsize=5, maxsize=-1,
                            appx_level=3):
        """
        This method performs blob extraction given a binary source image that is used
        to get the blob images, and a color source image.
        binarymg- The binary image with the blobs.
        colorImg - The color image.
        minSize  - The minimum size of the blobs in pixels.
        maxSize  - The maximum blob size in pixels.
        * *appx_level* - The blob approximation level - an integer for the maximum distance between the true edge and the approximation edge - lower numbers yield better approximation.
        """
        # If you hit this recursion limit may god have mercy on your soul.
        # If you really are having problems set the value higher, but this means
        # you have over 10,000,000 blobs in your image.
        sys.setrecursionlimit(5000)
        # h_next moves to the next external contour
        # v_next() moves to the next internal contour
        if maxsize <= 0:
            maxsize = colorImg.width * colorImg.height

        ret = []
        test = binaryImg.mean_color
        if test[0] == 0.00 and test[1] == 0.00 and test[2] == 0.00:
            return FeatureSet(ret)

        # There are a couple of weird corner cases with the opencv
        # connect components libraries - when you try to find contours
        # in an all black image, or an image with a single white pixel
        # that sits on the edge of an image the whole thing explodes
        # this check catches those bugs. -KAS
        # Also I am submitting a bug report to Willow Garage - please bare with us.
        ptest = (4 * 255.0) / (
        binaryImg.width * binaryImg.height)  # val if two pixels are white
        if test[0] <= ptest and test[1] <= ptest and test[2] <= ptest:
            return ret

        seq = cv.FindContours(binaryImg._get_gray_narray(),
                              self._mem_storage, cv.CV_RETR_TREE,
                              cv.CV_CHAIN_APPROX_SIMPLE)
        if not list(seq):
            warnings.warn("Unable to find Blobs. Retuning Empty FeatureSet.")
            return FeatureSet([])
        try:
            # note to self
            # http://code.activestate.com/recipes/474088-tail-call-optimization-decorator/
            ret = self._extract_from_binary(seq, False, colorImg, minsize,
                                               maxsize, appx_level)
        except RuntimeError as e:
            logger.warning("You exceeded the recursion limit. This means you "
                           "probably have too many blobs in your image. We "
                           "suggest you do some morphological operations "
                           "(erode/dilate) to reduce the number of blobs in "
                           "your image. This function was designed to max out "
                           "at about 5000 blobs per image.")
        except e:
            logger.warning("PhloxAR Find Blobs Failed - This could be an OpenCV "
                           "python binding issue")
        del seq
        return FeatureSet(ret)

    def _extract_from_binary(self, seq, isaHole, colorImg, minsize, maxsize,
                             appx_level):
        """
        The recursive entry point for the blob extraction. The blobs and holes 
        are presented as a tree and we traverse up and across the tree.
        """
        ret = []

        if seq is None:
            return ret

        nextLayerDown = []
        while True:
            # if we aren't a hole then we are an object, so get and 
            # return our featuress
            if not isaHole:
                temp = self._extract_data(seq, colorImg, minsize, maxsize,
                                          appx_level)
                if temp is not None:
                    ret.append(temp)

            nextLayer = seq.v_next()

            if nextLayer is not None:
                nextLayerDown.append(nextLayer)

            seq = seq.h_next()

            if seq is None:
                break

        for nextLayer in nextLayerDown:
            ret += self._extract_from_binary(nextLayer, not isaHole, colorImg,
                                             minsize, maxsize, appx_level)

        return ret

    def _extract_data(self, seq, color, minsize, maxsize, appx_level):
        """
        Extract the bulk of the data from a give blob. If the blob's are is too large
        or too small the method returns none.
        """
        if seq is None or not len(seq):
            return None
        area = cv.ContourArea(seq)
        if area < minsize or area > maxsize:
            return None

        ret = Blob()
        ret.image = color
        ret.mArea = area

        ret.mMinRectangle = cv.MinAreaRect2(seq)
        bb = cv.BoundingRect(seq)
        ret.x = bb[0] + (bb[2] / 2)
        ret.y = bb[1] + (bb[3] / 2)
        ret.mPerimeter = cv.ArcLength(seq)
        if seq is not None:  # KAS
            ret.contour = list(seq)
            if ret.contour is not None:
                ret.contourAppx = []
                appx = cv2.approxPolyDP(npy.array([ret.contour], 'float32'),
                                        appx_level, True)
                for p in appx:
                    ret.contourAppx.append((int(p[0][0]), int(p[0][1])))

        # so this is a bit hacky....

        # For blobs that live right on the edge of the image OpenCV reports the position and width
        #   height as being one over for the true position. E.g. if a blob is at (0,0) OpenCV reports
        #   its position as (1,1). Likewise the width and height for the other corners is reported as
        #   being one less than the width and height. This is a known bug.

        xx = bb[0]
        yy = bb[1]
        ww = bb[2]
        hh = bb[3]
        ret.points = [(xx, yy), (xx + ww, yy), (xx + ww, yy + hh),
                         (xx, yy + hh)]
        ret._update_extents()
        chull = cv.ConvexHull2(seq, cv.CreateMemStorage(), return_points=1)
        ret.mConvexHull = list(chull)

        del chull

        moments = cv.Moments(seq)

        # This is a hack for a python wrapper bug that was missing
        # the constants required from the ctype
        ret.m00 = area
        try:
            ret.m10 = moments.m10
            ret.m01 = moments.m01
            ret.m11 = moments.m11
            ret.m20 = moments.m20
            ret.m02 = moments.m02
            ret.m21 = moments.m21
            ret.m12 = moments.m12
        except:
            ret.m10 = cv.GetSpatialMoment(moments, 1, 0)
            ret.m01 = cv.GetSpatialMoment(moments, 0, 1)
            ret.m11 = cv.GetSpatialMoment(moments, 1, 1)
            ret.m20 = cv.GetSpatialMoment(moments, 2, 0)
            ret.m02 = cv.GetSpatialMoment(moments, 0, 2)
            ret.m21 = cv.GetSpatialMoment(moments, 2, 1)
            ret.m12 = cv.GetSpatialMoment(moments, 1, 2)

        ret.hu = cv.GetHuMoments(moments)

        # KAS -- FLAG FOR REPLACE 6/6/2012
        mask = self._get_mask(seq, bb)

        ret.avg_color = self._get_avg(color.bitmap, bb, mask)
        ret.avg_color = ret.avg_color[0:3]

        ret.mHoleContour = self._get_holes(seq)
        ret.mAspectRatio = ret.mMinRectangle[1][0] / \
                              ret.mMinRectangle[1][1]

        return ret

    def _get_holes(self, seq):
        """
        This method returns the holes associated with a blob as a list of tuples.
        """
        ret = None
        holes = seq.v_next()
        if holes is not None:
            ret = [list(holes)]
            while holes.h_next() is not None:
                holes = holes.h_next();
                temp = list(holes)
                if len(temp) >= 3:  # exclude single pixel holes
                    ret.append(temp)
        return ret

    def _get_mask(self, seq, bb):
        """
        Return a binary image of a particular contour sequence.
        """
        # bb = cv.BoundingRect(seq)
        mask = cv.CreateImage((bb[2], bb[3]), cv.IPL_DEPTH_8U, 1)
        cv.Zero(mask)
        cv.DrawContours(mask, seq, 255, 0, 0, thickness=-1,
                        offset=(-1 * bb[0], -1 * bb[1]))
        holes = seq.v_next()
        if holes is not None:
            cv.DrawContours(mask, holes, 0, 255, 0, thickness=-1,
                            offset=(-1 * bb[0], -1 * bb[1]))
            while holes.h_next() is not None:
                holes = holes.h_next()
                if holes is not None:
                    cv.DrawContours(mask, holes, 0, 255, 0, thickness=-1,
                                    offset=(-1 * bb[0], -1 * bb[1]))
        return mask

    def _get_hull_mask(self, hull, bb):
        """
        Return a mask of the convex hull of a blob.
        """
        bb = cv.BoundingRect(hull)
        mask = cv.CreateImage((bb[2], bb[3]), cv.IPL_DEPTH_8U, 1)
        cv.Zero(mask)
        cv.DrawContours(mask, hull, 255, 0, 0, thickness=-1,
                        offset=(-1 * bb[0], -1 * bb[1]))
        return mask

    def _get_avg(self, colorbitmap, bb, mask):
        """
        Calculate the average color of a blob given the mask.
        """
        cv.SetImageROI(colorbitmap, bb)
        # may need the offset parameter
        avg = cv.Avg(colorbitmap, mask)
        cv.ResetImageROI(colorbitmap)
        return avg

    def _get_blob_as_image(self, seq, bb, colorbitmap, mask):
        """
        Return an image that contains just pixels defined by the blob sequence.
        """
        cv.SetImageROI(colorbitmap, bb)
        outputImg = cv.CreateImage((bb[2], bb[3]), cv.IPL_DEPTH_8U, 3)
        cv.Zero(outputImg)
        cv.Copy(colorbitmap, outputImg, mask)
        cv.ResetImageROI(colorbitmap)
        return Image(outputImg)
