# -*- coding: utf-8 -*-
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PhloxAR.color import Color
from PhloxAR.base import time, np, warnings, cv
from PhloxAR.features.feature import Feature, FeatureSet
from PhloxAR.image import Image
import cv2


__all__ = [
    'Track', 'TrackSet', 'CAMShiftTrack', 'LKTrack', 'MFTrack', 'SURFTrack'
]


class Track(Feature):
    """
    
    Track class is the base of tracking. All different tracking algorithm
    return different classes but they all belong to Track class. All the
    common attributes are kept in this class
    """

    def __init__(self, img, bbox):
        """
        Initializes all the required parameters and attributes of the class.
        **PARAMETERS**
        * *img* - Image
        * *bbox* - A tuple consisting of (x, y, w, h) of the bounding box
        **RETURNS**
        Tracking.TrackClass.Track object
        :Example:
        >>> tracking = Track(image, bbox)
        """
        self._bbox = bbox
        self._image = img
        self.bb_x, self.bb_y, self.w, self.h = self._bbox
        self._x, self._y = self._center = self.center
        self.sizeRatio = 1
        self.vel = (0, 0)
        self.rt_vel = (0, 0)
        self._area = self.area
        self._time = time.time()
        self._cvnarray = self._image.cvnarray
        self._predict_pts = None
        super(Track, self).__init__(img, self._x, self._y, None)

    @property
    def center(self):
        """
        
        Get the center of the bounding box
        **RETURNS**
        * *tuple* - center of the bounding box (x, y)
        :Example:
        >>> tracking = Track(img, bb)
        >>> cen = tracking.center
        """
        return self.bb_x + self.w / 2, self.bb_y + self.h / 2

    @property
    def area(self):
        """
        
        Get the area of the bounding box
        **RETURNS**
        Area of the bounding box
        :Example:
        >>> tracking = Track(img, bb)
        >>> area = tracking.area
        """
        return self.w * self.h

    @property
    def image(self):
        """
        
        Get the Image
        **RETURNS**
        Image
        :Example:
        >>> tracking = Track(img, bb)
        >>> i = tracking.image
        """
        return self._image

    @property
    def bbox(self):
        """
        
        Get the bounding box
        **RETURNS**
        A tuple  - (x, y, w, h)
        :Example:
        >>> tracking = Track(img, bb)
        >>> print(tracking.bbox)
        """
        return self._bbox

    def draw(self, color=Color.GREEN, rad=1, thickness=1):
        """
        
        Draw the center of the object on the image.
        **PARAMETERS**
        * *color* - The color to draw the object. Either an BGR tuple or a
        member of the :py:class:`Color` class.
        * *rad* - Radius of the circle to be plotted on the center of the object.
        * *thickness* - Thickness of the boundary of the center circle.
        **RETURNS**
        Nada. Nothing. Zilch.
        :Example:
        >>> tracking = Track(img, bb)
        >>> tracking.draw()
        >>> img.show()
        """
        f = self
        f.image.drawCircle(f.center, rad, color, thickness)

    def draw_bbox(self, color=Color.GREEN, thickness=3):
        """
        
        Draw the bounding box over the object on the image.
        **PARAMETERS**
        * *color* - The color to draw the object. Either an BGR tuple or a
        member of the :py:class:`Color` class.
        * *thickness* - Thickness of the boundary of the bounding box.
        **RETURNS**
        Nada. Nothing. Zilch.
        :Example:
        >>> tracking = Track(img, bb)
        >>> tracking.draw_bbox()
        >>> img.show()
        """
        f = self
        f.image.draw_rect(f.bb_x, f.bb_y, f.w, f.h, color, thickness)

    def show_coordinates(self, pos=None, color=Color.GREEN, size=None):
        """
        
        Show the co-ordinates of the object in text on the Image.
        **PARAMETERS**
        * *pos* - A tuple consisting of x, y values. where to put to the text
        * *color* - The color to draw the object. Either an BGR tuple or a
        member of the :py:class:`Color` class.
        * *size* - Fontsize of the text
        **RETURNS**
        Nada. Nothing. Zilch.
        :Example:
        >>> tracking = Track(img, bb)
        >>> tracking.show_coordinates()
        >>> img.show()
        """
        f = self
        img = f.image
        if not pos:
            img_size = img.size()
            pos = (img_size[0] - 120, 10)
        if not size:
            size = 16
        text = "x = %d  y = %d" % (f.x, f.y)
        img.draw_text(text, pos[0], pos[1], color, size)

    def show_size_ratio(self, pos=None, color=Color.GREEN, size=None):
        """
        
        Show the sizeRatio of the object in text on the image.
        **PARAMETERS**
        * *pos* - A tuple consisting of x, y values. where to put to the text
        * *color* - The color to draw the object. Either an BGR tuple or a
        member of the :py:class:`Color` class.
        * *size* - Fontsize of the text
        **RETURNS**
        Nada. Nothing. Zilch.
        :Example:
        >>> while True:
            ... img1 = cam.image
            ... ts = img1.tracking("camshift", ts1, img, bb)
            ... ts[-1].show_size_ratio() # For continuous bounding box
            ... img = img1
        """
        f = self
        img = f.image
        if not pos:
            img_size = img.size()
            pos = (img_size[0] - 120, 30)
        if not size:
            size = 16
        text = "size = %f" % f.sizeRatio
        img.draw_text(text, pos[0], pos[1], color, size)

    def show_pixel_velocity(self, pos=None, color=Color.GREEN, size=None):
        """
        
        Show the Pixel Velocity (pixel/frame) of the object in text on the image.
        **PARAMETERS**
        * *pos* - A tuple consisting of x, y values. where to put to the text
        * *color* - The color to draw the object. Either an BGR tuple or a
        member of the :py:class:`Color` class.
        * *size* - Fontsize of the text
        **RETURNS**
        Nada. Nothing. Zilch.
        :Example:
        >>> while True:
            ... img1 = cam.image
            ... ts = img1.tracking("camshift", ts1, img, bb)
            ... ts[-1].show_pixel_velocity() # For continuous bounding box
            ... img = img1
        """
        f = self
        img = f.image
        vel = f.vel
        if not pos:
            img_size = img.size()
            pos = (img_size[0] - 120, 90)
        if not size:
            size = 16
        text = "Vx = %.2f Vy = %.2f" % (vel[0], vel[1])
        img.draw_text(text, pos[0], pos[1], color, size)
        img.draw_text("in pixels/frame", pos[0], pos[1] + size, color, size)

    def show_pixel_velocity_rt(self, pos=None, color=Color.GREEN, size=None):
        """
        
        Show the Pixel Velocity (pixels/second) of the object in text on the
        image.
        **PARAMETERS**
        * *pos* - A tuple consisting of x, y values. where to put to the text
        * *color* - The color to draw the object. Either an BGR tuple or a
        member of the :py:class:`Color` class.
        * *size* - Fontsize of the text
        **RETURNS**
        Nada. Nothing. Zilch.
        :Example:
        >>> while True:
            ... img1 = cam.image
            ... ts = img1.tracking("camshift", ts1, img, bb)
            ... ts[-1].show_pixel_velocity_rt() # For continuous bounding box
            ... img = img1
        """
        f = self
        img = f.image
        vel_rt = f.rt_vel
        if not pos:
            img_size = img.size()
            pos = (img_size[0] - 120, 50)
        if not size:
            size = 16
        text = "Vx = %.2f Vy = %.2f" % (vel_rt[0], vel_rt[1])
        img.draw_text(text, pos[0], pos[1], color, size)
        img.draw_text("in pixels/second", pos[0], pos[1] + size, color, size)

    def process_track(self, func):
        """
        
        This method lets you use your own function on the current image.
        **PARAMETERS**
        * *func* - some user defined function for Image object
        **RETURNS**
        the value returned by the user defined function
        :Example:
        >>> def foo(img):
            ... return img.mean_color()
        >>> mean_color = ts[-1].process_track(foo)
        """
        return func(self._image)

    @property
    def prediction_points(self):
        """
        
        get predicted Co-ordinates of the center of the object
        **PARAMETERS**
        None
        **RETURNS**
        * *tuple*
        :Example:
        >>> tracking = Track(img, bb)
        >>> tracking.prediction_points()
        """
        return self._predict_pts

    def draw_predicted(self, color=Color.GREEN, rad=1, thickness=1):
        """
        
        Draw the center of the object on the image.
        **PARAMETERS**
        * *color* - The color to draw the object. Either an BGR tuple or a
                    member of the :py:class:`Color` class.
        * *rad* - Radius of the circle to be plotted on the center of the object.
        * *thickness* - Thickness of the boundary of the center circle.
        **RETURNS**
        Nada. Nothing. Zilch.
        :Example:
        >>> tracking = Track(img, bb)
        >>> tracking.draw_predicted()
        >>> img.show()
        """
        f = self
        f.image.draw_circle(f.prediction_points, rad, color, thickness)

    def show_predicted_coordinates(self, pos=None, color=Color.GREEN,
                                   size=None):
        """
        
        Show the co-ordinates of the object in text on the Image.
        **PARAMETERS**
        * *pos* - A tuple consisting of x, y values. where to put to the text
        * *color* - The color to draw the object. Either an BGR tuple or a
                    member of the :py:class:`Color` class.
        * *size* - Fontsize of the text
        **RETURNS**
        Nada. Nothing. Zilch.
        :Example:
        >>> tracking = Track(img, bb)
        >>> tracking.show_predicted_coordinates()
        >>> img.show()
        """
        f = self
        img = f.image
        if not pos:
            img_size = img.size()
            pos = (5, 10)
        if not size:
            size = 16
        text = "Predicted: x = %d  y = %d" % (f.prediction_points[0],
                                              f.prediction_points[1])
        img.draw_text(text, pos[0], pos[1], color, size)

    @property
    def corrected_points(self):
        """
        
        Corrected Co-ordinates of the center of the object
        **PARAMETERS**
        None
        **RETURNS**
        * *tuple*
        :Example:
        >>> tracking = Track(img, bb)
        >>> tracking.corrected_points
        """
        return self.state_pt

    def show_corrected_coordinates(self, pos=None, color=Color.GREEN,
                                   size=None):
        """
        
        Show the co-ordinates of the object in text on the Image.
        **PARAMETERS**
        * *pos* - A tuple consisting of x, y values. where to put to the text
        * *color* - The color to draw the object. Either an BGR tuple or a
                    member of the :py:class:`Color` class.
        * *size* - Fontsize of the text
        **RETURNS**
        Nada. Nothing. Zilch.
        :Example:
        >>> tracking = Track(img, bb)
        >>> tracking.show_corrected_coordinates()
        >>> img.show()
        """
        f = self
        img = f.image
        if not pos:
            img_size = img.size()
            pos = (5, 40)
        if not size:
            size = 16
        text = "Corrected: x = %d  y = %d" % (f.state_pt[0], f.state_pt[1])
        img.draw_text(text, pos[0], pos[1], color, size)

    def draw_corrected(self, color=Color.GREEN, rad=1, thickness=1):
        """
        
        Draw the center of the object on the image.
        **PARAMETERS**
        * *color* - The color to draw the object. Either an BGR tuple or a
                    member of the :py:class:`Color` class.
        * *rad* - Radius of the circle to be plotted on the center of the object.
        * *thickness* - Thickness of the boundary of the center circle.
        **RETURNS**
        Nada. Nothing. Zilch.
        :Example:
        >>> tracking = Track(img, bb)
        >>> tracking.draw_corrected()
        >>> img.show()
        """
        f = self
        f.image.drawCircle(f.state_pt, rad, color, thickness)


class CAMShiftTrack(Track):
    """
    
    CAMShift Class is returned by tracking when CAMShift tracking is required.
    This class is a superset of Track Class. And all of Track class'
    attributes can be accessed.
    CAMShift class has "ellipse" attribute which is not present in Track
    """

    def __init__(self, img, bbox, ellipse):
        """
        
        Initializes all the required parameters and attributes of the CAMShift
        class.
        **PARAMETERS**
        * *img* - Image
        * *bbox* - A tuple consisting of (x, y, w, h) of the bounding box
        * ellipse* - A tuple
        **RETURNS**
        Tracking.TrackClass.CAMShiftTrack object
        :Example:
        >>> tracking = CAMShiftTrack(image, bbox, ellipse)
        """
        self._ellipse = ellipse
        super(CAMShiftTrack, self).__init__(img, bbox)

    @property
    def ellipse(self):
        """
        
        Returns the ellipse.
        **RETURNS**
        A tuple
        :Example:
        >>> tracking = CAMShiftTrack(image, bb, ellipse)
        >>> e = tracking.ellipse
        """
        return self._ellipse


class LKTrack(Track):
    """
    
    LK Tracking class is used for Lucas-Kanade Track algorithm. It's
    derived from Track Class. Apart from all the properties of Track class,
    LK has few other properties. Since in LK tracking method, we obtain tracking
    points, we have functionalities to draw those points on the image.
    """

    def __init__(self, img, bbox, pts):
        """
        
        Initializes all the required parameters and attributes of the class.
        **PARAMETERS**
        * *img* - Image
        * *bbox* - A tuple consisting of (x, y, w, h) of the bounding box
        * *pts* - List of all the tracking points
        **RETURNS**
        Tracking.TrackClass.LKTrack object
        :Example:
        >>> tracking = LKTrack(image, bbox, pts)
        """
        self._track_pts = pts
        super(LKTrack, self).__init__(img, bbox)

    @property
    def tracked_points(self):
        """
        
        Returns all the points which are being tracked.
        **RETURNS**
        A list
        :Example:
        >>> tracking = LKTrack(image, bb, pts)
        >>> pts = tracking.tracked_points
        """
        return self._track_pts

    def draw_tracked_points(self, color=Color.GREEN, radius=1, thickness=1):
        """
        
        Draw all the points which are being tracked.
        **PARAMETERS**
        * *color* - Color of the point
        * *radius* - Radius of the point
        *thickness* - thickness of the circle point
        **RETURNS**
        Nothing
        :Example:
        >>> tracking = LKTrack(image, bb, pts)
        >>> tracking.draw_tracked_points()
        """
        if self._track_pts is not None:
            for pt in self._track_pts:
                self._image.draw_circle(ctr=pt, rad=radius, thickness=thickness,
                                        color=color)


class SURFTrack(Track):
    """
    
    SURFTracker class is used for SURF Based keypoints matching tracking
    algorithm. It's derived from Track Class. Apart from all the properties of
    Track class SURFTracker has few other properties.
    Matches keypoints from the template image and the current frame.
    flann based matcher is used to match the keypoints.
    Density based clustering is used classify points as in-region (of bounding
    box) and out-region points. Using in-region points, new bounding box is
    predicted using k-means.
    """

    def __init__(self, img, new_pts, detector, descriptor, template_img, skp,
                 sd, tkp, td):
        """
        
        Initializes all the required parameters and attributes of the class.
        **PARAMETERS**
        * *img* - Image
        * *new_pts* - List of all the tracking points found in the image. - list of cv2.KeyPoint
        * *detector* - SURF detector - cv2.FeatureDetector
        * *descriptor* - SURF descriptor - cv2.DescriptorExtractor
        * *template_img* - Template Image (First image) - Image
        * *skp* - image keypoints - list of cv2.KeyPoint
        * *sd* - image descriptor - numpy.ndarray
        * *tkp* - Template Imaeg keypoints - list of cv2.KeyPoint
        * *td* - Template image descriptor - numpy.ndarray
        **RETURNS**
        Tracking.TrackClass.SURFTrack object
        :Example:
        >>> tracking = SURFTracker(image, pts, detector, descriptor, temp, skp, sd, tkp, td)
        """
        if td is None:
            bb = (1, 1, 1, 1)
            super(SURFTrack, self).__init__(img, bb)
            return
        if len(new_pts) < 1:
            bb = (1, 1, 1, 1)
            super(SURFTrack, self).__init__(img, bb)
            self._track_pts = None
            self.templateImg = template_img
            self.skp = skp
            self.sd = sd
            self.tkp = tkp
            self.td = td
            self._detector = detector
            self._descriptor = descriptor
            return
        if sd is None:
            bb = (1, 1, 1, 1)
            super(SURFTrack, self).__init__(img, bb)
            self._track_pts = None
            self.templateImg = template_img
            self.skp = skp
            self.sd = sd
            self.tkp = tkp
            self.td = td
            self._detector = detector
            self._descriptor = descriptor
            return

        np_pts = np.asarray([kp.pt for kp in new_pts])
        t, pts, center = cv2.kmeans(np.asarray(np_pts, dtype=np.float32), K=1,
                                    bestLabels=None,
                                    criteria=(
                                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                                        1, 10), attempts=1,
                                    flags=cv2.KMEANS_RANDOM_CENTERS)
        max_x = int(max(np_pts[:, 0]))
        min_x = int(min(np_pts[:, 0]))
        max_y = int(max(np_pts[:, 1]))
        min_y = int(min(np_pts[:, 1]))

        bb = (min_x - 5, min_y - 5, max_x - min_x + 5, max_y - min_y + 5)

        super(SURFTrack, self).__init__(img, bb)
        self.templateImg = template_img
        self.skp = skp
        self.sd = sd
        self.tkp = tkp
        self.td = td
        self._track_pts = np_pts
        self._detector = detector
        self._descriptor = descriptor

    @property
    def tracked_points(self):
        """
        
        Returns all the points which are being tracked.
        **RETURNS**
        A list of points.
        :Example:
        >>> tracking = SURFTrack(image, pts, detector, descriptor, temp, skp, sd, tkp, td)
        >>> pts = tracking.tracked_points
        """
        return self._track_pts

    def draw_tracked_points(self, color=Color.GREEN, radius=1, thickness=1):
        """
        
        Draw all the points which are being tracked.
        **PARAMETERS**
        * *color* - Color of the point
        * *radius* - Radius of the point
        *thickness* - thickness of the circle point
        **RETURNS**
        Nothing
        :Example:
        >>> tracking = SURFTrack(image, pts, detector, descriptor, temp, skp, sd, tkp, td)
        >>> tracking.draw_tracked_points()
        """
        if self._track_pts is not None:
            for pt in self._track_pts:
                self._image.drawCircle(ctr=pt, rad=radius, thickness=thickness,
                                       color=color)

    @property
    def detector(self):
        """
        
        Returns SURF detector which is being used.
        **RETURNS**
        detector - cv2.Detctor
        :Example:
        >>> tracking = SURFTrack(image, pts, detector, descriptor, temp, skp, sd, tkp, td)
        >>> detector = tracking.detector
        """
        return self._detector

    @property
    def descriptor(self):
        """
        
        Returns SURF descriptor extractor which is being used.
        **RETURNS**
        detector - cv2.DescriptorExtractor
        :Example:
        >>> tracking = SURFTrack(image, pts, detector, descriptor, temp, skp, sd, tkp, td)
        >>> descriptor= tracking.descriptor
        """
        return self._descriptor

    @property
    def image_keypoints(self):
        """
        
        Returns all the keypoints which are found on the image.
        **RETURNS**
        A list of points.
        :Example:
        >>> tracking = SURFTrack(image, pts, detector, descriptor, temp, skp, sd, tkp, td)
        >>> skp = tracking.image_keypoints
        """
        return self.skp

    @property
    def image_descriptor(self):
        """
        
        Returns the image descriptor.
        **RETURNS**
        Image descriptor - numpy.ndarray
        :Example:
        >>> tracking = SURFTrack(image, pts, detector, descriptor, temp, skp, sd, tkp, td)
        >>> sd = tracking.image_descriptor
        """
        return self.sd

    @property
    def template_keypoints(self):
        """
        
        Returns all the keypoints which are found on the template Image.
        **RETURNS**
        A list of points.
        :Example:
        >>> tracking = SURFTrack(image, pts, detector, descriptor, temp, skp, sd, tkp, td)
        >>> tkp = tracking.template_keypoints
        """
        return self.tkp

    @property
    def template_descriptor(self):
        """
        
        Returns the template image descriptor.
        **RETURNS**
        Image descriptor - numpy.ndarray
        :Example:
        >>> tracking = SURFTrack(image, pts, detector, descriptor, temp, skp, sd, tkp, td)
        >>> td = tracking.template_descriptor()
        """
        return self.td

    @property
    def template_image(self):
        """
        
        Returns Template Image.
        **RETURNS**
        Template Image - Image
        :Example:
        >>> tracking = SURFTrack(image, pts, detector, descriptor, temp, skp, sd, tkp, td)
        >>> templateImg = tracking.template_image
        """
        return self.templateImg


class MFTrack(Track):
    """
    
    MFTracker class is used for Median Flow Tracking algorithm. It's
    derived from Track Class. Apart from all the properties of Track class,
    MFTracker has few other properties.
    Media Flow Tracker is the base tracker that is used in OpenTLD. It is based on
    Optical Flow. It calculates optical flow of the points in the bounding box from
    frame 1 to frame 2 and from frame 2 to frame 1 and using back tracking error, removes
    false positives. As the name suggests, it takes the median of the flow, and eliminates
    points.
    """

    def __init__(self, img, bbox, shift):
        """
        
        Initializes all the required parameters and attributes of the class.
        **PARAMETERS**
        * *img* - Image
        * *bbox* - A tuple consisting of (x, y, w, h) of the bounding box
        * *shift* - Object Shift calcluated in Median Flow
        **RETURNS**
        Tracking.TrackClass.MFTrack object
        :Example:
        >>> tracking = MFTrack(image, bbox, shift)
        """
        super(MFTrack, self).__init__(img, bbox)
        self._shift = shift

    @property
    def shift(self):
        """
        
        Returns object shift that was calcluated in Median Flow.
        **RETURNS**
        float
        :Example:
        >>> tracking = MFTrack(image, bb, pts)
        >>> pts = tracking.shift
        """
        return self._shift

    def show_shift(self, pos=None, color=Color.GREEN, size=None):
        """
        
        Show the Pixel Velocity (pixels/second) of the object in text on the image.
        **PARAMETERS**
        * *pos* - A tuple consisting of x, y values. where to put to the text
        * *color* - The color to draw the object. Either an BGR tuple or a member of the :py:class:`Color` class.
        * *size* - Fontsize of the text
        **RETURNS**
        Nada. Nothing. Zilch.
        :Example:
        >>> ts = []
        >>> while True:
            ... img1 = cam.image
            ... ts = img1.tracking("mftrack", ts, img, bb)
            ... ts[-1].show_shift()
            ... img1.show()
        """
        f = self
        img = f.image
        shift = f.shift
        if not pos:
            img_size = img.size()
            pos = (img_size[0] - 120, 50)
        if not size:
            size = 16
        text = "Shift = %.2f" % shift
        img.draw_text(text, pos[0], pos[1], color, size)
        img.draw_text("in pixels/second", pos[0], pos[1] + size, color, size)


class TrackSet(FeatureSet):
    """
    **SUMMARY**
    TrackSet is a class extended from FeatureSet which is a class
    extended from Python's list. So, TrackSet has all the properties
    of a list as well as all the properties of FeatureSet.
    In general, functions dealing with attributes will return
    numpy arrays.
    This class is specifically made for Tracking.
    **EXAMPLE**
    >>> image = Image("/path/to/image.png")
    >>> ts = image.track("camshift", img1=image, bb)  #ts is the track set
    >>> ts.draw()
    >>> ts.x()
    """
    try:
        import cv2
    except ImportError:
        warnings.warn("OpenCV >= 2.3.1 required.")

    def __init__(self):
        self._kalman = None
        self.predict_pt = (0, 0)
        self._kalman()
        super(TrackSet, self).__init__()

    def append(self, f):
        """
        **SUMMARY**
        This is a substitute function for append. This is used in
        Image.track(). To get z, vel, etc I have to use this.
        This sets few parameters up and appends Tracking object to
        TrackSet list.
        Users are discouraged to use this function.
        **RETURNS**
            Nothing.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... img = img1
        >>> ts.append(CAMShift(img,bb,ellipse))
        """
        list.append(self, f)
        ts = self
        if ts[0].area <= 0:
            return
        f.sizeRatio = float(ts[-1].area) / float(ts[0].area)
        f.vel = self._pixel_velocity()
        f.rt_vel = self._pixel_velocity_real_time()
        self._set_kalman()
        self._predict_kalman()
        self._change_measure()
        self._correct_kalman()
        f.predict_pt = self.predict_pt
        f.state_pt = self.state_pt

    # Issue #256 - (Bug) Memory management issue due to too many number of images.
    def trim_list(self, num):
        """
        **SUMMARY**
        Trims the TrackSet(lists of all the saved objects) to save memory. It is implemented in
        Image.track() by default, but if you want to trim the list manually, use this.
        **RETURNS**
        Nothing.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... if len(ts) > 30:
                ... ts.trim_list(10)
            ... img = img1
        """
        ts = self
        for i in range(num):
            ts.pop(0)

    @property
    def area_ratio(self):
        """
        **SUMMARY**
        Returns a numpy array of the area_ratio of each feature.
        where area_ratio is the ratio of the size of the current bounding box to
        the size of the initial bounding box
        **RETURNS**
        A numpy array.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... img = img1
        >>> print(ts.area_ratio)
        """
        return np.array([f.area_ratio for f in self])

    def draw_path(self, color=Color.GREEN, thickness=2):
        """
        **SUMMARY**
        Draw the complete path traced by the center of the object on current frame
        **PARAMETERS**
        * *color* - The color to draw the object. Either an BGR tuple or a member of the :py:class:`Color` class.
        * *thickness* - Thickness of the tracing path.
        **RETURNS**
        Nada. Nothing. Zilch.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... ts.draw_path() # For continuous tracing
            ... img = img1
        >>> ts.draw_path() # draw the path at the end of tracking
        """

        ts = self
        img = self[-1].image
        for i in range(len(ts) - 1):
            img.drawLine((ts[i].center), (ts[i + 1].center), color=color,
                         thickness=thickness)

    def draw(self, color=Color.GREEN, rad=1, thickness=1):
        """
        **SUMMARY**
        Draw the center of the object on the current frame.
        **PARAMETERS**
        * *color* - The color to draw the object. Either an BGR tuple or a member of the :py:class:`Color` class.
        * *rad* - Radius of the circle to be plotted on the center of the object.
        * *thickness* - Thickness of the boundary of the center circle.
        **RETURNS**
        Nada. Nothing. Zilch.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... ts.draw() # For continuous tracking of the center
            ... img = img1
        """
        f = self[-1]
        f.image.draw_circle(f.center, rad, color, thickness)

    def draw_bbox(self, color=Color.GREEN, thickness=3):
        """
        **SUMMARY**
        Draw the bounding box over the object on the current frame.
        **PARAMETERS**
        * *color* - The color to draw the object. Either an BGR tuple or a member of the :py:class:`Color` class.
        * *thickness* - Thickness of the boundary of the bounding box.
        **RETURNS**
        Nada. Nothing. Zilch.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... ts.draw_bbox() # For continuous bounding box
            ... img = img1
        """
        f = self[-1]
        f.image.draw_rect(f.bb_x, f.bb_y, f.w, f.h, color, thickness)

    def track_length(self):
        """
        **SUMMARY**
        Get total number of tracked frames.
        **PARAMETERS**
        No Parameters required.
        **RETURNS**
        * *int* * -Number of tracked image frames
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... img = img1
        >>> print(ts.track_length())
        """
        return len(self)

    def track_images(self, cv2_numpy=False):
        """
        **SUMMARY**
        Get all the tracked images in a list
        **PARAMETERS**
        No Parameters required.
        **RETURNS**
        * *list* * - A list of all the tracked Image
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... img = img1
        >>> imgset = ts.track_images()
        """
        if cv2_numpy:
            return [f.cv2numpy for f in self]
        return [f.image for f in self]

    def bbox_track(self):
        """
        **SUMMARY**
        Get all the bounding box in a list
        **PARAMETERS**
        No Parameters required.
        **RETURNS**
        * *list* * - All the bounding box co-ordinates in a list
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... img = img1
        >>> print(ts.bbox_track())
        """
        return [f.bb for f in self]

    def _pixel_velocity(self):
        """
        **SUMMARY**
        Get Pixel Velocity of the tracked object in pixel/frame.
        **PARAMETERS**
        No Parameters required.
        **RETURNS**
        * *tuple* * - (Velocity of x, Velocity of y)
        """
        ts = self
        if len(ts) < 2:
            return 0, 0
        dx = ts[-1].x - ts[-2].x
        dy = ts[-1].y - ts[-2].y
        return dx, dy

    def pixel_velocity(self):
        """
        **SUMMARY**
        Get each Pixel Velocity of the tracked object in pixel/frames.
        **PARAMETERS**
        No Parameters required.
        **RETURNS**
        * *numpy array* * - array of pixel velocity tuple.
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... img = img1
        >>> print(ts.pixel_velocity())
        """
        return np.array([f.vel for f in self])

    def _pixel_velocity_real_time(self):
        """
        **SUMMARY**
        Get each Pixel Velocity of the tracked object in pixel/second.
        **PARAMETERS**
        No Parameters required.
        **RETURNS**
        * *tuple* * - velocity tuple
        """
        ts = self
        if len(ts) < 2:
            return (0, 0)
        dx = ts[-1].x - ts[-2].x
        dy = ts[-1].y - ts[-2].y
        dt = ts[-1].time - ts[-2].time
        return float(dx) / dt, float(dy) / dt

    def pixel_velocity_real_time(self):
        """
        **SUMMARY**
        Get each Pixel Velocity of the tracked object in pixel/frames.
        **PARAMETERS**
        No Parameters required.
        **RETURNS**
        * *numpy array* * - array of pixel velocity tuple.
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... img = img1
        >>> print(ts.pixel_velocity_real_time())
        """
        return np.array([f.rt_vel for f in self])

    def show_coordinates(self, pos=None, color=Color.GREEN, size=None):
        """
        **SUMMARY**
        Show the co-ordinates of the object in text on the current frame.
        **PARAMETERS**
        * *pos* - A tuple consisting of x, y values. where to put to the text
        * *color* - The color to draw the object. Either an BGR tuple or a member of the :py:class:`Color` class.
        * *size* - Fontsize of the text
        **RETURNS**
        Nada. Nothing. Zilch.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... ts.show_coordinates() # For continuous bounding box
            ... img = img1
        """
        ts = self
        f = ts[-1]
        img = f.image
        if not pos:
            img_size = img.size()
            pos = (img_size[0] - 120, 10)
        if not size:
            size = 16
        text = "x = %d  y = %d" % (f.x, f.y)
        img.draw_text(text, pos[0], pos[1], color, size)

    def show_size_ratio(self, pos=None, color=Color.GREEN, size=None):
        """
        **SUMMARY**
        Show the sizeRatio of the object in text on the current frame.
        **PARAMETERS**
        * *pos* - A tuple consisting of x, y values. where to put to the text
        * *color* - The color to draw the object. Either an BGR tuple or a member of the :py:class:`Color` class.
        * *size* - Fontsize of the text
        **RETURNS**
        Nada. Nothing. Zilch.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... ts.showZ() # For continuous bounding box
            ... img = img1
        """
        ts = self
        f = ts[-1]
        img = f.image
        if not pos:
            img_size = img.size()
            pos = (img_size[0] - 120, 30)
        if not size:
            size = 16
        text = "size = %f" % (f.sizeRatio)
        img.draw_text(text, pos[0], pos[1], color, size)

    def show_pixel_velocity(self, pos=None, color=Color.GREEN, size=None):
        """
        **SUMMARY**
        show the Pixel Velocity (pixel/frame) of the object in text on the current frame.
        **PARAMETERS**
        * *pos* - A tuple consisting of x, y values. where to put to the text
        * *color* - The color to draw the object. Either an BGR tuple or a member of the :py:class:`Color` class.
        * *size* - Fontsize of the text
        **RETURNS**
        Nada. Nothing. Zilch.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... ts.show_pixel_velocity() # For continuous bounding box
            ... img = img1
        """
        ts = self
        f = ts[-1]
        img = f.image
        vel = f.vel
        if not pos:
            img_size = img.size()
            pos = (img_size[0] - 120, 50)
        if not size:
            size = 16
        text = "Vx = %.2f Vy = %.2f" % (vel[0], vel[1])
        img.draw_text(text, pos[0], pos[1], color, size)
        img.draw_text("in pixels/frame", pos[0], pos[1] + size, color, size)

    def show_pixel_velocity_real_time(self, pos=None, color=Color.GREEN,
                                      size=None):
        """
        **SUMMARY**
        show the Pixel Velocity (pixels/second) of the object in text on the current frame.
        **PARAMETERS**
        * *pos* - A tuple consisting of x, y values. where to put to the text
        * *color* - The color to draw the object. Either an BGR tuple or a member of the :py:class:`Color` class.
        * *size* - Fontsize of the text
        **RETURNS**
        Nada. Nothing. Zilch.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... ts.show_pixel_velocity_real_time() # For continuous bounding box
            ... img = img1
        """
        ts = self
        f = ts[-1]
        img = f.image
        vel_rt = f.rt_vel
        if not pos:
            img_size = img.size()
            pos = (img_size[0] - 120, 90)
        if not size:
            size = 16
        text = "Vx = %.2f Vy = %.2f" % (vel_rt[0], vel_rt[1])
        img.draw_text(text, pos[0], pos[1], color, size)
        img.draw_text("in pixels/second", pos[0], pos[1] + size, color, size)

    def process_track(self, func):
        """
        **SUMMARY**
        This method lets you use your own function on the entire imageset.
        **PARAMETERS**
        * *func* - some user defined function for Image object
        **RETURNS**
        * *list* - list of the values returned by the function when applied on all the images
        **EXAMPLE**
        >>> def foo(img):
            ... return img.meanColor()
        >>> mean_color_list = ts.process_track(foo)
        """
        return [func(f.image) for f in self]

    @property
    def background(self):
        """
        **SUMMARY**
        Get Background of the Image. For more info read
        http://opencvpython.blogspot.in/2012/07/background-extraction-using-running.html
        **PARAMETERS**
        No Parameters
        **RETURNS**
        Image - Image
        **EXAMPLE**
        >>> while some_condition:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... img = img1
        >>> ts.background.show()
        """
        imgs = self.track_images(cv2_numpy=True)
        f = imgs[0]
        avg = np.float32(f)
        for img in imgs[1:]:
            f = img
            cv2.accumulateWeighted(f, avg, 0.01)
            res = cv2.convertScaleAbs(avg)
        return Image(res, cv2image=True)

    def _kalman(self):
        self._kalman = cv.CreateKalman(4, 2, 0)
        self.kalman_state = cv.CreateMat(4, 1, cv.CV_32FC1)  # (phi, delta_phi)
        self.kalman_process_noise = cv.CreateMat(4, 1, cv.CV_32FC1)
        self.kalman_measurement = cv.CreateMat(2, 1, cv.CV_32FC1)

    def _set_kalman(self):
        ts = self
        if len(ts) < 2:
            self.kalman_x = ts[-1].x
            self.kalman_y = ts[-1].y
        else:
            self.kalman_x = ts[-2].x
            self.kalman_y = ts[-2].y

        self._kalman.state_pre[0, 0] = self.kalman_x
        self._kalman.state_pre[1, 0] = self.kalman_y
        self._kalman.state_pre[2, 0] = self.predict_pt[0]
        self._kalman.state_pre[3, 0] = self.predict_pt[1]

        self._kalman.transition_matrix[0, 0] = 1
        self._kalman.transition_matrix[0, 1] = 0
        self._kalman.transition_matrix[0, 2] = 1
        self._kalman.transition_matrix[0, 3] = 0
        self._kalman.transition_matrix[1, 0] = 0
        self._kalman.transition_matrix[1, 1] = 1
        self._kalman.transition_matrix[1, 2] = 0
        self._kalman.transition_matrix[1, 3] = 1
        self._kalman.transition_matrix[2, 0] = 0
        self._kalman.transition_matrix[2, 1] = 0
        self._kalman.transition_matrix[2, 2] = 1
        self._kalman.transition_matrix[2, 3] = 0
        self._kalman.transition_matrix[3, 0] = 0
        self._kalman.transition_matrix[3, 1] = 0
        self._kalman.transition_matrix[3, 2] = 0
        self._kalman.transition_matrix[3, 3] = 1

        cv.SetIdentity(self._kalman.measurement_matrix, cv.RealScalar(1))
        cv.SetIdentity(self._kalman.process_noise_cov, cv.RealScalar(1e-5))
        cv.SetIdentity(self._kalman.measurement_noise_cov, cv.RealScalar(1e-1))
        cv.SetIdentity(self._kalman.error_cov_post, cv.RealScalar(1))

    def _predict_kalman(self):
        self.kalman_prediction = cv.KalmanPredict(self._kalman)
        self.predict_pt = (
            self.kalman_prediction[0, 0], self.kalman_prediction[1, 0])

    def _correct_kalman(self):
        self.kalman_estimated = cv.KalmanCorrect(self._kalman,
                                                 self.kalman_measurement)
        self.state_pt = (
            self.kalman_estimated[0, 0], self.kalman_estimated[1, 0])

    def _change_measure(self):
        ts = self
        self.kalman_measurement[0, 0] = ts[-1].x
        self.kalman_measurement[1, 0] = ts[-1].y

    def predicted_coordinates(self):
        """
        **SUMMARY**
        Returns a numpy array of the predicted coordinates of each feature.
        **RETURNS**
        A numpy array.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... img = img1
        >>> print ts.predicted_coordinates()
        """
        return np.array([f.predict_pt for f in self])

    def predict_x(self):
        """
        **SUMMARY**
        Returns a numpy array of the predicted x (vertical) coordinate of each feature.
        **RETURNS**
        A numpy array.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... img = img1
        >>> print(ts.predict_x())
        """
        return np.array([f.predict_pt[0] for f in self])

    def predict_y(self):
        """
        **SUMMARY**
        Returns a numpy array of the predicted y (vertical) coordinate of each feature.
        **RETURNS**
        A numpy array.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... img = img1
        >>> print(ts.predict_y())
        """
        return np.array([f.predict_pt[1] for f in self])

    def draw_predicted(self, color=Color.GREEN, rad=1, thickness=1):
        """
        **SUMMARY**
        Draw the predcited center of the object on the current frame.
        **PARAMETERS**
        * *color* - The color to draw the object. Either an BGR tuple or a member of the :py:class:`Color` class.
        * *rad* - Radius of the circle to be plotted on the center of the object.
        * *thickness* - Thickness of the boundary of the center circle.
        **RETURNS**
        Nada. Nothing. Zilch.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... ts.draw_predicted() # For continuous tracking of the center
            ... img = img1
        """
        f = self[-1]
        f.image.drawCircle(f.predict_pt, rad, color, thickness)

    def draw_corrected(self, color=Color.GREEN, rad=1, thickness=1):
        """
        **SUMMARY**
        Draw the predcited center of the object on the current frame.
        **PARAMETERS**
        * *color* - The color to draw the object. Either an BGR tuple or a member of the :py:class:`Color` class.
        * *rad* - Radius of the circle to be plotted on the center of the object.
        * *thickness* - Thickness of the boundary of the center circle.
        **RETURNS**
        Nada. Nothing. Zilch.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... ts.draw_predicted() # For continuous tracking of the center
            ... img = img1
        """
        f = self[-1]
        f.image.drawCircle(f.state_pt, rad, color, thickness)

    def draw_predicted_path(self, color=Color.GREEN, thickness=2):
        """
        **SUMMARY**
        Draw the complete predicted path of the center of the object on current frame
        **PARAMETERS**
        * *color* - The color to draw the object. Either an BGR tuple or a member of the :py:class:`Color` class.
        * *thickness* - Thickness of the tracing path.
        **RETURNS**
        Nada. Nothing. Zilch.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... ts.draw_predicted_path() # For continuous tracing
            ... img = img1
        >>> ts.draw_predicted_path() # draw the path at the end of tracking
        """

        ts = self
        img = self[-1].image
        for i in range(1, len(ts) - 1):
            img.drawLine((ts[i].predict_pt), (ts[i + 1].predict_pt),
                         color=color, thickness=thickness)

    def show_predicted_coordinates(self, pos=None, color=Color.GREEN, size=None):
        """
        **SUMMARY**
        Show the co-ordinates of the object in text on the current frame.
        **PARAMETERS**
        * *pos* - A tuple consisting of x, y values. where to put to the text
        * *color* - The color to draw the object. Either an BGR tuple or a member of the :py:class:`Color` class.
        * *size* - Fontsize of the text
        **RETURNS**
        Nada. Nothing. Zilch.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... ts.show_predicted_coordinates() # For continuous bounding box
            ... img = img1
        """
        ts = self
        f = ts[-1]
        img = f.image
        if not pos:
            img_size = img.size()
            pos = (5, 10)
        if not size:
            size = 16
        text = "Predicted: x = %d  y = %d" % (f.predict_pt[0], f.predict_pt[1])
        img.draw_text(text, pos[0], pos[1], color, size)

    def show_corrected_coordinates(self, pos=None, color=Color.GREEN, size=None):
        """
        **SUMMARY**
        Show the co-ordinates of the object in text on the current frame.
        **PARAMETERS**
        * *pos* - A tuple consisting of x, y values. where to put to the text
        * *color* - The color to draw the object. Either an BGR tuple or a member of the :py:class:`Color` class.
        * *size* - Fontsize of the text
        **RETURNS**
        Nada. Nothing. Zilch.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... ts.show_corrected_coordinates() # For continuous bounding box
            ... img = img1
        """
        ts = self
        f = ts[-1]
        img = f.image
        if not pos:
            img_size = img.size()
            pos = (5, 40)
        if not size:
            size = 16
        text = "Corrected: x = %d  y = %d" % (f.state_pt[0], f.state_pt[1])
        img.draw_text(text, pos[0], pos[1], color, size)

    def correct_x(self):
        """
        **SUMMARY**
        Returns a numpy array of the corrected x coordinate of each feature.
        **RETURNS**
        A numpy array.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... img = img1
        >>> print(ts.correct_x())
        """
        return np.array([f.state_pt[0] for f in self])

    def correct_y(self):
        """
        **SUMMARY**
        Returns a numpy array of the corrected y coordinate of each feature.
        **RETURNS**
        A numpy array.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... img = img1
        >>> print(ts.correct_y())
        """
        return np.array([f.state_pt[1] for f in self])

    def corrected_coordinates(self):
        """
        **SUMMARY**
        Returns a numpy array of the corrected coordinates of each feature.
        **RETURNS**
        A numpy array.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... img = img1
        >>> print ts.predicted_coordinates()
        """
        return np.array([f.state_pt for f in self])

    def draw_corrected_path(self, color=Color.GREEN, thickness=2):
        """
        **SUMMARY**
        Draw the complete corrected path of the center of the object on current frame
        **PARAMETERS**
        * *color* - The color to draw the object. Either an BGR tuple or a member of the :py:class:`Color` class.
        * *thickness* - Thickness of the tracing path.
        **RETURNS**
        Nada. Nothing. Zilch.
        **EXAMPLE**
        >>> while True:
            ... img1 = cam.getImage()
            ... ts = img1.track("camshift", ts1, img, bb)
            ... ts.draw_corrected_path() # For continuous tracing
            ... img = img1
        >>> ts.draw_predicted_path() # draw the path at the end of tracking
        """

        ts = self
        img = self[-1].image
        for i in range(len(ts) - 1):
            img.draw_line(ts[i].state_pt, ts[i + 1].state_pt, color=color,
                          thickness=thickness)
