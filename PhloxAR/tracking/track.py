# -*- coding: utf-8 -*-
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from PhloxAR.color import Color
from PhloxAR.base import time, npy
from PhloxAR.features.feature import Feature, FeatureSet
import cv2


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
        img.drawText(text, pos[0], pos[1], color, size)
        img.drawText("in pixels/second", pos[0], pos[1] + size, color, size)

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

    def show_predicted_coordinates(self, pos=None, color=Color.GREEN, size=None):
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

    def show_corrected_coordinates(self, pos=None, color=Color.GREEN, size=None):
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

        np_pts = npy.asarray([kp.pt for kp in new_pts])
        t, pts, center = cv2.kmeans(npy.asarray(np_pts, dtype=npy.float32), K=1,
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
        img.drawText(text, pos[0], pos[1], color, size)
        img.drawText("in pixels/second", pos[0], pos[1] + size, color, size)
