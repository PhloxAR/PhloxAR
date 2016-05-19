# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals
from __future__ import division, print_function

from .image import Image
import os
import pygame
import numpy as np

PYGAME_INITIALIZED = False


class Display(object):
    """
    WindowsStream opens a window (Pygame Display Surface) to which you
    can write images. The default resolution is (640, 480) but you can
    also specify (0, 0) which will maximize the display. Flags are
    pygame constants, including:

    By default display will attempt to scale the input image to fit neatly
    on the screen with minimal distortion. This means that if the aspect
    ratio matches the screen it will scale cleanly. If your image does not
    match the screen aspect ratio we will scale it to fit nicely while
    maintaining its natural aspect ratio. Because PhloxAR performs this
    scaling there are two sets of input mouse coordinates, the
    (mouse_x, mouse_y) which scale to the image, and (mouse_raw_x, mouse_raw_y)
    which do are the actual screen coordinates.

    pygame.FULLSCREEN: create a fullscreen display.
    pygame.DOUBLEBUF: recommended for HWSURFACE or OPENGL.
    pygame.HWSURFACE: hardware accelerated, only in FULLSCREEN.
    pygame.OPENGL: create an opengl renderable display.
    pygame.RESIZABLE: display window should be sizeable.
    pygame.NOFRAME: display window will have no border or controls.

    Display should be used in a while loop with the isDone() method,
    which checks events and sets the following internal state controls:

    mouse_x: the x position of the mouse cursor on the input image.
    mouse_y: the y position of the mouse cursor on the input image.
    mouse_raw_x: The position of the mouse on the screen.
    mouse_raw_y: The position of the mouse on the screen.

    Note:
    The mouse position on the screen is not the mouse position on the
    image. If you are trying to draw on the image or take in coordinates
    use mouse_x and mouse_y as these values are scaled along with the image.

    mouse_l: the state of the left button.
    mouse_r: the state of the right button.
    mouse_m: the state of the middle button.
    mouse_wheel_u: scroll wheel has been moved up.
    mouse_wheel_d: the wheel has been clicked towards the bottom of the mouse.
    """
    res = ''
    src_res = ''
    src_offset = ''
    screen = ''
    event_handler = ''
    mq = ''
    done = False
    mouse_x = 0
    mouse_y = 0
    # actual (x, y) position on the screen
    mouse_raw_x = 0
    mouse_raw_y = 0
    mouse_l = 0
    mouse_r = 0
    mouse_wheel_u = 0
    mouse_wheel_d = 0
    scale_x = 1.0
    scale_y = 1.0
    offset_x = 0
    offset_y = 0
    img_w = 0
    img_h = 0
    # lb for last left button & rb for right button
    last_lb = 0
    last_rb = 0
    lb_down = None
    lb_up = None
    rb_down = None
    rb_up = None
    display_type = None
    do_clamp = None
    pressed = []

    def __init__(self, res=(640, 480), flags=0, title='PhloxAR',
                 disptype='standard', headless=False):
        """
        This is the generic display object. You are able to set the
        display type.
        The standard display type will pop up a window.
        The notebook display type is to be used in conjunction with
        IPython Notebooks. If you have IPython Notebooks installed you
        just need to start IPython Notebooks an open in your browser.
        :param res: the size of the display in pixels
        :param flags: pygame flags
        :param title: the title bar on the display
        :param disptype: type of display. Options are as follows:
                          'standard': a pygame window
                          'notebook': IPython web notebook output.
        :param headless: if False we ignore headless mode. If True, all
                          rendering is suspended.
        """
        global PYGAME_INITIALIZED

        if headless:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'

        if not PYGAME_INITIALIZED:
            if not disptype == 'notebook':
                pygame.init()
            PYGAME_INITIALIZED = True

        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.last_lb = 0
        self.last_rb = 0
        self.lb_down = 0
        self.rb_down = 0
        self.lb_up = 0
        self.rb_up = 0
        self.pressed = None
        self.display_type = disptype
        self.mouse_raw_x = 0
        self.mouse_raw_y = 0
        self.res = res
        self.do_clamp = False

        if not disptype == 'notebook':
            self.screen = pygame.display.set_mode(res, flags)

        # checks if phloxar.png exists
        if os.path.isfile(os.path.join(LAUNCH_PATH, 'sample_images', 'phloxar.png')):
            plxlogo = Image('phloxar').scale(32, 32)
            pygame.display.set_icon(plxlogo.surface())

        if flags != pygame.FULLSCREEN and flags != pygame.NOFRAME:
            pygame.display.set_caption(title)

    def left_button_up_pos(self):
        """
        Returns the position where the left mouse button go up.
        :return: an (x, y) mouse position tuple.

        Note:
        You must call 'check_events' or 'is_done' in you main display loop
        for this method to work.
        """
        return self.lb_up

    def left_button_down_pos(self):
        """
        Returns the position where the left mouse button go down.
        :return: an (x, y) mouse position tuple.

        Note:
        You must call 'check_events' or 'is_done' in you main display loop
        for this method to work.
        """
        return self.lb_down

    def right_button_up_pos(self):
        """
        Returns the position where the right mouse button go up.
        :return: an (x, y) mouse position tuple.

        Note:
        You must call 'check_events' or 'is_done' in you main display loop
        for this method to work.
        """
        return self.rb_up

    def right_button_down_pos(self):
        """
        Returns the position where the right mouse button go down.
        :return: an (x, y) mouse position tuple.

        Note:
        You must call 'check_events' or 'is_done' in you main display loop
        for this method to work.
        """
        return self.rb_down

    def points2boundingbox(self, pt0, pt1):
        """
        Given two screen coordinates return the bounding box in x, y, w, h
        format. This is helpful for drawing regions on the display.
        :param pt0: first points
        :param pt1: second points
        :return: (x, y, w, h) tuple
        """
        max_x = np.max((pt0[0], pt1[0]))
        max_y = np.max((pt0[1], pt1[1]))
        min_x = np.min((pt0[0], pt1[0]))
        min_y = np.min((pt0[1], pt1[1]))

        return min_x, min_y, max_x-min_x, max_y-min_y

    def write_frame(self, img, fit=True):
        """
        Copies the given Image object to the display, you can also use
        Image.save()

        Write frame try to fit the image to the display with the minimum
        amount of distortion possible. When fit=True write frame will decide
        how to scale the image such that aspect ratio is maintained and the
        smallest amount of distortion possible is completed. This means the
        axis that has the minimum scaling needed will be shrunk or enlarged
        to match the display.
        :param img: the PhloxAR Image to save to the display
        :param fit: if False, write frame will crop and center the image
                     as best it can. If the image is too big it is cropped
                     and centered. If it is too small it is centered. If
                     it is too big along one axis that axis is cropped and
                     the other axis is centered if necessary.
        :return: None
        """
        wnd_ratio = self.res[0] / self.res[1]
        img_ratio = img.width / img.height
        self.src_res = img.size()
        self.img_w = img.width
        self.img_h = img.height
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0

        if img.size() == self.res:
            s = img.surface()
            self.screen.blit(s, s.get_rect())
            pygame.display.flip()
        elif img_ratio == wnd_ratio:
            self.scale_x = img.width / self.res[0]
            self.scale_y = img.height / self.res[1]
            img = img.scale(self.res[0], self.res[1])
            s = img.surface()
            self.screen.blit(s, s.get_rect())
            pygame.display.flip()
        elif fit:
            # scale factors
            wscale = img.width / self.res[0]
            hscale = img.height / self.res[1]
            w = img.width
            h = img.height
            # shrink what is the percent reduction
            if wscale > 1:
                wscale = 1.0 - (1 / wscale)
            else:
                # grow the image by a percentage
                wscale = 1.0 - wscale

            if hscale > 1:
                hscale = 1.0 - (1 / hscale)
            else:
                hscale =  1.0 - hscale

            if wscale == 0:
                x = 0
                y = (self.res[1] - img.height) / 2
                w = img.width
                h = img.height
                s = img.surface()
            elif hscale == 0:
                x = (self.res[0] - img.width) / 2
                y = 0
                w = img.width
                h = img.height
                s = img.surface()
            elif wscale < hscale:
                # width has less distortion
                sfactor = self.res[0] / img.width
                w = int(img.width * sfactor)
                h = int(img.height * sfactor)

                if w > self.res[0] or h > self.res[1]:
                    sfactor = self.res[1] / img.heigt
                    w = int(img.width * sfactor)
                    h = int(img.height * sfactor)
                    x = (self.res[0] - w) / 2
                    y = 0
                else:
                    x = 0
                    y = (self.res[1] - h) / 2

                img = img.scale(w, h)
                s = img.surface()
            else:
                # the height has more distortion
                sfactor = self.res[1] / img.height
                w = int(img.width * sfactor)
                h = int(img.height * sfactor)
                if w > self.res[0] or h > self.res[1]:
                    # aw shucks that still didn't work do the other way instead
                    sfactor = self.res[0] / img.width
                    w = int(img.width * sfactor)
                    h = int(img.height * sfactor)
                    x = 0
                    y = (self.res[1] - h) / 2
                else:
                    x = (self.res[0] - w) / 2
                    y = 0
                img = img.scale(w, h)
                s = img.surface()
            # clear out the screen so everything is clean
            black = pygame.Surface((self.res[0], self.res[1]))
            black.fill((0, 0, 0))
            self.screen.blit(black, black.get_rect())
            self.screen.blit(s, (x, y))
            self.src_offset = (x, y)
            pygame.display.flip()
            self.offset_x = x
            self.offset_y = y
            self.scale_x = self.img_w / w
            self.scale_y = self.img_h / h
        else:
            # crop
            self.do_clamp = False
            x = y = corner_x = corner_y = 0

            # center a too small image
            if img.width <= self.res[0] and img.height <= self.res[1]:
                # just center the too small image
                x = self.res[0] / 2 - img.width / 2
                y = self.res[1] / 2 - img.height / 2
                corner_x = x
                corner_y = y
                s = img.surface()
            elif img.width > self.res[0] and img.height < self.res[1]:
                # crop the too big image on both axes
                w = self.res[0]
                h = self.res[1]
                x = 0
                y = 0
                xx = (img.width - self.res[0]) / 2
                yy = (img.height - self.res[1]) / 2
                corner_x = -1 * xx
                corner_y = -1 * yy
                img = img.crop(xx, yy, w, h)
                s = img.surface()
            elif img.width < self.res[0] and img.height >= self.res[1]:
                # height too big
                # crop along the y dimension and center along the x dimension
                w = img.width
                h = self.res[1]
                x = (self.res[0] - img.width) / 2
                y = 0
                xx = 0
                yy = (img.height - self.res[1]) / 2
                corner_x = x
                corner_y = -1 * yy
                img = img.crop(xx, yy, w, h)
                s = img.surface()
            elif img.width > self.res[0] and img.height <= self.res[1]:
                # width too big
                # crop along the y dimension and center along the x dimension
                w = self.res[0]
                h = img.height
                x = 0
                y = (self.res[1] - img.height) / 2
                xx = (img.width - self.res[0]) / 2
                yy = 0
                corner_x = -1 * xx
                corner_y = y
                img = img.crop(xx, yy, w, h)
                s = img.getPGSurface()
            self.offset_x = corner_x
            self.offset_y = corner_y
            black = pygame.Surface((self.res[0], self.res[1]))
            black.fill((0, 0, 0))
            self.screen.blit(black, black.get_rect())
            self.screen.blit(s, (x, y))
            pygame.display.flip()

    def _set_button_state(self, state, button):
        if button == 1:
            self.mouse_l = state
        if button == 2:
            self.mouse_m = state
        if button == 3:
            self.mouse_r = state
        if button == 4:
            self.mouse_wheel_u = 1
        if button == 5:
            self.mouse_wheel_d = 1

    def check_events(self, rstr=False):
        """
        Checks pygame event queue and sets the internal display values
        based on any new generated events.
        :param rstr: pygame returns an enumerated int by default, when
                      this is set to true we return a list of strings.
        :return: a list of key down events. Parse them with
                  pygame.K_<lowercase_letter>
        """
        self.mouse_wheel_u = self.mouse_wheel_d = 0
        self.last_lb = self.mouse_l
        self.last_rb = self.mouse_r
        self.lb_down = self.lb_up = None
        self.rb_down = self.rb_up = None
        key = []

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                self.done = True
            if e.type == pygame.MOUSEMOTION:
                self.mouse_raw_x = e.pos[0]
                self.mouse_raw_y = e.pos[1]
                x = int((e.pos[0] - self.offset_x) * self.scale_x)
                y = int((e.pos[1] - self.offset_y) * self.scale_y)
                self.mouse_x, self.mouse_y = self._clamp(x, y)
            if e.type == pygame.MOUSEBUTTONUP:
                self._set_button_state(0, e.button)
            if e.type == pygame.MOUSEBUTTONDOWN:
                self._set_button_state(1, e.button)
            if e.type == pygame.KEYDOWN:
                if rstr:
                    key.append(pygame.key.name(e.key))
                else:
                    key.append(e.key)

        self.pressed = pygame.key.get_pressed()

        if self.last_lb == 0 and self.mouse_l == 1:
            self.lb_down = (self.mouse_x, self.mouse_y)
        if self.last_lb == 1 and self.mouse_l == 0:
            self.lb_up = (self.mouse_x, self.mouse_y)

        if self.last_rb == 0 and self.mouse_r == 1:
            self.rb_down = (self.mouse_x, self.mouse_y)
        if self.last_rb == 1 and self.mouse_r == 0:
            self.rb_up = (self.mouse_x, self.mouse_y)

        if self.pressed[pygame.K_ESCAPE] == 1:
            self.done = True

    def is_done(self):
        """
        Checks the event queue and returns True if a quit event has been issued.
        :return: True on a quit event, False otherwise.
        """
        self.check_events()
        return self.done

    def _clamp(self, x, y):
        """
        Clamp all values between zero and the image width and height.
        :param x: x value to crop
        :param y: y value to crop
        :return: cropped tuple
        """
        rx = x
        ry = y

        if x > self.img_w:
            rx = self.img_w

        if x < 0:
            rx = 0

        if y > self.img_h:
            ry = self.img_h

        if y < 0:
            ry = 0

        return rx, ry

    def quit(self):
        """
        Quit the pygame instance.
        :return: None
        """
        pygame.display.quit()
        pygame.quit()

