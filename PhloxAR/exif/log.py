#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# Copyright 2016(c) Matthias Y. Chen
# <matthiasychen@gmail.com/matthias_cy@outlook.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

import sys
import logging


TEXT_NORMAL = 0
TEXT_BOLD = 1
TEXT_RED = 31
TEXT_GREEN = 32
TEXT_YELLOW = 33
TEXT_BLUE = 34
TEXT_MAGENTA = 35
TEXT_CYAN = 36


class Formatter(logging.Formatter):
    def __init__(self, debug=False, color=False):
        self.color = color
        self.debug = debug

        if self.debug:
            log_format = '%(levelname)-6s %(message)s'
        else:
            log_format = '%(message)s'

        super(logging.Formatter, self).__init__(self, log_format)

    def format(self, record):
        if self.debug and self.color:
            if record.levelno >= logging.CRITICAL:
                color = TEXT_RED
            elif record.levelno >= logging.ERROR:
                color = TEXT_RED
            elif record.levelno >= logging.WARNING:
                color = TEXT_YELLOW
            elif record.levelno >= logging.INFO:
                color = TEXT_GREEN
            elif record.levelno >= logging.DEBUG:
                color = TEXT_CYAN
            else:
                color = TEXT_NORMAL
            record.levelname = '\x1b[%sm%s\x1b[%sm' % (color,
                                                       record.levelname,
                                                       TEXT_NORMAL)
        return logging.Formatter.format(self, record)


class Handler(logging.StreamHandler):
    def __init__(self, log_level, debug=False, color=False):
        super(logging.StreamHandler, self).__init__(self, sys.stdout)
        self.color = color
        self.debug = debug
        self.setFormatter(Formatter(debug, color))
        self.setLevel(log_level)


def get_logger():
    return logging.getLogger('exif')


def setup_logger(debug, color):
    """
    Configure the logger.
    """
    if debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logger = logging.getLogger('exif')
    stream = Handler(log_level, debug, color)
    logger.addHandler(stream)
    logger.setLevel(log_level)
