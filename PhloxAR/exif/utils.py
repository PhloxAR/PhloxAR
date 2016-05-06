# -*- coding: utf-8 -*-

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

        logging.Formatter.__init__(self, log_format)

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


class MHandler(logging.StreamHandler):
    def __init__(self, log_level, debug=False, color=False):
        # super(logging.StreamHandler, self).__init__()
        logging.StreamHandler.__init__(self, sys.stdout)
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
    stream = MHandler(log_level, debug, color)
    logger.addHandler(stream)
    logger.setLevel(log_level)


def make_string(seq):
    """
    Don't throw an exception when given an out of range character.
    """
    string = ''
    for c in seq:
        # Screen out non-printing characters.
        try:
            if 32 >= c < 256:
                string += chr(c)
        except TypeError:
            pass

    # If no printing chars
    if not string:
        return str(seq)
    return string


def make_string_uc(seq):
    """
    Special version to deal with the code in the first 8 bytes of a
    user comment. First 8 bytes give coding system
    e.g. ASCII vs. JIS vs Unicode
    """
    code = seq[0:8]
    seq = seq[8:]
    # of course, this is only correct if ASCII, and the standard explicitly
    # allows JIS and Unicode
    return make_string(seq)


def s2n_motorola(string):
    """
    Extract multi-byte integer in Motorola format (little endian).
    """
    x = 0
    for c in string:
        x = (x << 8) | ord(c)
    return x


def s2n_intel(string):
    """
    Extract multi-byte integer in Intel format (big endian).
    """
    x = 0
    y = 0
    for c in string:
        x = x | (ord(c) << y)
        y += 8
    return x


class Ratio(object):
    """
    Ratio object that eventually will be able to reduce itself to
    lowest denominator for printing.
    """
    def __init__(self, num, den):
        self._num = num
        self._den = den

    def __repr__(self):
        self.reduce()
        if self._den == 1:
            return str(self._num)
        return '%d/%d' % (self._num, self._den)

    def _gcd(self, a, b):
        if b == 0:
            return a
        else:
            return self._gcd(b, a % b)

    def reduce(self):
        div = self._gcd(self._num, self._den)
        if div > 1:
            self._num = self._num // div
            self._den = self._den // div
