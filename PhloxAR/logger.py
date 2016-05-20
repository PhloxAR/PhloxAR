# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import unicode_literals, absolute_import

import logging
import os
import sys
from PhloxAR.compat import PY2
from random import randint
from functools import partial


__all__ = [
    'Logger', 'LOG_LEVELS', 'COLORS', 'LoggerHistory'
]

Logger = None

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = list(range(8))

RESET_SEQ = '\033[0m'
COLOR_SEQ = '\033[1;%dm'
BOLD_SEQ = '\033[1m'

pre_stderr = sys.stderr


def formatter_message(msg, use_color=True):
    if use_color:
        msg = msg.replace('$RESET', RESET_SEQ)
        msg = msg.replace('$BOLD', BOLD_SEQ)
    else:
        msg = msg.replace('$RESET', '').replace('$BOLD', '')

    return msg

COLORS = {
    'TRACE': MAGENTA,
    'WARNING': YELLOW,
    'INFO': GREEN,
    'DEBUG': CYAN,
    'CRITICAL': RED,
    'ERROR': RED
}

logging.TRACE = 9

LOG_LEVELS = {
    'trace': logging.TRACE,
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


class FileHandler(logging.Handler):

    history = []
    filename = 'log.txt'
    fd = None

    def purge_logs(self, directory):
        """
        Purge log is called randomly to prevent the log directory from
        being filled by lots and lots of log files.

        Args:
            directory (file path): the directory to be purged

        Returns:
            None
        """
        if randint(0, 20) != 0:
            return

        # use config ?
        maxfiles = 100

        print('Purge log fired. Analysing...')
        join = os.path.join
        unlink = os.unlink

        # search for all files
        l = [join(directory, x) for x in os.listdir(directory)]

        if len(l) > maxfiles:
            # get creation time on every files
            l = [{'fn': x, 'ct': os.path.getctime(x)} for x in l]

            # sort by date
            l = sorted(l, key=lambda x: x['ct'])

            # get the oldest (keep last maxfiles)
            l = l[:-maxfiles]
            print('Purge {} log files'.format(len(l)))

            # now, unlink every files in the list
            for filename in l:
                unlink(filename['fn'])

        print('Purge finished!')

    def _configure(self, *args, **kwargs):
        # TODO
        from time import strftime
        pass

    def _write_message(self, record):
        if FileHandler.fd in (None, False):
            return

        msg = self.format(record)
        stream = FileHandler.fd
        fs = '%s\n'

        stream.write('[%-18s]' % record.levelname)
        stream.write(fs % msg.encode('UTF-8'))
        stream.flush()

    def emit(self, record):
        # during the startup, store the message in the history
        if Logger.logfile_activated is None:
            FileHandler.history += [record]
            return

        # startup done, if the log file is not activated, avoid history
        if Logger.logfile_activated is False:
            FileHandler.history = []
            return

        if FileHandler.fd is None:
            try:
                self._configure()
            except Exception:
                FileHandler.fd = False
                Logger.exception('Error while activating FileHanlder logger')
                return

            while FileHandler.history:
                _msg = FileHandler.history.pop()
                self._write_message(_msg)

        self._write_message(record)


class LoggerHistory(logging.Handler):

    history = []

    def emit(self, record):
        LoggerHistory.history = [record] + LoggerHistory.history[:100]


class ColoredFormatter(logging.Formatter):

    def __init__(self, msg, use_color=True):
        super(ColoredFormatter, self).__init__(msg)
        self.use_color = use_color

    def format(self, record):
        try:
            msg = record.msg.split(':', 1)
            if len(msg) == 2:
                record.msg = '[%-12s]%s' % (msg[0], msg[1])
        except:
            pass
        levelname = record.levelname
        if record.levelno == logging.TRACE:
            levelname = 'TRACE'
            record.levelname = levelname
        if self.use_color and levelname in COLORS:
            levelname_color = (
                COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ)
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


class ConsoleHandler(logging.StreamHandler):

    def filter(self, record):
        try:
            msg = record.msg
            k = msg.split(':', 1)
            if k[0] == 'stderr' and len(k) == 2:
                pre_stderr.write(k[1] + '\n')
                return False
        except:
            pass
        return True


class LogFile(object):

    def __init__(self, channel, func):
        self.buffer = ''
        self.func = func
        self.channel = channel
        self.errors = ''

    def write(self, s):
        s = self.buffer + s
        self.flush()
        f = self.func
        channel = self.channel
        lines = s.split('\n')
        for l in lines[:-1]:
            f('%s: %s' % (channel, l))
        self.buffer = lines[-1]

    def flush(self):
        return


def logger_config_update(section, key, value):
    if LOG_LEVELS.get(value) is None:
        raise AttributeError("Loglevel {0!r} doesn't exists".format(value))
    Logger.setLevel(level=LOG_LEVELS.get(value))


# PhloxAR default logger instance
Logger = logging.getLogger('phloxar')
Logger.logfile_activated = None
Logger.trace = partial(Logger.log, logging.TRACE)

# set phloxar logger as the default
logging.root = Logger

# add default phloxar logger
Logger.addHandler(LoggerHistory())

use_color = (
    os.name != 'nt' and
    os.environ.get('TERM') in ('xterm', 'rxvt', 'rxvt-unicode', 'xterm-256color')
)

color_fmt = formatter_message('[%(levelname)-18s] %(message)s', use_color)
formatter = ColoredFormatter(color_fmt, use_color=use_color)
console = ConsoleHandler()
console.setFormatter(formatter)
Logger.addHandler(console)

# install stderr handlers
sys.stderr = LogFile('stderr', Logger.warning)

LoggerHistory = LoggerHistory