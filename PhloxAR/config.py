try:
    from ConfigParser import ConfigParser as PythonConfigParser
except ImportError:
    from configparser import RawConfigParser as PythonConfigParser
from os import environ
from os.path import exists
from .logger import Logger, logger_config_update
from collections import OrderedDict
from .utils import platform
from .compat import PY2, string_types
from weakref import ref