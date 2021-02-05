
from __future__ import absolute_import
import warnings

try:
    from .molfile import libpymolfile
except ImportError:
    warnings.warn("libpymolfile package not available, pymolfile does not work without its library!")

from . import plugin_list
from . import pymolfile

__all__ = [ "pymolfile" ]

from .pymolfile import OpenMolfile, list_plugins

