"""Top-level package for pyFDN."""

__author__ = """Kalyan Pandey"""
__email__ = "pandey.kalyan416@gmail.com"
__version__ = "0.1.0"

from pyFDN.auxiliary.acoustics import one_pole_absorption, rt60_to_slope, slope_to_rt60
from pyFDN.auxiliary.delay import ms2smp
from pyFDN.auxiliary.filters import TFMatrix, ZFIR, ZFilter, ZScalar, ZSOS, ZTF
from pyFDN.auxiliary.math import matrix_convolution, matrix_polyval, polydiag
from pyFDN.auxiliary.utils import db2mag, hertz2unit, is_bounding_curve, mag2db, pole_boundaries
from pyFDN.generate.random_orthogonal import random_orthogonal
from pyFDN.process import process_fdn

processFDN = process_fdn
