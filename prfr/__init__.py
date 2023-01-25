__version__ = "0.2.4"
__uri__ = "http://github.com/al-jshen/prfr"
__author__ = "Jeff Shen"
__email__ = "jshen2014@hotmail.com"
__license__ = "MIT"

from prfr.model import ProbabilisticRandomForestRegressor, calibrate
from prfr.utils import split_arrays, check_calibration
from prfr.model import _has_jax as has_jax

__all__ = [
    "ProbabilisticRandomForestRegressor",
    "calibrate",
    "split_arrays",
    "has_jax",
    "check_calibration",
]
