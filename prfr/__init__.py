__version__ = "0.2.3"
__uri__ = "http://github.com/al-jshen/prfr"
__author__ = "Jeff Shen"
__email__ = "jshen2014@hotmail.com"
__license__ = "MIT"

from prfr.model import ProbabilisticRandomForestRegressor
from prfr.utils import split_arrays
from prfr.model import _has_jax as has_jax

__all__ = ["ProbabilisticRandomForestRegressor", "split_arrays", "has_jax"]
