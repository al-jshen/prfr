__version__ = "0.1.1"
__uri__ = "http://github.com/al-jshen/prfr"
__author__ = "Jeff Shen"
__email__ = "jshen2014@hotmail.com"
__license__ = "MIT"

from prfr.model import ProbabilisticRandomForestRegressor
from prfr.utils import split_arrays

__all__ = ["ProbabilisticRandomForestRegressor", "split_arrays"]
