## intensity distribution
# 1. skewness
# 2. kurtosis

from scipy.stats import kurtosis
from scipy.stats import skew
import numpy as np

from edges import get_distance_centroid, find_edge
from cell_size import find_alphashape


def find_skewness(shape_signatures):
    """
    find skewness of shape
    """
    skew_arr = [skew(x) for x in shape_signatures]
    return skew_arr

def find_kurtosis(shape_signatures):
    """
    Find kurtosis of shape
    """
    kurtosis_arr = [kurtosis(x) for x in shape_signatures]
    return kurtosis_arr
