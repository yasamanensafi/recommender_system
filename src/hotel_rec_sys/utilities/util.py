import numpy as np
import pandas as pd
import math
from config import config
from utilities import util

def custom_scale(array):
    result=[]
    for v in (array):
        min_new=0.5
        max_new=2
        max_old = array.max()
        min_old = array.min()
        result.append((max_new - min_new)/(max_old-min_old) * (v - max_old) + max_new)
    return np.array(result)