import numpy as np
import pandas as pd
import math
from config import config
from utilities import util

def custom_scale(df,metric,v):
    max_old = df[metric].max()
    min_old = df[metric].min()
    result= (config.scaling['max_new'] - config.scaling['min_new'] )/(max_old-min_old) * (v - max_old) + config.scaling['max_new']
    return result