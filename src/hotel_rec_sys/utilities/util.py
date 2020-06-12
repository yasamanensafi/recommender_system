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

def apply_custom_scale(df_result):

    return df_result

def choose_model(df1,df2,df3):
    result = df1.append([df2, df3])
    result=result.reset_index().drop('index',axis=1)
    a=util.custom_scale(1/result['RMSE'])
    b=util.custom_scale(1/result['MAE'])
    c=util.custom_scale(result['AUC'])   
    result['score']=np.round(a+b+(2*c),2)
    print(result)
    print("The best model is ",result[result['score']== result['score'].max()]["model"])
    return result[result['score']== result['score'].max()]["model"]
