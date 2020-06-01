from deepctr.inputs import build_input_features, get_linear_logit, input_from_feature_columns, combined_dnn_input
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.utils import add_func
from deepctr.models import WDL
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
import pandas as pd
import math
from config import config
import util

target = config.target

# WIDE AND DEEP
#1 dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
def find_dnn_hidden_units(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test):
    cols = ['dnn_hidden_units','RMSE','MAE','MSE','AUC']
    df_result = pd.DataFrame(columns=cols, index=range(len(config.param_rand['dnn_hidden_units'])))
    for i,x in enumerate(config.param_rand['dnn_hidden_units']): 
        model = WDL(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=x, 
                init_std=0.0001, seed=1024,task='binary')

        model.compile("adam", "mse", metrics=['mse'])

        history = model.fit(train_model_input, train[target].values,
                                batch_size=256, epochs=config.model_epoch['epoch'], verbose=2, validation_split=0.2, )
        pred_ans = model.predict(test_model_input, batch_size=256)
        
        auc = roc_auc_score(test[target].values, pred_ans)
        df_result.loc[i].dnn_hidden_units = x

        df_result.loc[i].RMSE = np.round(math.sqrt(mean_squared_error(test[target].values, pred_ans)),3)
        df_result.loc[i].MAE = np.round(mean_absolute_error(test[target].values, pred_ans),3)
        df_result.loc[i].MSE = np.round(mean_squared_error(test[target].values, pred_ans),3)
        df_result.loc[i].AUC = np.round(auc,3)    
    return df_result

def find_best_dnn_hidden_units(df_result):
    a = df_result.sort_values(['AUC','MAE','RMSE'],ascending=False).iloc[0]["dnn_hidden_units"]
    b= str(a).split(",")[0].split("(")[1]
    return b

#2  l2_reg_linear: float. L2 regularizer strength applied to wide part
def find_l2_reg_linear(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test):
    cols = ['l2_reg_linear','RMSE','MAE','MSE','AUC']
    df_result = pd.DataFrame(columns=cols, index=range(len(config.param_rand['l2_reg_linear'])))
    for i,x in enumerate(config.param_rand['l2_reg_linear']): 

        ##Add dnn_hidden_units as b later 
        model = WDL(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128,128),
                l2_reg_linear=x,l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024,task='binary')

        model.compile("adam", "mse", metrics=['mse'])

        history = model.fit(train_model_input, train[target].values,
                                batch_size=256, epochs=config.model_epoch['epoch'], verbose=2, validation_split=0.2, )
        pred_ans = model.predict(test_model_input, batch_size=256)
        
        auc = roc_auc_score(test[target].values, pred_ans)
        df_result.loc[i].l2_reg_linear = x
        df_result.loc[i].RMSE = np.round(math.sqrt(mean_squared_error(test[target].values, pred_ans)),3)
        df_result.loc[i].MAE = np.round(mean_absolute_error(test[target].values, pred_ans),3)
        df_result.loc[i].MSE = np.round(mean_squared_error(test[target].values, pred_ans),3)
        df_result.loc[i].AUC = np.round(auc,3)    
    return df_result

#3  l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
def find_l2_reg_embedding(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test):
    cols = ['l2_reg_embedding','RMSE','MAE','MSE','AUC']
    df_result = pd.DataFrame(columns=cols, index=range(len(config.param_rand['l2_reg_embedding'])))
    for i,x in enumerate(config.param_rand['l2_reg_embedding']): 

        ##Add dnn_hidden_units as b later 
        model = WDL(linear_feature_columns, dnn_feature_columns,
                # ADD LATER
                dnn_hidden_units=(2,2), l2_reg_linear=0.1,
                l2_reg_embedding=x, l2_reg_dnn=0, init_std=0.0001, seed=1024, task='binary')

        model.compile("adam", "mse", metrics=['mse'])
        history = model.fit(train_model_input, train[target].values,
                                batch_size=256, epochs=config.model_epoch['epoch'], verbose=2, validation_split=0.2, )
        pred_ans = model.predict(test_model_input, batch_size=256)
        
        auc = roc_auc_score(test[target].values, pred_ans)
        df_result.loc[i].l2_reg_embedding = x
        df_result.loc[i].RMSE = np.round(math.sqrt(mean_squared_error(test[target].values, pred_ans)),3)
        df_result.loc[i].MAE = np.round(mean_absolute_error(test[target].values, pred_ans),3)
        df_result.loc[i].MSE = np.round(mean_squared_error(test[target].values, pred_ans),3)
        df_result.loc[i].AUC = np.round(auc,3)    
    return df_result

#4 l2_reg_dnn: float. L2 regularizer strength applied to DNN
def find_l2_reg_dnn(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test):
    cols = ['l2_reg_dnn','RMSE','MAE','MSE','AUC']
    df_result = pd.DataFrame(columns=cols, index=range(len(config.param_rand['l2_reg_dnn'])))
    for i,x in enumerate(config.param_rand['l2_reg_dnn']): 

        ##Add dnn_hidden_units as b later 
        model = WDL(linear_feature_columns, dnn_feature_columns,
                # Add LATER
                dnn_hidden_units=(2,2), l2_reg_linear=1,l2_reg_embedding=0.001,
                l2_reg_dnn=x, init_std=0.0001, seed=1024,task='binary')

        model.compile("adam", "mse", metrics=['mse'])
        history = model.fit(train_model_input, train[target].values,
                                batch_size=256, epochs=config.model_epoch['epoch'], verbose=2, validation_split=0.2, )
        pred_ans = model.predict(test_model_input, batch_size=256)
        
        auc = roc_auc_score(test[target].values, pred_ans)
        df_result.loc[i].l2_reg_dnn = x
        df_result.loc[i].RMSE = np.round(math.sqrt(mean_squared_error(test[target].values, pred_ans)),3)
        df_result.loc[i].MAE = np.round(mean_absolute_error(test[target].values, pred_ans),3)
        df_result.loc[i].MSE = np.round(mean_squared_error(test[target].values, pred_ans),3)
        df_result.loc[i].AUC = np.round(auc,3)    
    return df_result
        
#5   dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
def find_dnn_dropout(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test):
    cols = ['dnn_dropout','RMSE','MAE','MSE','AUC']
    df_result = pd.DataFrame(columns=cols, index=range(len(config.param_rand['dnn_dropout'])))
    for i,x in enumerate(config.param_rand['dnn_dropout']): 

        ##Add dnn_hidden_units as b later 
        model = WDL(linear_feature_columns,dnn_feature_columns,
                    #Replace Later
                    dnn_hidden_units=(2,2),
                    init_std=0.0001, seed=1024, dnn_dropout=x, dnn_activation='relu',task='binary')

        model.compile("adam", "mse", metrics=['mse'])
        history = model.fit(train_model_input, train[target].values,
                                batch_size=256, epochs=config.model_epoch['epoch'], verbose=2, validation_split=0.2, )
        pred_ans = model.predict(test_model_input, batch_size=256)
        
        auc = roc_auc_score(test[target].values, pred_ans)
        df_result.loc[i].dnn_dropout = x
        df_result.loc[i].RMSE = np.round(math.sqrt(mean_squared_error(test[target].values, pred_ans)),3)
        df_result.loc[i].MAE = np.round(mean_absolute_error(test[target].values, pred_ans),3)
        df_result.loc[i].MSE = np.round(mean_squared_error(test[target].values, pred_ans),3)
        df_result.loc[i].AUC = np.round(auc,3)    
    return df_result