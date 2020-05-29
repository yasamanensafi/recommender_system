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

target = config.target
#1
def find_dnn_hidden_units(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test):
    cols = ['dnn_hidden_units','RMSE','MAE','MSE','AUC','score']
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
        df_result.iloc[i]['score']=(1/df_result.iloc[i]['RMSE'])*(1/df_result.iloc[i]['MAE'])*(2*df_result.iloc[i]['AUC'])
    a = df_result[df_result['score']==df_result['score'].max()]['dnn_hidden_units']
    best_dnn_hidden_units = int(str(a.values).split(",")[1][:-2])
    print("--------------------------")
    print(best_dnn_hidden_units, "is the best value for dnn_hidden_units")
    return best_dnn_hidden_units

#4
def find_l2_reg_dnn(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test,best_dnn_hidden_units,best_l2_reg_linear):
    cols4 = ['l2_reg_dnn','RMSE','MAE','MSE','AUC','score']
    df_result4 = pd.DataFrame(columns=cols4, index=range(len(config.param_rand['l2_reg_dnn'])))
    for i,x in enumerate(config.param_rand['l2_reg_dnn']): 
        model = WDL(linear_feature_columns, dnn_feature_columns,
                dnn_hidden_units=(best_dnn_hidden_units,best_dnn_hidden_units), 
                l2_reg_linear=best_l2_reg_linear,
                l2_reg_embedding=0.001, l2_reg_dnn=i, init_std=0.0001, seed=1024,task='binary')

        model.compile("adam", "mse", metrics=['mse'])

        history = model.fit(train_model_input, train[target].values,
                                batch_size=256, epochs=config.model_epoch['epoch'], verbose=2, validation_split=0.2)
        pred_ans = model.predict(test_model_input, batch_size=256)
        
        auc = roc_auc_score(test[target].values, pred_ans)
        
        df_result4.loc[i].l2_reg_dnn = x
        df_result4.loc[i].RMSE = np.round(math.sqrt(mean_squared_error(test[target].values, pred_ans)),3)
        df_result4.loc[i].MAE = np.round(mean_absolute_error(test[target].values, pred_ans),3)
        df_result4.loc[i].MSE = np.round(mean_squared_error(test[target].values, pred_ans),3)
        df_result4.loc[i].AUC = np.round(auc,3)    
        df_result4.iloc[i]['score']=(1/df_result4.iloc[i]['RMSE'])*(1/df_result4.iloc[i]['MAE'])*(2*df_result4.iloc[i]['AUC'])
    d = df_result4[df_result4['score']==df_result4['score'].max()]['l2_reg_dnn']
    best_l2_reg_dnn= d
    print("--------------------------")
    print(best_l2_reg_dnn, "is the best value for l2_reg_dnn")
    return best_l2_reg_dnn


        