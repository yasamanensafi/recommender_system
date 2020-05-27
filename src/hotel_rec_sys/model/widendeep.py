from deepctr.inputs import build_input_features, get_linear_logit, input_from_feature_columns, combined_dnn_input
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.utils import add_func
from deepctr.models import WDL
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
import math
from config import config

target = config.target

def widendeep_model(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test):
    model = WDL(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=config.widendeep_att["dnn_hidden_units"], 
    l2_reg_linear=config.widendeep_att["l2_reg_linear"],l2_reg_embedding=config.widendeep_att["l2_reg_embedding"], 
    l2_reg_dnn=config.widendeep_att["l2_reg_dnn"], init_std=config.widendeep_att["init_std"], 
    seed=config.widendeep_att["seed"], task=config.widendeep_att["task"])

    model.compile("adam", "mse", metrics=['mse'])

    history = model.fit(train_model_input, train[target].values,
                            batch_size=256, epochs=10, verbose=2, validation_split=0.2)
    
    pred_ans = model.predict(test_model_input, batch_size=256)
    auc = roc_auc_score(test[target].values, pred_ans)
    
    widendeep_result = {"RMSE":np.round(math.sqrt(mean_squared_error(test[target].values, pred_ans)),3),
                   "MAE":np.round(mean_absolute_error(test[target].values, pred_ans),3),
                   "MSE":np.round(mean_squared_error(test[target].values, pred_ans),3),
                   "AUC":np.round(auc,3)}
    return widendeep_result