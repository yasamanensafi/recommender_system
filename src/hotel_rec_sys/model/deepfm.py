from deepctr.inputs import build_input_features, get_linear_logit, input_from_feature_columns, combined_dnn_input
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.utils import add_func
from deepctr.models import DeepFM
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
import math
from config import config

target = ['rating']

def deepfm_model(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test):
    model = DeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units= config.deepfm_att["dnn_hidden_units"]
            , init_std=config.deepfm_att["init_std"], seed=config.deepfm_att["seed"],
             dnn_dropout=config.deepfm_att["dnn_dropout"], dnn_activation=config.deepfm_att["dnn_activation"],
             task=config.deepfm_att["task"],fm_group=config.deepfm_att["fm_group"],
             dnn_use_bn=config.deepfm_att["dnn_use_bn"])

    model.compile("adam", "mse", metrics=['mse'])

    history = model.fit(train_model_input, train[target].values,
                            batch_size=256, epochs=10, verbose=2, validation_split=0.2)
    
    pred_ans = model.predict(test_model_input, batch_size=256)
    auc = roc_auc_score(test[target].values, pred_ans)
    
    deepfm_result = {"RMSE":np.round(math.sqrt(mean_squared_error(test[target].values, pred_ans)),3),
                   "MAE":np.round(mean_absolute_error(test[target].values, pred_ans),3),
                   "MSE":np.round(mean_squared_error(test[target].values, pred_ans),3),
                   "AUC":np.round(auc,3)}
    return deepfm_result