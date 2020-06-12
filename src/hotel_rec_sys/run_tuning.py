import sys
sys.path.append('../hotel_rec_sys/feature/')
sys.path.append('../hotel_rec_sys/model/')
sys.path.append('../hotel_rec_sys/config/')
sys.path.append('../hotel_rec_sys/utilities/')

import pickle
import load
import feature_engineering
import sparse_featuring
import widendeep
import deepfm
import xdeepfm
import traintest
import warnings
import tuning
import config
import score
import util
warnings.filterwarnings("ignore")
from deepctr.layers import custom_objects
import tuning

#load data and choose number of rows
df = load.load_data(150000)

#Remove rows with the same user_id and item_id and different rating
df = load.remove_duplicate(df)

# Apply feature engineering
df = feature_engineering.extract_week(df,'date_time','click_week')
df = feature_engineering.extract_month_year(df)
df = feature_engineering.add_holiday(df)
df = feature_engineering.log_transform(df,'orig_destination_distance')
df = feature_engineering.create_cluster(df,'user_location_region',3)
df = feature_engineering.create_cluster(df,'user_location_city',3)
df = feature_engineering.extract_family_status(df)


linear_feature_columns,dnn_feature_columns,feature_names = sparse_featuring.simple_pre(df)

train,test,train_model_input,test_model_input = traintest.train_test(linear_feature_columns,dnn_feature_columns,feature_names,df)

#1) dnn_hidden_units tuning
df_result1= tuning.find_dnn_hidden_units(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test)
dhu= tuning.find_best_dnn_hidden_units(df_result1)
#print(df_result1)

# 2)l2_reg_linear tuning
#df_result2= tuning.find_l2_reg_linear(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test)
#l2rl = df_result2.sort_values(['AUC','MAE','RMSE'],ascending=False).iloc[0]["l2_reg_linear"]
#print(df_result2)

#3) l2_reg_embedding
#df_result3= tuning.find_l2_reg_embedding(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test)
#l2re = df_result3.sort_values(['AUC','MAE','RMSE'],ascending=False).iloc[0]["l2_reg_embedding"]
#print(df_result3)

#4) l2_reg_dnn
#df_result4= tuning.find_l2_reg_dnn(linear_feature_columns, dnn_feature_columns, train_model_input, train, test_model_input, test)
#l2rd = df_result4.sort_values(['AUC','MAE','RMSE'],ascending=False).iloc[0]["l2_reg_dnn"]
#print(df_result4)

#5) But overall, the result of model without defining the dnn_dropout is better
#df_result5= tuning.find_dnn_dropout(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test)
dd = df_result5.sort_values(['AUC','MAE','RMSE'],ascending=False).iloc[0]["dnn_dropout"]
#print(df_result5)

print("The best dnn_hidden_units is(",dhu,",",dhu,")")
#print("The best l2_reg_linear is",l2rl)
#print("The best l2_reg_embedding is",l2re)
#print("The best l2_reg_dnn is",l2rd)
##print("The best dnn_dropout is",dd)

# ------- OUTPUT with 10 epochs----------
#The best dnn_hidden_units is( 2 , 2 )
#The best l2_reg_linear is 0.1
#The best l2_reg_embedding is 0.001
#The best l2_reg_dnn is 4
#The best dnn_dropout is 0.6