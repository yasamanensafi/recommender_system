import sys
sys.path.append('../hotel_rec_sys/feature/')
sys.path.append('../hotel_rec_sys/model/')

import feat_eng
import holiday
import preprocessing
import widendeep
import deepfm
import xdeepfm




#load data and choose number of rows
df = feat_eng.load(1000)

df= holiday.add_holiday(df)

linear_feature_columns,dnn_feature_columns,feature_names = preprocessing.simple_pre(df)

train,test,train_model_input,test_model_input = preprocessing.train_test(linear_feature_columns,dnn_feature_columns,feature_names,df)

# wide and deep
widendeep_result = widendeep.widendeep_model(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test)

#DeepFM
deepfm_result = deepfm.deepfm_model(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test)

#XDeepFM
xdeepfm_result = xdeepfm.xdeepfm_model(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test)

print("Wide and Deep", widendeep_result,"DeepFM", deepfm_result,"XDeepFM", xdeepfm_result, sep='\n')
