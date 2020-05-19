import sys
sys.path.append('../hotel_rec_sys/feature/')

import feat_eng
import holiday
import preprocessing


#load data and choose number of rows
df = feat_eng.load(1000)

df= holiday.add_holiday(df)

linear_feature_columns,dnn_feature_columns,feature_names = preprocessing.simple_pre(df)

#train,test,train_model_input,test_model_input = preprocessing.train_test(df)

print(feature_names)