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
import config
import score
warnings.filterwarnings("ignore")
from deepctr.layers import custom_objects
from utilities import util
from tensorflow.python.keras.models import load_model
from deepctr.layers import custom_objects

#load data and choose number of rows
df = load.load_data(150000)

#Remove rows with the same user_id and item_id and different rating
df = load.remove_duplicate(df)

# Apply feature engineering
df = feature_engineering.extract_week(df,'date_time','click_week')
df = feature_engineering.extract_month_year(df)
df = feature_engineering.add_holiday(df)
df = feature_engineering.log_transform(df,'orig_destination_distance')
#df = feature_engineering.z_score_normalizing(df,'cnt')
df = feature_engineering.create_cluster(df,'user_location_region',3)
df = feature_engineering.create_cluster(df,'user_location_city',3)
df = feature_engineering.extract_family_status(df)


linear_feature_columns,dnn_feature_columns,feature_names = sparse_featuring.simple_pre(df)

train,test,train_model_input,test_model_input = traintest.train_test(linear_feature_columns,dnn_feature_columns,feature_names,df)

# wide and deep
widendeep_result= widendeep.widendeep_model(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test)

#DeepFM
deepfm_result = deepfm.deepfm_model(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test)

#XDeepFM
xdeepfm_result= xdeepfm.xdeepfm_model(linear_feature_columns,dnn_feature_columns,train_model_input,train,test_model_input,test)
xdeepfm_result = util.apply_custom_scale(xdeepfm_result)
warnings.filterwarnings("ignore")
best_model = util.choose_model(widendeep_result,deepfm_result,xdeepfm_result).values
print("The models are saved and ready to use")   
if best_model=="Wide and Deep":
    model = load_model('saved_widendeep.h5',custom_objects)# load_model,just add a parameter 
elif best_model=="DeepFM":
    model = load_model('saved_deepfm.h5',custom_objects)
elif best_model=="XDeepFM":
    model = load_model('saved_xdeepfm.h5',custom_objects)

new_df= score.score(model,test_model_input,test)
hotel_df = score.create_hotel_df()

