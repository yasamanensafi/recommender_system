from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.inputs import SparseFeat,get_feature_names
from sklearn.model_selection import train_test_split

sparse_features = ['site_name','posa_continent','user_location_country','user_location_region','user_location_city',
             'user_id','is_mobile','is_package','channel','click_month','checkin_month','checkout_month',
            'srch_adults_cnt','srch_children_cnt','srch_rm_cnt','srch_destination_id','hotel_continent',
               'hotel_country','cnt','north_am_ci', 'north_am_co', 'europe_ci', 'europe_co',
             'click_month', 'checkin_month',
                   'checkout_month']

dense_features = ['hotel_market']
target = ['rating'] 
    

    
# categ_sparse / conti_dense



def simple_pre(df):

    # Label Encoding for sparse features,and normalization for dense numerical features
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])

    mms = MinMaxScaler(feature_range=(0,1))
    df[dense_features] = mms.fit_transform(df[dense_features])


    #Generate feature columns
    #For sparse features, we transform them into dense vectors by embedding techniques.
    #For dense numerical features, we concatenate them to the input tensors of fully connected layer.

    # count #unique features for each sparse field
    fixlen_feature_columns = [SparseFeat(feat, df[feat].nunique(),embedding_dim=4)
                              for feat in sparse_features]

    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    return linear_feature_columns,dnn_feature_columns,feature_names

def train_test(df):
    # generate input data for model
    train, test = train_test_split(df, test_size=0.2)
    train_model_input = {name:train[name].values for name in feature_names}
    test_model_input = {name:test[name].values for name in feature_names}
    return train,test,train_model_input,test_model_input