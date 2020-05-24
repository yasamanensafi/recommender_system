from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.inputs import SparseFeat,get_feature_names
from config import config

target = ['rating'] 

def simple_pre(df):

    # Label Encoding for sparse features,and normalization for dense numerical features
    for feat in config.sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])

    mms = MinMaxScaler(feature_range=(0,1))
    df[config.dense_features] = mms.fit_transform(df[config.dense_features])


    #Generate feature columns
    #For sparse features, we transform them into dense vectors by embedding techniques.
    #For dense numerical features, we concatenate them to the input tensors of fully connected layer.

    # count #unique features for each sparse field
    fixlen_feature_columns = [SparseFeat(feat, df[feat].nunique(),embedding_dim=4)
                              for feat in config.sparse_features]

    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    return linear_feature_columns,dnn_feature_columns,feature_names

