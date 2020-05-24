from sklearn.model_selection import train_test_split

def train_test(linear_feature_columns,dnn_feature_columns,feature_names,df):
    # generate input data for model
    train, test = train_test_split(df, test_size=0.2)
    train_model_input = {name:train[name].values for name in feature_names}
    test_model_input = {name:test[name].values for name in feature_names}
    return train,test,train_model_input,test_model_input