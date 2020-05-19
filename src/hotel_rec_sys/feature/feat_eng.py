import pandas as pd

def load(num_rows):
    # load data
    dataframe = pd.read_csv('../../data/hotel_data/train.csv', sep=',', nrows=num_rows)
    # rename 2 columns
    df = dataframe.rename(columns={'hotel_cluster': 'item_id', 'is_booking': 'rating'})
    
    df= df.drop(['orig_destination_distance'],axis=1)
    df = df.dropna()
    
    # extract month from date_time
    df['click_month'] = pd.DatetimeIndex(df['date_time']).month
    df['checkin_month'] = pd.DatetimeIndex(df['srch_ci']).month
    df['checkout_month'] = pd.DatetimeIndex(df['srch_co']).month
    
    return df