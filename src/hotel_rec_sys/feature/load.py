import pandas as pd

def load_data(num_rows):
    # load data
    dataframe = pd.read_csv('../../data/hotel_data/train.csv', sep=',', nrows=num_rows)
    # rename 2 columns
    df = dataframe.rename(columns={'hotel_cluster': 'item_id', 'is_booking': 'rating'})
    destinations = pd.read_csv('../../data/hotel_data/destinations.csv', sep=',')
    # merge 2 dataframes
    df = pd.merge(df,destinations[['srch_destination_id','d33', 'd64', 'd52', 'd120', 'd72', 'd136', 'd7', 'd59', 'd50', 'd30']],on='srch_destination_id')
    df = df.dropna()
    return df

#Remove rows with the same user_id and item_id and different rating
def remove_duplicate(df):
    max_rating = df.groupby(['user_id', 'item_id']).rating.transform(max)
    df = df.loc[df.rating == max_rating]
    df.drop_duplicates(keep='first',inplace=True)
    return df

