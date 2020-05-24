import holidays
import numpy as np
import pandas as pd
import datetime
from sklearn.cluster import KMeans

def extract_week(df,feature,week):
    df[feature] =  pd.to_datetime(df[feature], infer_datetime_format=True)
    df[feature] = df.date_time.dt.strftime('%Y-%m-%d')
    d = datetime.timedelta(days=14)
    df['lag_date_time'] = df[feature].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d") + d)
    df['week'] = pd.DatetimeIndex(df['lag_date_time']).week
    df['year']=pd.DatetimeIndex(df['lag_date_time']).year
    
    # countinue week numbers for the next year
    df[week] = df['week'].where(df['year'] ==2013 , df['week']+52)
    return df

def extract_month_year(df):
    # extract month from date_time
    df['click_month'] = pd.DatetimeIndex(df['date_time']).month
    df['checkin_month'] = pd.DatetimeIndex(df['srch_ci']).month
    df['checkout_month'] = pd.DatetimeIndex(df['srch_co']).month
    df['checkin_year'] = pd.DatetimeIndex(df['srch_ci']).year
    df['checkout_year'] = pd.DatetimeIndex(df['srch_co']).year
    return df

def add_holiday(df):
    # Define holidays in some countries
    ca_holidays = holidays.Canada()
    us_holidays = holidays.UnitedStates()
    
    # check if checkin or checkout date is in holiday of different countries
    df['north_am_ci'] = df['srch_ci'].apply(lambda x: 1 if x in (us_holidays or ca_holidays)  else 0)
    df['north_am_co'] = df['srch_co'].apply(lambda x: 1 if x in (us_holidays or ca_holidays)  else 0)    
    # remove original columns
    df= df.drop(['date_time'],axis=1)
    df= df.drop(['week'],axis=1)
    df= df.drop(['year'],axis=1)
    df= df.drop(['srch_ci'],axis=1)
    df= df.drop(['srch_co'],axis=1)
    df= df.drop(['lag_date_time'],axis=1)
    return df

def log_transform(df,feature):
    #Note that we add 1 to the raw count to prevent the logarithm from
    # exploding into negative infinity in case the count is zero.
    df['log_orig_destination_distance'] = np.log10(df[feature] + 1)
    df= df.drop([feature],axis=1)
    return df

def create_cluster(df,feature,n_clusters):
    y = df[feature]
    X = df.drop(feature,axis=1)
    kmeansmodel = KMeans(n_clusters= n_clusters, init='k-means++', random_state=0)
    y_kmeans= kmeansmodel.fit_predict(X)
    df['kmeans_'+feature]=y_kmeans
    df= df.drop([feature],axis=1)
    return df

def extract_family_status(df):
    condlist = [(df['srch_adults_cnt']==0) & (df['srch_children_cnt']==0),
            (df['srch_adults_cnt']==2) & (df['srch_children_cnt']==0),
            (df['srch_adults_cnt']==2) & (df['srch_children_cnt']==1),
            (df['srch_adults_cnt']==2) & (df['srch_children_cnt']==2),
           (df['srch_adults_cnt']==1) & (df['srch_children_cnt']==0),
            (df['srch_adults_cnt']>1) & (df['srch_children_cnt']>0),
           (df['srch_adults_cnt']==1) & (df['srch_children_cnt'] > 0),
           (df['srch_adults_cnt']>2) & (df['srch_children_cnt'] == 0),
           (df['srch_adults_cnt']==0) & (df['srch_children_cnt'] > 0)]

    choicelist = ['empty_room',
                'couple_with_no_children',
                'couple_with_one_child',
                'couple_with_two_children',
                'single',
                'big_family',
                'single_parent',
                'friends',
                'unsupervised_children']

    df['family_status'] = np.select(condlist,choicelist)
    #Convert the family_status into dummy variables
    dummies = pd.get_dummies(df['family_status'],drop_first=True)
    df= pd.concat( [df.drop('family_status',axis=1),dummies],axis=1)
    df= df.drop("unsupervised_children",axis=1)
    return df