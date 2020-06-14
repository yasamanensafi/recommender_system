from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.utils import add_func
from deepctr.models import WDL
from deepctr.models import DeepFM
from deepctr.models import xDeepFM
import pandas as pd
import warnings
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import implicit
from utilities import util

def score(model,test_model_input,test):
    pred_ans = model.predict(test_model_input, batch_size=256)
    warnings.filterwarnings("ignore")
    new_df = test[['rating','item_id','user_id']]
    #replace the rating with algorithm generated output
    new_df['rating']=pred_ans
    #Create dataframe to store clusters
    return new_df

def create_hotel_df():
    hotel_df = pd.DataFrame(columns=['item_id','hotel_type'])
    hotel_df['item_id']=list(range(100))

    cluster = {"apartment":[5, 11, 22, 28,41, 56, 73],
            'business_hotels':[ 64,69, 70, 97],
            "condo":[3,8,36, 37, 55],
            "private_vacation_homes":[ 4, 9, 21, 49, 75, 77],
            "motel":[2,25,27, 95, 98],
            "beach_resort":[0, 17, 26, 31, 34, 80, 84, 92],
            "casino_hotel":[1, 19, 45, 54, 79,89, 93],
            "hotel_resort":[52, 65, 66, 87, 96],
            "bed_n_breakfast":[23, 39, 50, 51, 76],
            "hosetel":[12, 20, 38, 53, 57, 60, 61, 85, 86]}
    # store it on df
    warnings.filterwarnings("ignore")
    for i in cluster.keys():
        hotel_df['hotel_type'][cluster[i]]= i
    hotel_df = hotel_df.dropna().reset_index().drop('index',axis=1)
    return hotel_df,cluster

def als_model(new_df):
    #csr_matrix((data, (row, col))
    sparse_item_user = sparse.csr_matrix((new_df['rating'].astype(float),(new_df['item_id'], new_df['user_id'])))
    sparse_user_item = sparse.csr_matrix((new_df['rating'].astype(float),(new_df['user_id'], new_df['item_id'])))
    model = implicit.als.AlternatingLeastSquares(factors=20,regularization=0.1,iterations=20)
    alpha_val = 15
    data_conf = (sparse_item_user * alpha_val).astype('double')
    model.fit(data_conf)
    return model

def create_csv_df(hotel_df,new_df,cluster,als_model):
    csv_df = pd.DataFrame(columns= ['item_id','sim1','sim2','sim3'],
                      index=range(len(hotel_df['item_id'].unique())))

    #find 25 similar clusters for each item_id in hotel_df
    for i,x in enumerate(hotel_df['item_id'].unique()):
        similar = als_model.similar_items(x,25)
        a=similar
        #a= find_similar_clusters(x,25,new_df,als_model)
        #store them in a dataframe
        tt = pd.DataFrame(a, columns =['item_id', 'Score'])
        # keep only clusters that have different type with the current cluster
        for j in range(len(tt)):
            if tt['item_id'][j] in cluster[hotel_df['hotel_type'][i]]:
                tt=tt.drop([j])
        bb = tt.copy()
        # keep the top 5
        bb=bb.reset_index(drop=True)
        # keep only clusters that are available in hotel_df
        for k in range(len(bb)):
            if bb['item_id'][k] not in hotel_df['item_id']:
                bb=bb.drop([k])
        cc = bb.copy()
        cc= cc.reset_index(drop=True)
        csv_df["item_id"][i]=x
        csv_df["sim1"][i]=cc['item_id'][0]
        csv_df["sim2"][i]=cc['item_id'][1]
        csv_df["sim3"][i]=cc['item_id'][2]
    return csv_df