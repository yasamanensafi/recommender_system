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
    return hotel_df

def find_similar_clusters(item_id,n_similar):
    similar = model.similar_items(item_id,n_similar)
    return similar

hotel_df = create_hotel_df()
