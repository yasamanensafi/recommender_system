import numpy as np
import pandas as pd

model_epoch = {"epoch":8}

#hyper-parameter tuning

param_rand = {'dnn_hidden_units' : [(1,1),(2,2),(4,4),(32,32),(128,128),(256,256)],
              'l2_reg_linear':[1e-5,1e-3,1e-1,1,10],
              'l2_reg_embedding':[1e-7,1e-5,1e-3,1e-1,1],
              'l2_reg_dnn':[0,0.2,2,4],
              'dnn_dropout':[0,0.2,0.4,0.5,0.6,0.8]
             }

cluster = {"user_region_n_cluster" : 3,"user_city_n_cluster": 3 }

#Scale

scaling = {'min_new':0.1, 'max_new':1.5}

# categ_sparse / conti_dense

target = ['rating']

sparse_features = ["site_name", #ID of the Expedia point of sale (i.e. Expedia.com, Expedia.co.uk, Expedia.co.jp, …)
"posa_continent", #ID of continent associated with site_name
"user_location_country", #The ID of the country the customer is located
"kmeans_user_location_region", #The ID of the region the customer is located clustered in 2 groups
"kmeans_user_location_city", #The ID of the city the customer is located clustered in 2 groups
"user_id", #ID of user
"is_mobile", #1 when a user connected from a mobile device, 0 otherwise
"is_package", #1 if the click/booking was generated as a part of a package (i.e. combined with a flight), 0 otherwise
"channel", #ID of a marketing channel
"cnt", #Numer of similar events in the context of the same user session
"srch_destination_id", #ID of the destination where the hotel search was performed'
"srch_destination_type_id", #Type of destination
"hotel_continent", #'Hotel continent',
"hotel_country", #Hotel country
"item_id", #(hotel_cluster)ID of a hotel cluster
"north_am_ci", # 1 if check-in date it's a holiday in north America
"north_am_co",# 1 if check-out date it's a holiday in north America
'hotel_market', #Hotel market
'couple_with_no_children','couple_with_one_child','couple_with_two_children',"friends","single","single_parent",
#hotel search latent attributes highly correlated with rating:
'd33', 'd64','d52','d120', 'd72', 'd136', 'd7', 'd59', 'd50', 'd30'] 


dense_features = ["srch_adults_cnt", #The number of adults specified in the hotel room
"srch_children_cnt", #The number of (extra occupancy) children specified in the hotel room
"srch_rm_cnt", #The number of hotel rooms specified in the search
'log_orig_destination_distance', # Log transformed physical distance between a hotel and a customer at the time of search
"click_week",
"click_month",
"checkin_month",
"checkout_month",
"checkin_year",
"checkout_year"]

# wide and deep
widendeep_att =  {
    "dnn_hidden_units":(2,2),
    #"l2_reg_linear":0.1, 
    #"l2_reg_embedding":0.001, 
    #"l2_reg_dnn":4,
    "init_std":0.0001,
    "seed":1024,
    "task":'binary',
    "dnn_dropout": 0.4,
    "dnn_activation":'relu'
}
                     

deepfm_att= {"dnn_hidden_units":(128,128),
            "init_std":0.0001,
            "seed":1024,
            "dnn_dropout":0.5,
            "dnn_activation":'relu',
            "task":'binary',
            "fm_group":['default_group'],
            "dnn_use_bn":False}

xdeepfm_att= {"dnn_hidden_units":(256, 256),
                "init_std":0.0001,
                "cin_layer_size":(128, 128),
                "cin_split_half":True,
                "cin_activation":'relu',
                "l2_reg_cin":1e-07,
                "seed":1024,
                "dnn_dropout":0.5,
                "dnn_activation":'relu',
                "task":'binary',
                "dnn_use_bn":False}
 

                
    


