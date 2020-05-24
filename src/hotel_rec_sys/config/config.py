# categ_sparse / conti_dense
sparse_features = ["site_name","posa_continent","user_location_country","user_id",
"is_mobile","is_package","channel","srch_adults_cnt","srch_children_cnt","srch_rm_cnt",
"srch_destination_id","srch_destination_type_id","cnt","hotel_continent","hotel_country",
"item_id","click_week","click_month","checkin_month","checkout_month","checkin_year",
"checkout_year","north_am_ci","north_am_co","kmeans_user_location_region","kmeans_user_location_city",
'couple_with_no_children','couple_with_one_child','couple_with_two_children',"empty_room","friends",
"single","single_parent",'d33', 'd64', 'd52', 'd120', 'd72', 'd136', 'd7', 'd59', 'd50', 'd30']

dense_features = ['hotel_market','log_orig_destination_distance']

# wide and deep
widendeep_att =  {
    "dnn_hidden_units":(2,2),
    "l2_reg_linear":0.1,
    "l2_reg_embedding":0.001,
    "l2_reg_dnn":0,
    "init_std":0.0001,
    "seed":1024,
    "task":'binary'
}

deepfm_att= {"dnn_hidden_units":(128,128),
            "init_std":0.0001,
            "seed":1024,
            "dnn_dropout":0.9,
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
 

                
    


