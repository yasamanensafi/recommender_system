# Hybrid Recommender Systems for Personalized Travel Destinations

This project was conducted under the supervision of Amir Tavasoli at the SharpestMinds fellowship program.
## Context
Nowadays, online reservations have become very popular and travellers find it much easier to book hotels of their choice online. However, people have a difficult time choosing the hotel that they want to book given the effectively limitless choices most major destinations have to offer. Recommender systems are a viable tool to create a better user experience in such a context.

The goals of the following project are to build a hybrid recommender system using state-of-the-art models such as Wide and Deep, DeepFM, and XDeepFM and also, identify similar hotel clusters for each of them.  

##  The Expedia dataset

The dataset that has been used on this project is from the Kaggle [Expedia Hotel Recommendations](https://www.kaggle.com/c/expedia-hotel-recommendations/data) competition. However, we looked at this dataset from another perspective and tackled a different problem.
The training set consists of 37,670,293 rows from 2013 to 2014 and we trained our algorithm on a small subset of the whole dataset (1 million rows). Also, some latent features for each of the destinations recorded and the top most correlated features were chosen to be used. 
Each row of data represents a different hotel search performed by a user looking for a specific destination on specific dates. 

## Notebooks

###  **1.0-hotel-feature-eng.ipynb**
The first step was to perform exploratory data analysis to extract useful insights and clean and pre-process the dataset.
In order to simplify the features names and understand them better in all of the notebooks, "hotel_cluster" was changed to "item_id" and  "is_booking" was changed to "rating" and was considered as an implicit rating.
Moreover, feature engineering was performed and many new features such as holidays, duration, family_status and etc. were extracted from dataset. Also, the KMeans clustering algorithm was used on some features such as user_location_region and user_location_city to convert each of them to 2 clusters. 

### **2.0-hotel-linear.ipynb**

The two basic architectures for a recommendation system are _Content-Based_ and _Collaborative-Filtering models_. Content-Based models recommend items based on how similar this item is to the other items. 
On the other hand, Collaborative-Filtering models are entirely based on past behaviours and focus on the relationship between users and items.

For this project, we used Collaborative-Filtering model as the baseline algorithm which is implemented in this notebook.
First of all, the rows with the same user_id and item_id that have different ratings were removed. Then user-item matrices for train and test set were created which can be used to calculate the similarity between users and items.
*Note that these matrices should be the same size but filled with different data (ratings on train and test).
 After that, two collaborative filtering models were implemented from scratch.
- Memory-Based CF by computing cosine similarity:
The Memory-Based model uses either user-user similarity or item-item similarity in order to make recommendations from the ratings matrix.*

- Model-Based CF by using singular value decomposition (SVD) and Alternating Least Squares (ALS) method:
*These types of CF models are developed using ML algorithms to predict a user's rating of unrated items. To name a few of these algorithms, we can mention KNN, SVD, Matrix Factorization and etc.*

### **3.0-hotel-fastai.ipynb**
**This notebook should be placed and run in the Recommenders folder ([reco library environment](https://github.com/microsoft/recommenders))**.
The aim of this notebook is to make the same Collaborative-Filtering recommendation system but using the FastAI library. 
First of all, we create a CollabDataBunch, which is a databunch specifically created for collaborative filtering problems. 
Then, we used EmbeddingDotBias which is provided by FastAI and it creates embeddings for both users and items and then takes the dot product of them. This model can be created using the collab_learner class.
In this model we could recommend hotel clusters to a user and also score the model to find the top K recommendations.

### **4.0-hotel-wideNdeep.ipynb**
The problem with our baseline model (collaborative-filtering recommender system) was that we couldn't use more features. Therefore, we decided to build hybrid recommender systems.
Wide & Deep is a hybrid model that joins trained wide linear models and deep neural networks to combine the benefits of memorization and generalization for recommender systems.

![enter image description here](https://1.bp.blogspot.com/-Dw1mB9am1l8/V3MgtOzp3uI/AAAAAAAABGs/mP-3nZQCjWwdk6qCa5WraSpK8A7rSPj3ACLcB/s1600/image04.png)

As the left side of the above figure shows, the wide network is a single layer feed-forward network which assigns weights to each feature and adds bias to them to model the matrix factorization method. The deep model is a feed forward neural network as shown in the above figure. The combined wide and deep model takes the weighted sum of the outputs from both wide model and deep model as the prediction value.
In this notebook, we implemented the wide and deep model using [DeepCTR](https://pypi.org/project/deepctr/) package. After the features were divided into sparse (categorical) and dense (continuous) features, we applied Label Encoding for sparse features, and normalization for dense numerical features. Then, the model was defined and trained and its performance was evaluated using the most common metric systems: RMSE, MAE, and AUC.
It should be mentioned that we found the best model attributes by performing a semi-manual hyper-parameter tuning.
### **5.0-hotel-deepfm.ipynb**
Compared to the latest Wide & Deep model from Google, DeepFM has a shared raw feature input to both its “wide” and “deep” components, with no need for feature engineering besides raw features. The wide component of DeepFM is an FM layer. The Deep Component of DeepFM can be any neural network.
![enter image description here](https://deepctr-doc.readthedocs.io/en/latest/_images/DeepFM.png)
In this notebook, same as the wide and deep notebook we divided features to sparse and dense, defined the model, trained it and also applied hyper-parameter tuning. 
### **6.0-hotel-xdeepfm.ipynb**
XDeepFM (eXtreme Deep Factorization Machine) uses a Compressed Interaction Network (CIN) to learn both low and high order feature interaction explicitly, and uses a classical DNN (MLP) to learn feature interaction implicitly. 
![enter image description here](https://deepctr-doc.readthedocs.io/en/latest/_images/xDeepFM.png)
Also, in this notebook, same as the wide and deep and deepFM notebooks, we divided features to sparse and dense, defined the model, trained it and also applied hyper-parameter tuning. 

### **7.0-hotel-similar_clusters.ipynb**

After creating a collaborative-filtering recommender system as a baseline and developing three hybrid recommender systems, we decided to tackle a new problem and identify similar clusters (hotel types). 
The Expedia hotel dataset is an anonymized dataset and 
the true values of some features such as item_id (or previously hotel_cluster) are hidden behind integer codes. Inspired by [The locations puzzle kernel](https://www.kaggle.com/dvasyukova/the-locations-puzzle) we tried to identify the location of some of the hotels with the help of user location country, region, city and most importantly, orig_destination_distance which represents the distance between the hotel and the user. 
After finding the location of some of the hotel_markets (New York. Las Vegas, Cancún, London, Miami, ...) and combing the results with the most common Accommodation Types in that city (that extracted from Expedia website), we can guess the hotel type of that cluster. For example, we know that hotel_market ID of Hawaii is 212. Also, we extracted from data that the most common hotel_cluster for Hawaii is 0. Therefore, considering the information that we gained from Expedia website, we assigned a name to hotel_cluster = 0 and we named it  "beach resort". 
We took this finding further and after finding the top most similar clusters to 0, we assigned beach resorts to them too. 
We applied this method to a few more clusters, came up with 10 categories such as apartment, business_hotels, bed_n_breakfast, and etc and covered more than 60 item_ids out of 100. 

### **8.0-hotel-top_n_items_deepfm.ipynb**

In the last notebook, after training one of our hybrid recommender system models (here we used DeepFM), we tried to find the similar clusters based on the results of DeepFM model and in order to do this we used implicit library. After that, we compared the results to the defined  clusters from the previous notebook, and checked if the there are similar. Fortunately, the results confirmed our assumptions.
In the end, as an output, a csv file was generated that consists of item_id of the hotels and the three most similar clusters to them. It should be mentioned that for each item_id we excluded the recommended clusters that are in the same cluster category. For example, on the notebook 7, we assigned the category "apartment" to cluster number 5. In the list of the five most similar clusters to this one, we have  apartment, motel, apartment, casino_hotel, and private_vacation_homes. We chose to only recommend a motel, an apartment and a casino_hotel to the user. 
Finally, we recommend 5 hotel clusters to each user. 

## src/hotel_rec_sys

This folder consists of several modules that have been created by dividing the program into different parts.
The hyper-parameter tuning here is a semi-manual process. If you would like to find the optimal hyper-parameters for the model, after running the [run_tuning.py](https://github.com/yasamanensafi/recommender_system/blob/master/src/hotel_rec_sys/run_tuning.py "run_tuning.py"), you need to place the results to the model dictionary at [config.py](https://github.com/yasamanensafi/recommender_system/blob/master/src/hotel_rec_sys/config/config.py) file.
By running the  [run.py](https://github.com/yasamanensafi/recommender_system/blob/master/src/hotel_rec_sys/run.py) the program will 
- Load the datasets
- Perform pre-processing
- Perform feature engineering
- Split the data into trainset and testset
- Create Wide and Deep, DeepFM, and XDeepFM models.
- Score each model and choose the best one.
- Load the best model
- Find the 3 most similar clusters to each item_id based on the prediction of the model
- Generate output.csv
## Results
The following table shows a sample of results from the generated CSV file which consists of item_id of the hotels and the three most similar clusters to them.
It should be mentioned that for each item_id the recommended clusters that are in the same cluster category were excluded.

| item_id | Hotel type | Similar Hotel 1 | Similar Hotel 2 |Similar Hotel 3 |
| :---         |     :---:      |     :---:      |               :---:      |          ---: |
| 0 | beach_resort   | hotel_resort    |hotel_resort|private_vacation_homes  |
| 4 | private_vacation_homes  | apartment  |apartment  |casino_hotel  |
| 12| hosetel  | bed_n_breakfast  |condo  |condo  |
| 25| motel  | apartment  |apartment  |condo  |
| 52| hotel_resort  | beach_resort  |beach_resort  |beach_resort  |

Finally, 5 hotel clusters were recommended to each user. The following table, shows the recommendation for user_id=800. 

| item_id (hotel_cluster) |Hotel Type |
| :---         |     ---: |    
| 39 | bed_n_breakfast   | 
| 65 | hotel_resort  | 
| 77| private_vacation_homes  |
| 26| beach_resort  | 
| 51| bed_n_breakfast  |