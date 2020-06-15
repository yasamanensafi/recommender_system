# Hybrid Recommender Systems for Personalized Travel Destinations

## Context
Nowadays, online reservations have become very popular and travellers find it much easier to book hotels of their choice online. However, people have a difficult time choosing the hotel that they want to book. Therefore, the recommender system comes into play.

The goals of the following project are to build a hybrid recommender system using state-of-the-art models such as Wide and Deep, DeepFM, and XDeepFM and also, identify similar hotel clusters for each of them.  

##  The Expedia dataset

The dataset that has been used on this project is from the Kaggle [Expedia Hotel Recommendations](https://www.kaggle.com/c/expedia-hotel-recommendations/data) competition. However, we looked at this dataset from another perspective and tackled a different problem.
The training set consists of 37,670,293 rows from 2013 to 2014 and we trained our algorithm on a small subset of the whole dataset (1 million rows). Also, some latent features for each of the destinations recorded and the top most correlated features were chosen to be used. 
Each row of data represents a different hotel search performed by a user looking for a specific destination on specific dates. 

## Notebooks

###  **1.0-hotel-feature-eng.ipynb**
The first step was to clean and pre-process the data and perform exploratory analysis to extract useful insights.
In order to simplify the features names and understand them better in all of the notebooks, "hotel_cluster" was changed to "item_id" and  "is_booking" was changed to "rating" and was considered as an implicit rating.
Moreover, feature engineering was performed and many new features such as holidays, duration, family_status and etc. were extracted from dataset. Also, the KMeans clustering algorithm was used on some features such as user_location_region and user_location_city to convert each of them to 2 clusters. 

### **2.0-hotel-linear.ipynb**

The two basic architectures for a recommendation system are _Content-Based_ and _Collaborative-Filtering models_. Content-Based models recommend items based on how similar this item is to the other items. 
On the other hand, Collaborative-Filtering models are entirely based on past behaviours and focus on the relationship between users and items.

For this project, we used Collaborative-Filtering model as the baseline algorithm which is implemented in this notebook.
First of all, the rows with the same user_id and item_id that have different ratings were removed. Then user-item matrices for train and test set were created which can be used to calculate the similarity between users and items. *Note that these matrices should be the same size but filled with different data (ratings on train and test ).
 After that, two collaborative filtering models were implemented from scratch.
- Memory-Based CF by computing cosine similarity:
T*he Memory-Based model uses either user-user similarity or item-item similarity in order to make recommendations from the ratings matrix.*

- Model-Based CF by using singular value decomposition (SVD) and ALS method:
*These types of CF models are developed using ML algorithms to predict a user's rating of unrated items. To name a few of these algorithms, we can mention KNN, Singular Value Decomposition (SVD), Matrix Factorization (ALS) and etc.*

### **3.0-hotel-fastai.ipynb**
**This notebook should be placed and run in the Recommenders folder ([reco library environment](https://github.com/microsoft/recommenders))**.
The aim of this notebook is to make the same Collaborative-Filtering recommendation system but using the FastAI library. 
First of all, we create a CollabDataBunch, which is a databunch specifically created for collaborative filtering problems. 
Then, we used EmbeddingDotBias which is provided by FastAI and it creates embeddings for both users and items and then takes the dot product of them. This model can be created using the collab_learner class.
In this model we could recommend hotel clusters to a user and also score the model to find the top K recommendations.

### **4.0-hotel-wideNdeep.ipynb**