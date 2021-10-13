# Disaster Response Pipeline Project


## Table of Contents
1 Description  
2 Getting Started  
 - Dependencies
 - how to run  
3 data structure  
4 Screenshots  
## 1 Description
This project is the 2. part of the "Data Science-Nanodegree-Program" by Udacity.
The goal is to use prelabled text data to train a system to classify disaster-related information.
This is done in 3 steps:  
- preparing and cleaning text data, which has been classifies into 36 different categories. This is done using an ETL-pipeline.  
- build and train a model using this data and classification to classify unclassified text. This is done using an machine-learning-pipeline.  
- deploy this model by setting up a web-page including an online-classification and some graphical description of the data.  
## Getting started
### Dependencies  
  - Python 3
  - Pandas,
  - Numpy,
  - NLTK,
  - sqlalchemy,
  - scikit-learn,
  - plotly,
  - Flask  
### how to run
You can run the following commands in the project's directory to set up the database, train model and save the model.  
-  To run the ETL pipeline to clean data and store the processed data in the database, type:  
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db
-  To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file, type:  
python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl
-  to start the web app, type:  
python run.py
-  Go to http://0.0.0.0:3001/
## 3 data structure  
- app:  
 - template:
  - master.html # main page of web app  
  - go.html # classification result page of web app
  - run.py # Flask file that runs app
- data 
 - disaster_categories.csv # data to process  
 - disaster_messages.csv # data to process
 - process_data.py # data cleaning pipeline
 - DisasterResponse.db # database to save clean data to
- models
 - train_classifier.py # machine learning pipeline
 - classifier.pkl # saved model
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
