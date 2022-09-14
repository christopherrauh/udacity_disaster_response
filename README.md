# Disaster Response Pipeline Project

# Introduction 

This project has two purposes:

It will classify messages according to a given set of aid- / disaster specific attributes. It can be utilized to quickly categorize incoming messages of an aid providing organization. This will help save time and resources.

On top of that some metrics of the foundation data set are being displayed.

To realize the web app, as preparation steps labelled data must be processed in an ETL pipeline and a classifier must be created from the output of the ETL pipeline in an ML pipeline.

## Setup

The project is devided into the following functions:

### data preparation

This part of the project will clean classified messages and process them together with a given set of categories for further machine learning tasks.

### classifier

Based on the prepared data, an ML model will be created that can later be used to classify new incoming messages.

### web app

Web interface to classify new messages accoring to the given set of categories. Also, some metrics of the base data set are being displayed:

- Amount of messages that can be attributed with the clusters: aid, infrastructure, natural disaster
- Distribution of all categories for 

## Included files
This is a description of the files comprising the project.

### data preparation

data folder:

- disaster_categories.csv # data to process (given classification of the messages)
- disaster_messages.csv - data to process (messages)
- process_data.py - processing code


### classifier

models folder:

- train_classifier.py - code to set up the classifier
- classifier.pkl # saved model, to be created


### web app

app folder:
- run.py # Flask file that runs web app

app/template folder:
- master.html # main page of web app
- go.html # classification result page of web app

README.md: This file

### Instructions from the Udacity project description on how to execute the various sub processes
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    Make sure the classifier.pkl file is present

3. Go to http://0.0.0.0:3001/



### Required libraries:

- numpy
- pandas
- scikitlearn
- nltk
- flask
- plotly

Attention: Some Python environments do not work with sklearn.externals.joblib and need to import joblib directly

### Changes not mentioned in the requirements

master.html contains a refence to a plotly package that at least in my environment didn't work. I updated accordingly.

