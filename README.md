# Disaster Response Pipeline Project

### Instructions:
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

