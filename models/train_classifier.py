import sys
from matplotlib.pyplot import table

#the basics
import numpy as np
import pandas as pd

import re
import pickle

# sql handling
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import classification_report, accuracy_score, precision_score, precision_recall_fscore_support, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import warnings

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

nltk.download(['punkt', 'wordnet', 'stopwords'])
warnings.simplefilter('ignore')

def load_data(database_filepath):
    '''
    reads the sqlite db file
    extracts the table name from the path (just as in process_data.py)

    extracts the category names and values

    returns the dta load ("X"), the values ("y") and the category names 
    
    '''

    sqlitepath = "sqlite:///"  + database_filepath
    table_name = 'disaster_response'

    
    engine = create_engine(sqlitepath)
    df = pd.read_sql_table(table_name, engine)
    X = df.message

    #  keep only the category columns as the resulting values
    y = df[df.columns[4:]]
    
    #y = df.drop(['id', 'message', 'genre', 'original'], axis = 1)

    category_names = y.columns

    return X, y, category_names




def tokenize(text):
    """
    inputs:
    messages (strings)
       
    Returns:
    normalized, stemmed tokens representing the input
    """
    # Converting everything to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # normalization word tokens and remove stop words
    normlizer = PorterStemmer()
    stop_words = stopwords.words("english")
    
    normalized = [normlizer.stem(word) for word in tokens if word not in stop_words]
    
    return normalized  


def build_model():
    '''
    Build a pipe line and 
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])


    # important: these values have been found when using the preparatory Jupyter notebook
    params_grid = {
        'clf__estimator__min_samples_split': [2], 
        'clf__estimator__n_estimators': [100]
    }
        

    #build the model
    #strangle an n_jobs value >1 leads to strange errors
    model = GridSearchCV(pipeline, param_grid = params_grid, verbose=2, cv=3)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    print the classification reports.
    the function only prints and doesn't return anything
    '''
    
    Y_pred = model.predict(X_test)

    # somehow Y_test contains the columns headers. This must be removed
    #Y_test = Y_test.iloc[1: , :]

    print("********")
    print("***category_names****", type(category_names))
    print(category_names)
    print("********")
    print("***Y test****", type(Y_test))
    print(Y_test)
    print("********")
    print("***Y pred****", type(Y_pred))
    print(Y_pred)



    #Y_test.to_csv('y_test.csv')
    #Y_pred.to_csv('y_pred.csv')

    class_report = classification_report(Y_test, Y_pred, target_names=category_names)
    
    print(class_report)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()