import json
import plotly
import pandas as pd
import sys

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from flask import Flask
from flask import render_template, request, jsonify

from plotly.graph_objs import Bar

#this is to circumvent verison conflicts
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib



from sqlalchemy import create_engine


aid_categories = ['aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food' , 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid']
infrastructure_categories = ['infrastructure_related', 'transport' , 'buildings', 'electricity', 'tools', 'hospitals' , 'shops', 'aid_centers', 'other_infrastructure']
natural_disaster_categories = ['weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather']

app = Flask(__name__)

def tokenize(text):
    '''
    Tokenizes a given text incl. lemmatizing and transforming to lower case

    Inputs:
    text - to be tokenized

    Returns:
    clean_tokens: corresponding tokens

    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def no_subset(df, cluster):
    '''
    calculates the number of rows of a given subset of a Pandas data frame, where binary values are actually set to 1 

    Inputs:
    df: Pandas dataframe to consider
    cluster: list of column names constituting a logical cluster

    Returns:
    number_of_rows: the amount of rows where the given cluster of attributes has binary values set to 1

    '''
    temp_df = df[cluster]     
    number_of_rows = len(temp_df[temp_df.isin([1]).any(axis=1)])

    return number_of_rows


# load data
engine = create_engine('sqlite:///./data/disaster_response.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("./models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''
    this function is called to render the front page of the web application
    specifically 3 metrics are calculated and passed on to the flask engine to be rendered as a Plotly graph
    '''
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    
    # get number of messages per message cluster. There can be overlaps in rows
    # meta information, such as id and related columns have been removed from the sets
    no_aid = no_subset(df, aid_categories)
    no_infrastructure = no_subset(df, infrastructure_categories)
    no_disaster = no_subset(df, natural_disaster_categories)
    
    cluster_counts=[no_aid, no_infrastructure, no_disaster]
    cluster_names = ['aid related', 'infrastructure related', 'natural disaster related'] 

    # get the cateories distribution in 'Direct' genre
    # drop the columns id and related as they carry only meta information
    direct_df = df[df.genre == 'direct'].drop(['id', 'related'], axis=1)
    direct_no_rows = len(direct_df)
    direct_category_counts = (direct_df.mean() * direct_no_rows).sort_values(ascending = False)
    direct_category_names = list(direct_category_counts.index)

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # visualization #2: distribution of message clusters
        {
            'data': [
                Bar(
                    x=cluster_names,
                    y=cluster_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Emergency types (cluster)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Cluster"
                }
            }
        },
        # visualization #3: sorted categories in genre direct
        {
            'data': [
                Bar(
                    x=direct_category_names,
                    y=direct_category_counts
                )
            ],

            'layout': {
                'title': 'Categories in genre Direct',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }



    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    '''
    method to perform the classification of the entered message 
    '''
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    '''
    the main function initiales the Flask framework and starts the web server
    '''
    #this package needs a download and cannot be handled by an import 
    nltk.download('omw-1.4')
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()