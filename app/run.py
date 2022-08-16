import json
import plotly
import pandas as pd
import nltk
import re
import string

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from pandas import DataFrame


app = Flask(__name__)

def tokenize(text):
    '''Process text data ie. normalise, tokenize and lemmatize
    
    Args:
    text: X containing messages
    
    Returns:
    clean_tokens: tokenized and lemmatized text data without punctuation and all in lower case
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Count number of genres and categories (the code below is provided by Udacity)
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    category_counts=[]
    category_names=list(df.columns[4:])
    
    # Count the number of messages assigned to each category (true values)
    for i in category_names:
        col_sum=df[df[i]==1].count()['message']
        category_counts.append(col_sum)
        
    # Count the number of categories that each message is assigned to
    df['num_categories']=0
    for j in category_names:
       df['num_categories']=df['num_categories']+df[j]
    
    num_categories = list(df['num_categories'].unique()).sort()
    num_categories_dist = df.groupby(['num_categories']).count()['message']
    
    # create visuals
    graphs = [
        # Plot messages by genre (the code below is provided by Udacity)
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Genres',
                'yaxis': {
                    'title': "Number of messages"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        # Plot messages by category
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Number of messages"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        
        # Plot categories per message and the total messages assigned to each category
        {
            'data': [
                Bar(
                    x=num_categories,
                    y=num_categories_dist
                )
            ],

            'layout': {
                'title': 'Distribution of Categories per Message',
                'yaxis': {
                    'title': "Number of messages"
                },
                'xaxis': {
                    'title': "Number of categories per message"
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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()