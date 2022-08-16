import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier


def load_data(database_filepath):
    '''Load cleaned data from database and split into features and target
    
    Args:
    database_filepath: filepath of the disaster messages database 
    
    Returns:
    X: features for modelling (message)
    Y: target columns for modelling (categories)
    category_names: names of target columns
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages_categories', con=engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


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

def build_model():
    '''Build the machine learning pipeline in 3 steps:
    Step 1: Structure the pipeline with text transformers and multi-output decision tree classifier
    Step 2: Specify parameters for grid search
    Step 3: Create grid search object
    
    Returns:
    model: model object ready to be fit on X_train and Y_train
    '''
    # step 1
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('moc', MultiOutputClassifier(RandomForestClassifier()))
                ])
    
    # step 2
    parameters = {
        'vect__ngram_range' : ((1, 1), (1, 2)),
        'tfidf__smooth_idf' : [True, False],
        'moc__estimator__max_depth' : [40, 50]
    }
    
    # step 3
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluate the model on test data
    
    Args:
    model: the model that has been fit on training data
    X_test: test features dataset
    Y_test: test target datset
    category_names: names of target columns
    
    Returns:
    result = classification report with precision, recall, F1 score and support
    '''
    Y_pred = model.predict(X_test)
    result = classification_report(Y_test, Y_pred, target_names=category_names)
    return result

def save_model(model, model_filepath):
    '''Export model as a pickle file
    
    Args:
    model: model to be exported
    model_filepath: filepath of the pickle file to save the model to (ending with .pkl)
    '''
    pickle.dump(model, open('classifier.pkl','wb'))


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
        result = evaluate_model(model, X_test, Y_test, category_names)
        print(result)

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