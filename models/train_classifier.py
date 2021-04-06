import sys
import pandas as pd
from sqlalchemy import create_engine

import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    """
    Loads the data from the database and splits into base data and labels.

        Parameters:
            database_filepath (str): path to database

        Returns:
            X (df): base data with disaster messages
            y (df): categorical labels that classify messages based on type
            category_names (list): list of message category label names
    """
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql('select * from DisasterResponse', engine)
    X = df.message.values
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values

    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    """
    Case normalize, lemmatize, and tokenize the message and returns a cleaned list string.

        Parameters:
            text (str): input disaster response message

        Returns:
            clean_tokens (list): tokenized, cleaned, lemmatized message TFIDF list.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Returns a sklearn pipeline that transforms the message/category data that first vectorizes then applies 
    TFIDF and finally multioutput random forest.  Grid Search is performed to find the optimal hyperparameters.

        Parameters:

        Returns:
            model (pipeline): gridsearch cv pipeline model that can be used to predict the label of a disaster response message
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 1.0),
        'vect__max_features': (None, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__max_depth': (None, 10, 20),
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluates trained model on the testing data and prints relevent metrics.
    
        Parameters:
            model (sklearn model): model trained on testing data and tuned with grid search CV
            X_test (df): test messages
            y_test (df): categorical message labels for the test messages
            category_names (list): category_names (list): list of message category label names

        Returns:

        Prints model Precision, Recall, F1, Support
    """
    y_pred = model.predict(X_test)

    for i in range(y_test.shape[1]):
        print(classification_report(y_test[:, i], y_pred[:, i], labels=category_names[i])


def save_model(model, model_filepath):
    """Save thoe model to a pickle file specified by model_filepath."""
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, model_filepath)


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