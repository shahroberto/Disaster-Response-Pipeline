# Disaster Response Pipeline Project

## Table of Contents
1. [Project Summary](#summary)
2. [Instructions](#instructions)
3. [File Descriptions](#files)
4. [Acknowledgements](#acknowledgements)

## Project Summary<a name="motivation"></a>

In this project I'm analyzing disaster data from Figure Eight to build a model for an API that classifies disaster messages.

Using a data set containing real messages that were sent during disaster events I'll be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.  More specifically, the model is trained on a full pipeline beginning with tokenizing, stemming, and lemmatizing input text and a Multi Output Random Forrest Classifier to correctly identify the disaster response category.  

This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to https://SPACEID-3001.SPACEDOMAIN to view the web app (where SPACEID and SPACEDOMAIN can be found with the env | grep WORK terminal command)

## File Descriptions <a name="files"></a>

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- DisasterResponse.db # cleaned data input for machine learning
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py	# python file to clean raw data and create categories
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py  # train classifier for sentiment analysis
|- classifier.pkl  # saved model 

- README.md
```

## Acknowledgements
Figure Eight for the data and to Udacity for the course content.