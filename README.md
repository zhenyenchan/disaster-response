# disaster-response
Text classifier pipeline and Flask app to visualise disaster messages data - Data Scientist nanodegree project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [How to interact with this project](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Libraries used: sys nltk re string numpy pandas pickle sklearn flask json plotly sqlalchemy. Python version: 3.7.12

## Project Motivation<a name="motivation"></a>

Following a disaster, disaster response organisations get millions of communications via social media, when they have the least capacity to filter the messages that are the most important. Different organisations look after different parts of the problem e.g. water, blocked roads, medical supplies.

This project analyses disaster data from Appen to build a model for a web app that classifies disaster messages. The app allows an emergency worker to input a new message and get classification results in several categories. It also visualises the distribution of messages across genres and categories and the total number of messages that are assigned to a given number of categories.

## File Descriptions <a name="files"></a>

+ data/disaster_categories.csv data/disaster_messages.csv: messages and categories datasets
+ data/process_data.py: ETL pipeline outputs a SQL database stored in the 'data' folder.
+ model/train_classifier.py: ML pipeline which trains the classifier and saves it as a pickle file in the 'models' folder.
+ app/templates/go.html: html webpage that receives user input and displays model results
+ app/templates/master.html: html webpage with visualisations
+ app/run.py: runs the Flask app
+ notebooks/ETL_Pipeline_Preparation.ipynb: notebook used to explore and analyse the data for cleaning
+ notebooks/ML_Pipeline_Preparation.ipynb: notebook used to explore different classifier algorithms and pipeline structures

## How to interact with this project <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
The dataset was sourced from Appen. Some of the functions used in the code were taken from exercises as part of the Udacity Data Scientist nanodegree 2022.
