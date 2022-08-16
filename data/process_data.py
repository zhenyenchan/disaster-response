import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''Load the messages and categories datasets and merge using the common id.
    
    Args:
    messages_filepath: filepath of the messages dataset (saved in CSV format)
    categories_filepath: filepath of the categories dataset (saved in CSV format)
    
    Returns:
    df: the merged dataset
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df

def clean_data(df):
    '''Clean the merged dataset in 4 steps:
    Step 1: Split the categories dataset into separate category columns
    Step 2: Convert category values to 0 or 1
    Step 3: Replace categories column in df with new category columns
    Step 4: Remote duplicates.
    
    Args: 
    df: the merged dataset with messages and categories data
    
    Returns:
    df_clean: the cleaned dataset ready for processing
    '''
    # step 1
    categories = df['categories'].str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = list(row.apply(lambda x: x[:-2]))
    categories.columns = category_colnames
    
    # step 2
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])
        categories[column] = categories[column].apply(lambda x: 1 if x == 2 else x)
    
    # step 3
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    
    # step 4
    df_clean = df.drop_duplicates()
    return df_clean

def save_data(df_clean, database_filename):
    '''Save the clean dataset into a sqlite database with the table name 'messages_categories'
    
    Args:
    df_clean: cleaned dataset with messages and categories data
    database_filename: filepath of the database to save the cleaned data 
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df_clean.to_sql('messages_categories', engine, if_exists='replace', index=False)  
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the sqlite filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()