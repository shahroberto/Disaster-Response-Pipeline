import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages and categories data specified by the filepaths and returns a dataframe
    merged by id.

        Parameters:
            messages_filepath (str): path to the messages data
            categories_filepath (str): path to the categories data

        Returns:
            merged dataframe (df): merged dataframe containing both messages and categories data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return messages.merge(categories, on='id')


def clean_data(df):
    """
    Returns a cleaned dataframe that maps disaster messages to categories as labels.

        Parameters:
            df (df): an input dataframe that is merged, raw messages and categories data

        Returns:
            cleaned df (df): a prepped data frame with messages data as an input mapped
            to a categorical one hot vector label set.
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[0:-2])
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1]).astype(str)
        categories[column] = pd.to_numeric(categories[column])

    df.drop('categories', axis=1, inplace=True)

    df = pd.concat([df, categories.reindex(df.index)], sort=False, axis=1)
    
    return df.drop_duplicates()


def save_data(df, database_filename):
    """Saves the dataframe df to the database specified by database_filename."""
    engine = create_engine(database_filename)
    df.to_sql('DisasterResponse', engine, index=False)


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
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()