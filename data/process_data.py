import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' 
    this reads the two csv files,
    merges them
    extract the categories

    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, how='inner', on='id')
    categories = df['categories'].str.split(";", expand=True)
    row = categories.iloc[0,:]

    # use  lambda to remove the final two characters from the category indicator
    slice = lambda x: x[:-2]
    category_colnames = row.transform(slice).tolist()

    categories.columns = category_colnames

    # lambda to get the value 
    slice = lambda x: x[-1:]
    
    # lambda to cast the original string to a proper integer
    datatype = lambda x: int(x)

    # do the transformations for each column
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].transform(slice)
    
        # convert column from string to numeric
        categories[column] = categories[column].transform(datatype)

    # replace 2 to mode value of 1 of related column
    categories.related=categories.related.replace(2, 1)
    
    # replace the original category column with the processed columns: drop + concat
    df.drop('categories', axis =1, inplace=True)
    df = pd.concat([df, categories], axis = 1)

    return df

def clean_data(df):
    '''
    remove duplicates
    '''
    d=sum(df.duplicated())

    # if there are duplicates, remove by using a pandas function
    if d > 0:
        df.drop_duplicates(keep='first', inplace=True)

    return df


def save_data(df, database_filename):
    '''
    write a dataframe to an sqlite database
    the main table in the database will be the database name without the db suffix
    '''
    sqlitepath = "sqlite:///"  + database_filename
    table_name = 'disaster_response'
    
    engine = create_engine(sqlitepath)
    df.to_sql(table_name, engine, if_exists='replace', index=False)  


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