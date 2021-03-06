# import libraries
import pandas as pd
import sqlalchemy
import sys


def load_data(messages_filepath, categories_filepath):
    """
    Loads data csv-files:
        - disaster_categories.csv
        - disaster_messages.csv .
    
    Parameters:
    - Filepaths as mentioned above.
    
    Returns:
    -
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    return(messages.merge(categories))

def clean_data(df):
    """
    cleans dataframe:
        - extracts categories and create columns for each category
        - changes 2 to 0 in category 'relevant'
        - removes category 'child_alone'
    
    Parameters:
    - dataframe to be cleaned.
    
    Returns:
    - cleaned dataframe.
    """
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df.categories.str.split(';', expand = True))
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.str.split('-', expand = True)[0]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        
        # there are some '2's... , these rows contain no data
        categories[column] = categories[column].replace('2','0')
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # there is no message with category 'child_alone', so drop it...
    categories.drop('child_alone', axis = 1, inplace = True)
    
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)
    
    # check number of duplicates
    #df.duplicated().sum()
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # check number of duplicates
    #df.duplicated().sum()

    return(df)

def save_data(df, database_filename):
    """
    saves dataframe to SQL-db
    
    Parameters:
    - dataframe,
    - filename of SQL-db
    
    Returns:
    -
    """
    engine = sqlalchemy.create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse_db', engine, index=False)  


def main():
    """
    main funtions, flow control
    
    Parameters:
    -
    
    Returns:
    -
    """
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