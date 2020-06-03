import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
	'''
	Gets two CSV datafiles - messages and categories. Merges them on id and returns a dataframe.

	Args:
	messages_filepath - message csv file location
	categories_filepath - categories csv file location

	Returns:
	df - dataframe returned after merging two inputs on id
	'''


	messages = pd.read_csv(messages_filepath)
	categories = pd.read_csv(categories_filepath)
	df = messages.merge(categories, on='id')
	return df




def clean_data(df):
	'''
	Cleans the dataframe by making making and encoding category columns.

	Args:
	df - original dataframe returned after merging two datafiles

	Returns:
	df- clean dataframe, ready for processing
	'''
	
	categories = df.categories.str.split(';', expand=True)
	row = categories.iloc[0,:].values
	new_cols = [r[:-2] for r in row]
	categories.columns = new_cols
	
	for column in categories:
	
		categories[column] = categories[column].apply(lambda x : x.split("-")[-1])
	
		categories[column] = pd.to_numeric(categories[column])
	   
	df.drop('categories', axis=1, inplace = True)
	df = pd.concat([df,categories], axis=1)
	df = df.drop_duplicates()
	
	return df


def save_data(df, database_filename):
	'''
	Saves the dataframe in sqlite format.

	Args:
	df - the cleaned dataframe
	database_filename - location and filename used for saving the dataframe in sqlite format

	Returns:
	N/A
	'''
	engine = create_engine('sqlite://', echo=False)

	engine = create_engine('sqlite:///' + database_filename)
	df.to_sql('DisasterResponseDataset', engine, index=False, if_exists='replace')
	
	return df

def main():
	'''
	This functions creates a sqlite database by merging messages and category files.
	'''
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