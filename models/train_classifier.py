# importing libraries

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
import pickle
nltk.download('punkt')
nltk.download('wordnet')


# load data from database
def load_data(database_filepath):
	'''
	Load data from sqlite and split it into target and features dataframe

	Args:
	database_filepath - location of sqlite database

	Returns:
	X - array of features
	Y - array of targets
	category_names - target labels
	'''
	print(database_filepath)
	engine = create_engine('sqlite:///' + database_filepath)

	df = pd.read_sql_query('SELECT * FROM DisasterResponseDataset', engine)

	# split into X and Y
	X = df['message']
	Y = df.iloc[:, 4:]
	category_names = Y.columns.values

	return X.values, Y.values, category_names


def tokenize(text):
	'''
	Tokenizes and Lemmatizes test.

	Args:
	text - untokenized text

	Returns:
	clean_tokens - tokenized text ready for processing
	'''
	tokens = nltk.word_tokenize(text)
	lemmatizer = nltk.WordNetLemmatizer()

	clean_tokens = []
	for tok in tokens:
		clean_tok = lemmatizer.lemmatize(tok).lower().strip()
		clean_tokens.append(clean_tok)

	return clean_tokens


def build_model():
	'''
	Builds a pipeline model

	Args:
	N/A

	Returns:
	model - the model pipeline
	'''
	pipeline = Pipeline([
						('vect', CountVectorizer(tokenizer=tokenize)),
						('tfidf', TfidfTransformer()),
						('clf', MultiOutputClassifier(RandomForestClassifier()))
						])

	parameters = {
				  #'tfidf__use_idf': (True, False),
				  'vect__ngram_range': ((1, 1), (1, 2)),
				  'clf__estimator__n_estimators': [50, 100],
				 }
	
	return GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs = -1)

def evaluate_model(model, X_test, Y_test, category_names):
	'''
	Evaluates model on test data

	Args:
	model - model trained
	X_test - Test features dataframe
	Y_test - Test targets dataframe
	category_names - Target labels

	Returns:
	N/A
	'''
	predictions = model.predict(X_test)
	for i in range(predictions.shape[1]):
		print(category_names[i])
		print(classification_report(Y_test[:, i], predictions[:, i]))

def save_model(model, model_filepath):
	'''
	Saves model in pickle format

	Args:
	model - the trained model
	model_filepath - location where to save model

	Returns:
	N/A
	'''
	with open(model_filepath, 'wb') as file:
		pickle.dump(model, file)


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