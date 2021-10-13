import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from joblib import dump

def load_data(database_filepath):
    # load data from database
    engine = create_engine(database_filepath)

    # read table
    df = pd.read_sql('InsertTableName', engine)
    
    # message column -> feature
    X = df.message
    
    # message category columns -> target
    y = df.drop(['id','message','original','genre'], axis=1)
    
    # target headlines
    category_names = y.columns


def tokenize(text):
    # separite words
    tokens = word_tokenize(text)
    
    # create lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        # find lemma for each word, set to lower cast, remove blanks
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        
        # add to array
        clean_tokens.append(clean_token)

    return clean_tokens

def build_model():
    # build pipeline:
    # small Ramdom Forest
    # => little number of trees
    
    pipeline1 = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 100,random_state=1)))
        ])
    
    return pipeline1


def evaluate_model(model, X_test, Y_test, category_names):
    # make prediction of test-data using optimized model
    y_pred = model.predict(X_test)
    
    # create classification_report for each category
    for i, category_name in enumerate(category_names):
            print('classification_report: ' + category_name)
            print(classification_report(Y_test[category_name], y_pred[:, i]))
            
    # create over-all classification_report
    print('over-all classification_report: ')
    print(classification_report(Y_test, y_pred, target_names = category_names))


def save_model(model, model_filepath):
    dump(model,model_filepath)


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