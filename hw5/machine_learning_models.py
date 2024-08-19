import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import itertools
import re
import nltk
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
import warnings


def evaluate(model_name, x_train, x_test, y_train, y_test):
    if model_name == 'Random Forest':
        model = RandomForestClassifier()
    elif model_name == 'XGBoost':
        model = xgb.XGBClassifier()
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier()
    elif model_name == 'AdaBoost':
        model = AdaBoostClassifier()
    elif model_name == 'SVM':
        model = SVC()
    elif model_name == 'MultinomialNB':
        model = MultinomialNB()

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

def cv_designer(X, min_df, max_features):
    cnt_vct = CountVectorizer(analyzer='word', min_df=min_df, max_features=max_features)
    X_cv = cnt_vct.fit_transform(X['text'])
    X = pd.DataFrame(X_cv.toarray(), columns=cnt_vct.get_feature_names())
    x_train = X[:50333]
    x_test = X[50333:]
    return x_train, x_test

def tf_designer(X, min_df, max_features):
    tf_vct = TfidfVectorizer(analyzer='word', min_df=min_df, max_features=max_features)
    X_tf = tf_vct.fit_transform(X['text'])
    X = pd.DataFrame(X_tf.toarray(), columns=tf_vct.get_feature_names())
    x_train = X[:50333]
    x_test = X[50333:]
    return x_train, x_test

def main():
    min_df = 10
    max_features = 10000
    model_names = ['Random Forest', 'XGBoost', 'Decision Tree', 'AdaBoost', 'SVM', 'MultinomialNB']
    results = pd.DataFrame(columns=model_names, index=['CountVectorizer', 'TF-IDF'])

    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    #print(train_data.shape)
    #print(test_data.shape)
    #print(train_data.head())
    #print(test_data.head())

    x_train = train_data.drop(['id', 'label'], 1)
    y_train = train_data['label']
    x_test = test_data.drop(['id', 'label'], 1)
    y_test = test_data['label']
    #print(x_train.shape)
    #print(y_train.shape)
    #print(x_test.shape)
    #print(y_test.shape)
    #print(x_train.head())
    #print(y_train.head())
    #print(x_test.head())
    #print(y_test.head())

    X = x_train.append(x_test, ignore_index=True)

    for model_name in model_names:
        x_train, x_test = cv_designer(X, min_df, max_features)
        results.loc['CountVectorizer'][model_name] = '{:.2f}'.format(evaluate(model_name, x_train, x_test, y_train, y_test) * 100)
        x_train, x_test = tf_designer(X, min_df, max_features)
        results.loc['TF-IDF'][model_name] = '{:.2f}'.format(evaluate(model_name, x_train, x_test, y_train, y_test) * 100)
        print('Done: ', model_name)

    results.to_csv('results_dumb_approaches.csv')

if __name__ == '__main__':
    main()
