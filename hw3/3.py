import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings


warnings.filterwarnings('ignore')

def read_split_data(dataset_name, label_index):
    data = pd.read_csv(f'{dataset_name}' + '.data', header=None)
    counts = data[label_index].value_counts()
    data = data[data[label_index].isin(counts[counts > 5].index)]
    x = data.drop(label_index, 1)
    y = data[label_index]
    return x, y

dataset_names = {'banknote_authentication': 4, 'sonar': 60, 'iris': 4, 'Pima Indians Diabetes Dataset': 8, 'wine': 0, 'zoo': 16, 'lymphography': 0, 'ionosphere': 34, 'ecoli': 7, 'seeds': 7}

params = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, None]}
grid = GridSearchCV(DecisionTreeClassifier(), param_grid=params, n_jobs=-1, cv=10)

for dataset_name, label_index in dataset_names.items():
    x, y = read_split_data(dataset_name, label_index)
    grid.fit(x, y)
    results = pd.DataFrame(grid.cv_results_)
    results = results.filter(['params', 'mean_test_score'])
    results['mean_test_score'] = results['mean_test_score'].apply(lambda x: '{:.2f}'.format(x * 100))
    results.to_csv('(results) ' + dataset_name + '.csv', index=None, sep='\t')
