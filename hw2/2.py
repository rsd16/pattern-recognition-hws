import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
import collections
import random
from sklearn.decomposition import PCA


warnings.filterwarnings('ignore')

labelencoder = LabelEncoder() # initializing an object of class LabelEncoder


# First section, missing values:

# create missing values:
def create_missing(data, label_index):
    temp_class = data[label_index]
    data.drop(label_index, 1, inplace=True)
    replaced = collections.defaultdict(set)
    indices = [(row, column) for row in range(data.shape[0]) for column in range(data.shape[1])]
    random.shuffle(indices)
    to_replace = int(round(.2 * len(indices))) # 20 percent of all of the cell values.
    for row, column in indices:
        if len(replaced[row]) < data.shape[1] - 1:
            data.iloc[row, column] = np.nan
            to_replace -= 1
            replaced[row].add(column)
            if to_replace == 0:
                break

    data[label_index] = temp_class
    return data

# approach one, feature-based mean:
def missing_feature_based(data, label_index):
    data = create_missing(data, label_index)
    data.fillna(data.mean(), inplace=True)
    return data

# approach two, class-based mean:
def missing_class_based(data, label_index):
    data = create_missing(data, label_index)
    for column in data:
        data[column].fillna(data.loc[data[label_index].isin(data[label_index].value_counts().keys()), column].mean(), inplace=True)

    return data

# approach three, dropping the rows:
def missing_drop_rows(data, label_index):
    data = create_missing(data, label_index)
    data.dropna(inplace=True) # drop the rows where at least, one element is missing.
    return data


# Second section:

# part one, dimensionality reduction, approach one, Randomly:
def dimensionality_reduction(data, label_index):
    temp_class = data[label_index]
    data.drop(label_index, 1, inplace=True)
    num = len(data.columns) // 3 # about 33 percent of the features are dropped., except of course for class.
    for i in range(num):
        data.drop(np.random.choice(data.columns), 1, inplace=True)

    data[label_index] = temp_class
    return data

# part one, dimensionality_reduction, approach two, PCA:
def dimensionality_reduction_pca(data, label_index):
    temp_class = data[label_index]
    data.drop(label_index, 1, inplace=True)
    num = len(data.columns) // 3
    pca = PCA(n_components=num)
    principal_components = pca.fit_transform(data)
    data = pd.DataFrame(principal_components)
    data[label_index] = temp_class
    return data

# part two, datapoint reduction:
def datapoint_reduction(data, label_index):
    num = data.shape[0] // 3 # about 33 percent of datapoints are dropped.
    drop_indices = np.random.choice(data.index, num, replace=False)
    data.drop(drop_indices, inplace=True)
    return data


# Third section, noisy dataset:

# create noisy dataset:
def create_noisy(data, label_index):
    temp_class = data[label_index]
    data.drop(label_index, 1, inplace=True)
    replaced = collections.defaultdict(set)
    indices = [(row, column) for row in range(data.shape[0]) for column in range(data.shape[1])]
    random.shuffle(indices)
    to_replace = int(round(.2 * len(indices))) # 10 percent of values.
    for row, column in indices:
        if len(replaced[row]) < data.shape[1] - 1:
            data.iloc[row, column] = np.random.randint(-50, 50)
            to_replace -= 1
            replaced[row].add(column)
            if to_replace == 0:
                break

    data[label_index] = temp_class
    return data

# for classification:
def noisy(data, label_index):
    data = create_noisy(data, label_index)
    return data


def split_data(data, label_index):
    counts = data[label_index].value_counts()
    data = data[data[label_index].isin(counts[counts > 2].index)]
    x = data.drop(label_index, 1)
    y = data[label_index]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle=True, stratify=y)
    return x_train, x_test, y_train, y_test

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

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

dataset_names = {'banknote_authentication': 4, 'sonar': 60, 'iris': 4, 'Pima Indians Diabetes Dataset': 8, 'wine': 0, 'zoo': 16, 'lymphography': 0, 'ionosphere': 34, 'ecoli': 7, 'seeds': 7}
model_names = ['Random Forest', 'XGBoost', 'Decision Tree', 'AdaBoost', 'SVM']
test_segments = ['1. Missing, Feature-based mean', '1. Missing, Class-based mean', '2. Dimensionality Reduction, Randomly', '2. Dimensionality Reduction, PCA', '2. Datapoint Reduction, Randomly', '3. Noisy']

results_a = pd.DataFrame(columns=model_names, index=dataset_names.keys())
results_b = pd.DataFrame(columns=model_names, index=dataset_names.keys())
results_c = pd.DataFrame(columns=model_names, index=dataset_names.keys())
results_d = pd.DataFrame(columns=model_names, index=dataset_names.keys())
results_e = pd.DataFrame(columns=model_names, index=dataset_names.keys())
results_f = pd.DataFrame(columns=model_names, index=dataset_names.keys())

for dataset_name, label_index in dataset_names.items():
    data = pd.read_csv(f'{dataset_name}' + '.data', header=None)
    for test_segment in test_segments:
        if test_segment == '1. Missing, Feature-based mean':
            data = missing_feature_based(data, label_index)
        elif test_segment == '1. Missing, Class-based mean':
            data = missing_class_based(data, label_index)
        elif test_segment == '2. Dimensionality Reduction, Randomly':
            data = dimensionality_reduction(data, label_index)
        elif test_segment == '2. Dimensionality Reduction, PCA':
            data = dimensionality_reduction_pca(data, label_index)
        elif test_segment == '2. Datapoint Reduction, Randomly':
            data = datapoint_reduction(data, label_index)
        elif test_segment == '3. Noisy':
            data = noisy(data, label_index)
        #elif test_segment == '1. Missing, Dropping the rows':
        #    data = missing_drop_rows(data, label_index)

        x_train, x_test, y_train, y_test = split_data(data, label_index)
        for model_name in model_names:
            accuracy = '{:.2f}'.format(evaluate(model_name, x_train, x_test, y_train, y_test) * 100)
            #print(f'Accuracy for dataset: "{dataset_name}" from "{model_name}" model is: {accuracy}')
            if test_segment == '1. Missing, Feature-based mean':
                results_a.loc[dataset_name][model_name] = accuracy
            elif test_segment == '1. Missing, Class-based mean':
                results_b.loc[dataset_name][model_name] = accuracy
            elif test_segment == '2. Dimensionality Reduction, Randomly':
                results_c.loc[dataset_name][model_name] = accuracy
            elif test_segment == '2. Dimensionality Reduction, PCA':
                results_d.loc[dataset_name][model_name] = accuracy
            elif test_segment == '2. Datapoint Reduction, Randomly':
                results_e.loc[dataset_name][model_name] = accuracy
            elif test_segment == '3. Noisy':
                results_f.loc[dataset_name][model_name] = accuracy
            #elif test_segment == '1. Missing, Dropping the rows':
            #    results_g.loc[dataset_name][model_name] = accuracy
            #with open(test_segment + '.txt', 'w') as file:
                #file.write(f'Accuracy for dataset: "{dataset_name}" from "{model_name}" model is: {accuracy}')
                #file.write('\n')

results_a.to_csv('(results)1. Missing, Feature-based mean.csv')
results_b.to_csv('(results)1. Missing, Class-based mean.csv')
results_c.to_csv('(results)2. Dimensionality Reduction, Randomly.csv')
results_d.to_csv('(results)2. Dimensionality Reduction, PCA.csv')
results_e.to_csv('(results)2. Datapoint Reduction, Randomly.csv')
results_f.to_csv('(results)3. Noisy.csv')
