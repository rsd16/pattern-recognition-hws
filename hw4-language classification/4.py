import pandas as pd
import spacy
import itertools
import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
import warnings


warnings.filterwarnings('ignore')

data = pd.read_csv('dataMining.csv', encoding='utf8')
data = data.sample(frac=1).reset_index(drop=True)
#data = data.apply(lambda x: x.astype(str).str.lower())
#print(data.head())

X = data.drop('class', 1)
y = data['class']
#print(X.head())
#print(y.head())
#print(y.value_counts())

X = X.apply(lambda x: x.astype(str).str.lower())
#print(X.head())

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

def cv_designer(test_name, X, y, stopwords_list=None): # CountVectorizer with and without Stopwords.
    if stopwords_list == None:
        if test_name == 'char-bi':
            cnt_vct = CountVectorizer(analyzer='char', ngram_range=(2, 2))
        elif test_name == 'char-tri':
            cnt_vct = CountVectorizer(analyzer='char', ngram_range=(3, 3))
        elif test_name == 'word':
            cnt_vct = CountVectorizer(analyzer='word')
        elif test_name == 'word-bi':
            cnt_vct = CountVectorizer(analyzer='word', ngram_range=(2, 2))
        elif test_name == 'word-tri':
            cnt_vct = CountVectorizer(analyzer='word', ngram_range=(3, 3))
    else:
        if test_name == 'char-bi':
            cnt_vct = CountVectorizer(analyzer='char', ngram_range=(2, 2), stop_words=stopwords_list)
        elif test_name == 'char-tri':
            cnt_vct = CountVectorizer(analyzer='char', ngram_range=(3, 3), stop_words=stopwords_list)
        elif test_name == 'word':
            cnt_vct = CountVectorizer(analyzer='word', stop_words=stopwords_list)
        elif test_name == 'word-bi':
            cnt_vct = CountVectorizer(analyzer='word', ngram_range=(2, 2), stop_words=stopwords_list)
        elif test_name == 'word-tri':
            cnt_vct = CountVectorizer(analyzer='word', ngram_range=(3, 3), stop_words=stopwords_list)

    X_cv = cnt_vct.fit_transform(X['content'])
    #print(X_cv)
    #print(X_cv.shape)
    #print(cnt_vct.vocabulary_)
    #print(X_cv.toarray())
    #print(X_cv.shape)

    X = pd.DataFrame(X_cv.toarray(), columns=cnt_vct.get_feature_names())
    X.columns = [re.sub('[\(\[\],]', '', x) for x in X.columns] # Because of stemming results. ./
    X = X.loc[:, ~X.columns.duplicated()] # Because of stemming results. ./
    #print(X.shape)
    #print(X.head())
    #print(X.columns)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, shuffle=True)
    #print(x_train.head())
    #print(y_train.head())
    #print(x_test.head())
    #print(y_test.head())
    #print(x_train.shape)
    #print(y_train.shape)
    #print(x_test.shape)
    #print(y_test.shape)

    return x_train, x_test, y_train, y_test

def tf_designer(test_name, X, y, stopwords_list=None): # TF-IDF with and without Stopwords.
    if stopwords_list == None:
        if test_name == 'char-bi':
            tf_vct = TfidfVectorizer(analyzer='char', ngram_range=(2, 2))
        elif test_name == 'char-tri':
            tf_vct = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
        elif test_name == 'word':
            tf_vct = TfidfVectorizer(analyzer='word')
        elif test_name == 'word-bi':
            tf_vct = TfidfVectorizer(analyzer='word', ngram_range=(2, 2))
        elif test_name == 'word-tri':
            tf_vct = TfidfVectorizer(analyzer='word', ngram_range=(3, 3))
    else:
        if test_name == 'char-bi':
            tf_vct = TfidfVectorizer(analyzer='char', ngram_range=(2, 2), stop_words=stopwords_list)
        elif test_name == 'char-tri':
            tf_vct = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), stop_words=stopwords_list)
        elif test_name == 'word':
            tf_vct = TfidfVectorizer(analyzer='word', stop_words=stopwords_list)
        elif test_name == 'word-bi':
            tf_vct = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), stop_words=stopwords_list)
        elif test_name == 'word-tri':
            tf_vct = TfidfVectorizer(analyzer='word', ngram_range=(3, 3), stop_words=stopwords_list)

    X_tf = tf_vct.fit_transform(X['content'])
    #print(X_tf)
    #print(X_tf.shape)
    #print(tf_vct.vocabulary_)
    #print(X_tf.toarray())
    #print(X_tf.shape)

    X = pd.DataFrame(X_tf.toarray(), columns=tf_vct.get_feature_names())
    X.columns = [re.sub('[\(\[\],]', '', x) for x in X.columns]
    X = X.loc[:, ~X.columns.duplicated()]
    #print(X.shape)
    #print(X.head())
    #print(X.columns)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, shuffle=True)
    #print(x_train.head())
    #print(y_train.head())
    #print(x_test.head())
    #print(y_test.head())
    #print(x_train.shape)
    #print(y_train.shape)
    #print(x_test.shape)
    #print(y_test.shape)

    return x_train, x_test, y_train, y_test

def stemming(data):
    data['content'] = data['content'].apply(lambda sentence: word_tokenize(sentence))
    #print(data.head())

    english_stemmer = SnowballStemmer('english')
    german_stemmer = SnowballStemmer('german')
    spanish_stemmer = SnowballStemmer('spanish')
    french_stemmer = SnowballStemmer('french')

    for i, row in data.iterrows():
        if row['class'] == 'english':
            data.iloc[i][1] = [english_stemmer.stem(word) for word in row['content']]
        elif row['class'] == 'german':
            data.iloc[i][1] = [german_stemmer.stem(word) for word in row['content']]
        elif row['class'] == 'french':
            data.iloc[i][1] = [french_stemmer.stem(word) for word in row['content']]
        elif row['class'] == 'spanish':
            data.iloc[i][1] = [spanish_stemmer.stem(word) for word in row['content']]

    #data = data['content'].apply(lambda x: [porter_stemmer.stem(y) for y in x]) # Another approach, if we had only English sentences...

    X_stem = data.drop('class', 1)
    return X_stem

def lemmatization(data):
    english_nlp = spacy.load('en_core_web_sm')
    german_nlp = spacy.load('de_core_news_sm')
    spanish_nlp = spacy.load("es_core_news_sm")
    french_nlp = spacy.load('fr_core_news_sm')

    for i, row in data.iterrows():
        if row['class'] == 'english':
            data.iloc[i][1] = [word.lemma_ for word in english_nlp(row['content'])]
        elif row['class'] == 'german':
            data.iloc[i][1] = [word.lemma_ for word in german_nlp(row['content'])]
        elif row['class'] == 'french':
            data.iloc[i][1] = [word.lemma_ for word in french_nlp(row['content'])]
        elif row['class'] == 'spanish':
            data.iloc[i][1] = [word.lemma_ for word in spanish_nlp(row['content'])]

    X_lem = data.drop('class', 1)
    return X_lem

model_names = ['Random Forest', 'XGBoost', 'Decision Tree', 'AdaBoost', 'SVM', 'MultinomialNB']
tests_names = ['char-bi', 'char-tri', 'word', 'word-bi', 'word-tri']

params = {}
params['min_df'] = [0.1, 0.05, 0.075]
params['max_df'] = [0.25, 0.5, 0.75]
params['max_features'] = [250, 500, 750]
params['ngram_range'] = [(1, 1), (2, 2), (3, 3)]
params['analyzer'] = ['word', 'char']

stopwords_list = stopwords.words('english') + stopwords.words('french') + stopwords.words('german') + stopwords.words('spanish')

X_stem = pd.DataFrame(columns=['content'])
X_lem = pd.DataFrame(columns=['content'])
#print(X_stem.head())
#print(X_lem.head())

X_stem = stemming(data.copy())
X_stem = X_stem.apply(lambda x: x.astype(str).str.lower())
#print(X_stem.head())

X_lem = lemmatization(data.copy())
X_lem = X_lem.apply(lambda x: x.astype(str).str.lower())
#print(X_lem.head())

results_cv = pd.DataFrame(columns=model_names, index=tests_names) # Simple CountVectorizer
results_tf = pd.DataFrame(columns=model_names, index=tests_names) # Simple TF-IDF
results_tf_limited = pd.DataFrame(columns=model_names, index=tests_names)
results_sw_cv = pd.DataFrame(columns=model_names, index=tests_names) # CountVectorizer with Stopwords
results_sw_tf = pd.DataFrame(columns=model_names, index=tests_names) # TF-IDF with Stopwords
results_stem_cv = pd.DataFrame(columns=model_names, index=tests_names) # CountVectorizer with Stemming
results_stem_tf = pd.DataFrame(columns=model_names, index=tests_names) # TF-IDF with Stemming
results_lem_cv = pd.DataFrame(columns=model_names, index=tests_names) # CountVectorizer with Lemmatization
results_lem_tf = pd.DataFrame(columns=model_names, index=tests_names) # TF-IDF with Lemmatization
results_tf_min_df = pd.DataFrame(columns=model_names) # TF-IDF with different values for min_df parameter.
results_tf_max_df = pd.DataFrame(columns=model_names) # TF-IDF with different values for max_df parameter.
results_tf_max_features = pd.DataFrame(columns=model_names) # TF-IDF with different values for max_features parameter.

for model_name in model_names: # Simple CountVectorizer
    for test_name in tests_names:
        x_train, x_test, y_train, y_test = cv_designer(test_name, X, y)
        results_cv.loc[test_name][model_name] = '{:.2f}'.format(evaluate(model_name, x_train, x_test, y_train, y_test) * 100)

for model_name in model_names: # Simple TF-IDF
    for test_name in tests_names:
        x_train, x_test, y_train, y_test = tf_designer(test_name, X, y)
        results_tf.loc[test_name][model_name] = '{:.2f}'.format(evaluate(model_name, x_train, x_test, y_train, y_test) * 100)

for model_name in model_names: # CountVectorizer with Stopwords
    for test_name in tests_names:
        x_train, x_test, y_train, y_test = cv_designer(test_name, X, y, stopwords_list=stopwords_list)
        results_sw_cv.loc[test_name][model_name] = '{:.2f}'.format(evaluate(model_name, x_train, x_test, y_train, y_test) * 100)

for model_name in model_names: # TF-IDF with Stopwords
    for test_name in tests_names:
        x_train, x_test, y_train, y_test = tf_designer(test_name, X, y, stopwords_list=stopwords_list)
        results_sw_tf.loc[test_name][model_name] = '{:.2f}'.format(evaluate(model_name, x_train, x_test, y_train, y_test) * 100)

for model_name in model_names: # CountVectorizer with Stemming
    for test_name in tests_names:
        x_train, x_test, y_train, y_test = cv_designer(test_name, X_stem, y)
        results_stem_cv.loc[test_name][model_name] = '{:.2f}'.format(evaluate(model_name, x_train, x_test, y_train, y_test) * 100)

for model_name in model_names: # TF-IDF with Stemming
    for test_name in tests_names:
        x_train, x_test, y_train, y_test = tf_designer(test_name, X_stem, y)
        results_stem_tf.loc[test_name][model_name] = '{:.2f}'.format(evaluate(model_name, x_train, x_test, y_train, y_test) * 100)

for model_name in model_names: # CountVectorizer with Lemmatization
    for test_name in tests_names:
        x_train, x_test, y_train, y_test = cv_designer(test_name, X_lem, y)
        results_lem_cv.loc[test_name][model_name] = '{:.2f}'.format(evaluate(model_name, x_train, x_test, y_train, y_test) * 100)

for model_name in model_names: # TF-IDF with Lemmatization
    for test_name in tests_names:
        x_train, x_test, y_train, y_test = tf_designer(test_name, X_lem, y)
        results_lem_tf.loc[test_name][model_name] = '{:.2f}'.format(evaluate(model_name, x_train, x_test, y_train, y_test) * 100)

indices = []
for min_df in params['min_df']:
    for analyzer in params['analyzer']:
        for ngram_range in params['ngram_range']:
            if analyzer == 'char':
                if ngram_range == (1, 1):
                    pass

            tf_vct = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df)
            X_tf= tf_vct.fit_transform(X['content'])
            X_tf = pd.DataFrame(X_tf.toarray(), columns=tf_vct.get_feature_names())
            x_train, x_test, y_train, y_test = train_test_split(X_tf, y, test_size=0.33, stratify=y, shuffle=True)

            idx = f'min_df={min_df}, analyzer={analyzer}, ngram_range={ngram_range}'
            indices.append(idx)
            results_tf_min_df = results_tf_min_df.reindex(indices)

            for model_name in model_names:
                results_tf_min_df.loc[idx][model_name] = '{:.2f}'.format(evaluate(model_name, x_train, x_test, y_train, y_test) * 100)

indices = []
for max_df in params['max_df']:
    for analyzer in params['analyzer']:
        for ngram_range in params['ngram_range']:
            if analyzer == 'char':
                if ngram_range == (1, 1):
                    pass

            tf_vct = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, max_df=max_df)
            X_tf = tf_vct.fit_transform(X['content'])
            X_tf = pd.DataFrame(X_tf.toarray(), columns=tf_vct.get_feature_names())
            x_train, x_test, y_train, y_test = train_test_split(X_tf, y, test_size=0.33, stratify=y, shuffle=True)

            idx = f'max_df={max_df}, analyzer={analyzer}, ngram_range={ngram_range}'
            indices.append(idx)
            results_tf_max_df = results_tf_max_df.reindex(indices)

            for model_name in model_names:
                results_tf_max_df.loc[idx][model_name] = '{:.2f}'.format(evaluate(model_name, x_train, x_test, y_train, y_test) * 100)

indices = []
for max_feature in params['max_features']:
    for analyzer in params['analyzer']:
        for ngram_range in params['ngram_range']:
            if analyzer == 'char':
                if ngram_range == (1, 1):
                    pass

            tf_vct = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, max_features=max_feature)
            X_tf = tf_vct.fit_transform(X['content'])
            X_tf = pd.DataFrame(X_tf.toarray(), columns=tf_vct.get_feature_names())
            x_train, x_test, y_train, y_test = train_test_split(X_tf, y, test_size=0.33, stratify=y, shuffle=True)

            idx = f'max_features={max_feature}, analyzer={analyzer}, ngram_range={ngram_range}'
            indices.append(idx)
            results_tf_max_features = results_tf_max_features.reindex(indices)

            for model_name in model_names:
                results_tf_max_features.loc[idx][model_name] = '{:.2f}'.format(evaluate(model_name, x_train, x_test, y_train, y_test) * 100)

results_cv.to_csv('results_Simple_CountVectorizer.csv')
results_tf.to_csv('results_Simple_TF-IDF.csv')
results_sw_cv.to_csv('results_Stopwords_CountVectorizer.csv')
results_sw_tf.to_csv('results_Stopwords_TF-IDF.csv')
results_stem_cv.to_csv('results_Stemming_CountVectorizer.csv')
results_stem_tf.to_csv('results_Stemming_TF-IDF.csv')
results_lem_cv.to_csv('results_Lemmatization_CountVectorizer.csv')
results_lem_tf.to_csv('results_Lemmatization_TF-IDF.csv')
results_tf_min_df.to_csv('results_TF-IDF_with_min_df', sep='\t')
results_tf_max_df.to_csv('results_TF-IDF_with_max_df', sep='\t')
results_tf_max_features.to_csv('results_TF-IDF_with_max_features', sep='\t')
