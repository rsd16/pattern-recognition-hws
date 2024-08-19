import pandas as pd
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

def cv_designer(test_name, X, y):
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

    X_cv = cnt_vct.fit_transform(X['content'])
    #print(X_cv)
    #print(X_cv.shape)
    #print(cnt_vct.vocabulary_)
    #print(X_cv.toarray())
    #print(X_cv.shape)

    X = pd.DataFrame(X_cv.toarray(), columns=cnt_vct.get_feature_names())
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

def tf_designer(test_name, X, y):
    if test_name == 'unigram':
        tf_vct = TfidfVectorizer(ngram_range=(1, 1))
    elif test_name == 'bigram':
        tf_vct = TfidfVectorizer(ngram_range=(2, 2))
    elif test_name == 'trigram':
        tf_vct = TfidfVectorizer(ngram_range=(3, 3))

    X_tf = tf_vct.fit_transform(X['content'])
    #print(X_tf)
    #print(X_tf.shape)
    #print(tf_vct.vocabulary_)
    #print(X_tf.toarray())
    #print(X_tf.shape)

    X = pd.DataFrame(X_tf.toarray(), columns=tf_vct.get_feature_names())
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

model_names = ['Random Forest', 'XGBoost', 'Decision Tree', 'AdaBoost', 'SVM', 'MultinomialNB']
tests_names_cv = ['char-bi', 'char-tri', 'word', 'word-bi', 'word-tri']
tests_names_tf = ['unigram', 'bigram', 'trigram']

results_cv = pd.DataFrame(columns=model_names, index=tests_names_cv)
results_tf = pd.DataFrame(columns=model_names, index=tests_names_tf)

for model_name in model_names:
    for test_name in tests_names_cv:
        x_train, x_test, y_train, y_test = cv_designer(test_name, X, y)
        results_cv.loc[test_name][model_name] = '{:.2f}'.format(evaluate(model_name, x_train, x_test, y_train, y_test) * 100)

for model_name in model_names:
    for test_name in tests_names_tf:
        x_train, x_test, y_train, y_test = tf_designer(test_name, X, y)
        results_tf.loc[test_name][model_name] = '{:.2f}'.format(evaluate(model_name, x_train, x_test, y_train, y_test) * 100)

#print(results)
results_cv.to_csv('results_CountVectorizer.csv')
results_tf.to_csv('results_Tf-Idf.csv')
