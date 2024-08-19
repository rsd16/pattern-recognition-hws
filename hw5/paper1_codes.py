import pandas as pd
import numpy as np
from jgraph import *
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator, ClassifierMixin
from gensim.models import Word2Vec, KeyedVectors
from collections import Counter


class W2VTransformer(TransformerMixin):
    def __init__(self, model):
        self.model = model

    def transform(self, X, y=None, **fit_params):
        out_m = []
        for text in X:
            parag_M = []
            for token in text.split():
                if token in self.model:
                    parag_M.append(self.model[token])

            if parag_M:
                out_m.append(np.average(parag_M, axis=0))
            else:
                out_m.append(np.random.rand(1, 300)[0])

        return np.array(out_m)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


class TfidfEmbeddingVectorizer(TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None, **fit_params):
        tfidf = TfidfVectorizer()
        tfidf.fit(X)
        self.max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(int, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self

    def transform(self, X, y=None, **fit_params):
        out_m = []
        for text in X:
            parag_M = []
            for token in text.split():
                if token in self.model:
                    if token in self.word2weight:
                        parag_M.append(self.model[token] * self.word2weight[token])
                    else:
                        parag_M.append(self.model[token] * self.max_idf)

            if parag_M:
                out_m.append(np.average(parag_M, axis=0))
            else:
                out_m.append(np.random.rand(1, 300)[0])

        return np.array(out_m)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class ComplexNetworkClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=4):
        self.k = k

    def fit(self, X, y):
        eucl = euclidean_distances(X)
        k = self.k
        while True:
            simi_m = 1 / (1 + eucl)
            to_remove = simi_m.shape[0] - (k + 1)
            for vec in simi_m:
                vec[vec.argsort()[:to_remove]] = 0

            g = Graph.Weighted_Adjacency(simi_m.tolist(), mode=ADJ_UNDIRECTED, loops=False)
            if g.is_connected():
                break

            k += 1

        self.k = k
        comm = g.community_multilevel()
        self.y_comm = np.array(comm.membership)
        self.y = y
        self.X = X
        self.mapping = {}
        for c in list(set(comm.membership)):
            com_clas = self.y[self.y_comm==c]
            self.mapping[c] = Counter(com_clas).most_common(1)[0][0]

    def predict(self, X):
        y_pred = []
        for x in X:
            dists = euclidean_distances([x], self.X)[0]
            simi_m = 1 / (1 + dists)
            nearest_com = self.y_comm[simi_m.argsort()[-self.k:]]
            y_pred.append(self.mapping[Counter(nearest_com).most_common(1)[0][0]])

        return np.array(y_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

def main():
    # Load word2vec
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    # Lexicon load
    lex_file = open('/opinion-lexicon/negative-words.txt', 'r')
    neg_lexicon = lex_file.read().splitlines()
    lex_file.close()

    lex_file = open('/opinion-lexicon/positive-words.txt', 'r')
    pos_lexicon = lex_file.read().splitlines()
    lex_file.close()

    # Emoticons
    pos_emo = [':-)', ':)', '(-:', '(:', ':-]', '[-:', ':]', '[:', ':-d', ':>)', ':>d', '(<:', ':d', 'b-)', ';-)',
                '(-;', ';)', '(;', ';-d', ';>)', ';>d', '(>;', ';]', '=)', '(=', '=d', '=]', '[=', '(^_^)', '(^_~)',
                '^_^', '^_~', ':->', ':>', '8-)', '8)', ':-}', ':}', ':o)', ':c)', ':^)', '<-:', '<:', '(-8', '(8',
                '{-:', '{:', '(o:', '(^:', '=->', '=>', '=-}', '=}', '=o)', '=c)', '=^)', '<-=', '<=', '{-=', '{=',
                '(o=', '(^=', '8-]', '8]', ':o]', ':c]', ':^]', '[-8', '[8', '[o:', '[^:', '=o]', '=c]', '=^]', '[o=',
                '[^=', '8‑d', '8d', 'x‑d', 'xd', ':-))', '((-:', ';-))', '((-;', '=))', '((=', ':p', ';p', '=p']

    neg_emo = ['#-|', ':-&', ':&', ':-(', ')-:', '(t_t)', 't_t', '8-(', ')-8', '8(', ')8', '8o|', '|o8', ':$', ':\'(',
                ':\'-(', ':(', ':-/', ')\':', ')-\':', '):', '\-:', ':\'[', ':\'-[', ':-[', ']\':', ']-\':', ']-:',
                '=-(', '=-/', ')\'=', ')-\'=', ')-=', '\-=', ':-<', ':-c', ':-s', ':-x', ':-|', ':-||', ':/', ':<',
                ':[', ':o', ':|', '=(', '=[', '=\'(', '=\'[', ')=', ']=', '>-:', 'x-:', '|-:', '||-:', '\:', '>:', ']:',
                'o:', '|:', '=|', '=x', 'x=', '|=', '>:(', ':((', '):<', ')):', '>=(', '=((', ')=<', '))=', ':{', ':@',
                '}:', '@:', '={', '=@', '}=', '@=', 'd‑\':', 'd:<', 'd:', 'd8', 'd;', 'd=', 'd‑\'=', 'd=<', 'dx']

    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    le = LabelEncoder()
    le.fit(df_train['label'].values)

    x_train = df_train['text'].values
    y_train = le.transform(df_train['label'].values)

    x_test = df_test['text'].values
    y_test = le.transform(df_test['label'].values)

    pipelines = []

    pipe1 = make_pipeline(TfidfVectorizer(sublinear_tf=True), SVC(kernel='linear', probability=True))
    pipelines.append(('tfidf-SVM', pipe1))

    pipe62 = make_pipeline(W2VTransformer(model=model), SVC(kernel='linear', probability=True))
    pipelines.append(('w2v-SVM', pipe62))

    pipe71 = make_pipeline(TfidfEmbeddingVectorizer(model=model), LogisticRegression())
    pipelines.append(('w2vW-LR', pipe71))

    eclf = VotingClassifier(estimators=pipelines, voting='soft', n_jobs=4)

    eclf.fit(x_train, y_train)

    print('Ensemble ====')
    print('Accuracy for train-set: %.5f' % accuracy_score(y_train, eclf.predict(x_train)))
    print('Accuracy for test-set: %.5f' % accuracy_score(y_test, eclf.predict(x_test)))

    print('\nIndividual Evaluation')
    for estimator, (name, _) in zip(eclf.estimators_, eclf.estimators):
        print('Accuracy (%s): %.5f' % (name, accuracy_score(y_test, estimator.predict(x_test))))

if __name__ == '__main__':
    main()
