import numpy
import nltk
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_similarity_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin
import errno
import os
import pickle
from abc import ABCMeta, abstractmethod
from frozendict import frozendict


label_encoder = LabelEncoder()


class NBOWVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, aggregation, embeddings=None, word2idx=None):
        self.aggregation = aggregation
        self.embeddings = embeddings
        self.word2idx = word2idx
        self.dim = embeddings[0].size

    def aggregate_vecs(self, vectors):
        feats = []
        for method in self.aggregation:
            if method == 'sum':
                feats.append(numpy.sum(vectors, axis=0))

            if method == 'mean':
                feats.append(numpy.mean(vectors, axis=0))

            if method == 'min':
                feats.append(numpy.amin(vectors, axis=0))

            if method == 'max':
                feats.append(numpy.amax(vectors, axis=0))

        return numpy.hstack(feats)

    def transform(self, X, y=None):
        docs = []
        for doc in X:
            vectors = []
            for word in doc:
                if word not in self.word2idx:
                    continue

                vectors.append(self.embeddings[self.word2idx[word]])

            if len(vectors) == 0:
                vectors.append(numpy.zeros(self.dim))

            feats = self.aggregate_vecs(numpy.array(vectors))
            docs.append(feats)

        return docs

    def fit(self, X, y=None):
        return self


class ResourceManager(metaclass=ABCMeta):
    def __init__(self):
        self.wordvector_filename = ''
        self.parsed_filename = ''

    @abstractmethod
    def write(self):
        pass

    @abstractmethod
    def read(self):
        pass

    def read_hashable(self):
        return frozendict(self.read())


class WordVectorsManager(ResourceManager):
    def __init__(self, corpus=None, dim=None, omit_non_english=False):
        super().__init__()

        self.omit_non_english = omit_non_english
        self.wordvector_filename = '{}.{}d.txt'.format(corpus, str(dim))
        self.parsed_filename = '{}.{}d.pickle'.format(corpus, str(dim))

    def is_ascii(self, text):
        try:
            text.encode('ascii')
            return True
        except:
            return False

    def write(self):
        wordvector_file = os.path.join(os.path.dirname(__file__), self.wordvector_filename)
        if os.path.exists(wordvector_file):
            print('Indexing file {} ...'.format(self.wordvector_filename))
            embeddings_dict = {}
            f = open(wordvector_file, 'r', encoding='utf-8')
            for i, line in enumerate(f):
                values = line.split()
                word = values[0]
                coefs = numpy.asarray(values[1:], dtype='float32')
                if self.omit_non_english and not self.is_ascii(word):
                    continue

                embeddings_dict[word] = coefs

            f.close()
            print('Found %s word vectors.' % len(embeddings_dict))
            with open(os.path.join(os.path.dirname(__file__), self.parsed_filename), 'wb') as pickle_file:
                pickle.dump(embeddings_dict, pickle_file)
        else:
            print('{} not found!'.format(wordvector_file))
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), wordvector_file)

    def read(self):
        parsed_file = os.path.join(os.path.dirname(__file__), self.parsed_filename)
        if os.path.exists(parsed_file):
            with open(parsed_file, 'rb') as f:
                return pickle.load(f)
        else:
            self.write()
            return self.read()


def get_embeddings(corpus, dim):
    vectors = WordVectorsManager(corpus, dim).read()
    vocab_size = len(vectors)
    print('Loaded %s word vectors.' % vocab_size)
    wv_map = {}
    pos = 0
    emb_matrix = numpy.ndarray((vocab_size + 2, dim), dtype='float32')
    for i, (word, vector) in enumerate(vectors.items()):
        if len(vector) > 199:
            pos = i + 1
            wv_map[word] = pos
            emb_matrix[pos] = vector

    pos += 1
    wv_map['<unk>'] = pos
    emb_matrix[pos] = numpy.random.uniform(low=-0.05, high=0.05, size=dim)
    return emb_matrix, wv_ma

def eval_reg(y_hat, y):
    results = {'pearson': pearsonr([float(x) for x in y_hat], [float(x) for x in y])[0]}
    return results

def eval_clf(y_test, y_p):
    results = {'f1': f1_score(y_test, y_p, average='macro'), 'recall': recall_score(y_test, y_p, average='macro'),
               'precision': precision_score(y_test, y_p, average='macro'), 'accuracy': accuracy_score(y_test, y_p)}

    return results

def eval_mclf(y, y_hat):
    results = {'jaccard': jaccard_similarity_score(numpy.array(y), numpy.array(y_hat)),
               'f1-macro': f1_score(numpy.array(y), numpy.array(y_hat), average='macro'),
               'f1-micro': f1_score(numpy.array(y), numpy.array(y_hat), average='micro')}

    return results

def bow_model(task, max_features=10000):
    if task == 'clf':
        algo = LogisticRegression(C=0.6, random_state=0, class_weight='balanced')
    elif task == 'reg':
        algo = SVR(kernel='linear', C=0.6)
    else:
        raise ValueError('invalid task!')

    word_features = TfidfVectorizer(ngram_range=(1, 1), tokenizer=lambda x: x, analyzer='word', min_df=5, lowercase=False,
                                    use_idf=True, smooth_idf=True, max_features=max_features, sublinear_tf=True)

    model = Pipeline([('bow-feats', word_features), ('normalizer', Normalizer(norm='l2')), ('clf', algo)])

    return model

def nbow_model(task, embeddings, word2idx):
    if task == 'clf':
        algo = LogisticRegression(C=0.6, random_state=0, class_weight='balanced')
    elif task == 'reg':
        algo = SVR(kernel='linear', C=0.6)
    else:
        raise ValueError('invalid task!')

    embeddings_features = NBOWVectorizer(aggregation=['mean'], embeddings=embeddings, word2idx=word2idx, stopwords=False)

    model = Pipeline([('embeddings-feats', embeddings_features), ('normalizer', Normalizer(norm='l2')), ('clf', algo)])

    return model

def tok(text):
    return text

def main():
    wordvector_corpus = 'datastories.twitter'
    wordvector_dim = 300
    embeddings, word_indices = get_embeddings(corpus=wordvector_corpus, dim=wordvector_dim)

    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    x_train = train_data.drop(['id', 'label'], 1)
    y_train = train_data['label']
    x_test = test_data.drop(['id', 'label'], 1)
    y_test = test_data['label']

    y_train = np.array(y_train)
    y_train = label_encoder.fit_transform(y_train)
    y_train = utils.to_categorical(y_train, 3)

    y_test = np.array(y_test)
    y_test = label_encoder.fit_transform(y_test)
    y_test = utils.to_categorical(y_test, 3)

    print('LinearSVC')
    nbow = nbow_model('clf', embeddings, word_indices)
    nbow.fit(x_train, y_train)
    results = eval_clf(nbow.predict(x_test), y_test)
    for res, val in results.items():
        print('{}: {:.3f}'.format(res, val))

if __name__ == '__main__':
    main()
