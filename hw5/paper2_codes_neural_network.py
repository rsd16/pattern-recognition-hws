import tensorflow as tf
from keras.constraints import maxnorm
from keras.engine import Input, Model
from keras.layers import Dropout, Dense, Bidirectional, LSTM, Embedding, GaussianNoise, Activation, Flatten, RepeatVector, MaxoutDense, GlobalMaxPooling1D, Convolution1D, MaxPooling1D, concatenate, Conv1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from kutilities.layers import AttentionWithContext, Attention, MeanOverTime
from sklearn import preprocessing
import pickle
import numpy
from sklearn.metrics import f1_score, precision_score
from sklearn.metrics import recall_score
from abc import ABCMeta, abstractmethod
from frozendict import frozendict


class ResourceManager(metaclass=ABCMeta):
    def __init__(self):
        self.wv_filename = ''
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
        self.wv_filename = '{}.{}d.txt'.format(corpus, str(dim))
        self.parsed_filename = '{}.{}d.pickle'.format(corpus, str(dim))

    def is_ascii(self, text):
        try:
            text.encode('ascii')
            return True
        except:
            return False

    def write(self):
        _word_vector_file = os.path.join(os.path.dirname(__file__), self.wv_filename)
        if os.path.exists(_word_vector_file):
            print('Indexing file {} ...'.format(self.wv_filename))
            embeddings_dict = {}
            f = open(_word_vector_file, 'r', encoding='utf-8')
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
            print('{} not found!'.format(_word_vector_file))
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), _word_vector_file)

    def read(self):
        _parsed_file = os.path.join(os.path.dirname(__file__), self.parsed_filename)
        if os.path.exists(_parsed_file):
            with open(_parsed_file, 'rb') as f:
                return pickle.load(f)
        else:
            self.write()
            return self.read()


def get_embeddings(corpus, dim):
    vectors = WordVectorsManager(corpus, dim).read()
    vocab_size = len(vectors)
    print(f'Loaded {vocab_size} word vectors.')
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
    return emb_matrix, wv_map


def embeddings_layer(max_length, embeddings, trainable=False, masking=False, scale=False, normalize=False):
    if scale:
        print('Scaling embedding weights...')
        embeddings = preprocessing.scale(embeddings)

    if normalize:
        print('Normalizing embedding weights...')
        embeddings = preprocessing.normalize(embeddings)

    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]

    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length if max_length > 0 else None, trainable=trainable,
                          mask_zero=masking if max_length > 0 else False, weights=[embeddings])

    return embedding

def get_RNN(unit=LSTM, cells=64, bi=False, return_sequences=True, dropout_U=0., consume_less='cpu', l2_reg=0):
    rnn = unit(cells, return_sequences=return_sequences, consume_less=consume_less, dropout_U=dropout_U, W_regularizer=l2(l2_reg))
    if bi:
        return Bidirectional(rnn)
    else:
        return rnn

def build_attention_RNN(embeddings, classes, max_length, unit=LSTM, cells=64, layers=1, **kwargs):
    bi = kwargs.get('bidirectional', False)
    noise = kwargs.get('noise', 0.)
    dropout_words = kwargs.get('dropout_words', 0)
    dropout_rnn = kwargs.get('dropout_rnn', 0)
    dropout_rnn_U = kwargs.get('dropout_rnn_U', 0)
    dropout_attention = kwargs.get('dropout_attention', 0)
    dropout_final = kwargs.get('dropout_final', 0)
    attention = kwargs.get('attention', None)
    final_layer = kwargs.get('final_layer', False)
    clipnorm = kwargs.get('clipnorm', 1)
    loss_l2 = kwargs.get('loss_l2', 0.)
    lr = kwargs.get('lr', 0.001)

    model = Sequential()

    model.add(embeddings_layer(max_length=max_length, embeddings=embeddings, trainable=False, masking=True, scale=False, normalize=False))

    if noise > 0:
        model.add(GaussianNoise(noise))

    if dropout_words > 0:
        model.add(Dropout(dropout_words))

    for i in range(layers):
        rs = (layers > 1 and i < layers - 1) or attention
        model.add(get_RNN(unit, cells, bi, return_sequences=rs, dropout_U=dropout_rnn_U))
        if dropout_rnn > 0:
            model.add(Dropout(dropout_rnn))

    if attention == 'memory':
        model.add(AttentionWithContext())
        if dropout_attention > 0:
            model.add(Dropout(dropout_attention))
    elif attention == 'simple':
        model.add(Attention())
        if dropout_attention > 0:
            model.add(Dropout(dropout_attention))

    if final_layer:
        model.add(MaxoutDense(100, W_constraint=maxnorm(2)))
        if dropout_final > 0:
            model.add(Dropout(dropout_final))

    model.add(Dense(classes, activity_regularizer=l2(loss_l2)))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr), loss='categorical_crossentropy')

    return model

def target_RNN(wv, tweet_max_length, aspect_max_length, classes=2, **kwargs):
    noise = kwargs.get('noise', 0)
    trainable = kwargs.get('trainable', False)
    rnn_size = kwargs.get('rnn_size', 75)
    rnn_type = kwargs.get('rnn_type', LSTM)
    final_size = kwargs.get('final_size', 100)
    final_type = kwargs.get('final_type', 'linear')
    use_final = kwargs.get('use_final', False)
    drop_text_input = kwargs.get('drop_text_input', 0.)
    drop_text_rnn = kwargs.get('drop_text_rnn', 0.)
    drop_text_rnn_U = kwargs.get('drop_text_rnn_U', 0.)
    drop_target_rnn = kwargs.get('drop_target_rnn', 0.)
    drop_rep = kwargs.get('drop_rep', 0.)
    drop_final = kwargs.get('drop_final', 0.)
    activity_l2 = kwargs.get('activity_l2', 0.)
    clipnorm = kwargs.get('clipnorm', 5)
    bi = kwargs.get('bi', False)
    lr = kwargs.get('lr', 0.001)

    attention = kwargs.get('attention', 'simple')

    shared_RNN = get_RNN(rnn_type, rnn_size, bi=bi, return_sequences=True, dropout_U=drop_text_rnn_U)

    input_tweet = Input(shape=[tweet_max_length], dtype='int32')
    input_aspect = Input(shape=[aspect_max_length], dtype='int32')

    tweets_emb = embeddings_layer(max_length=tweet_max_length, embeddings=wv, trainable=trainable, masking=True)(input_tweet)
    tweets_emb = GaussianNoise(noise)(tweets_emb)
    tweets_emb = Dropout(drop_text_input)(tweets_emb)

    aspects_emb = embeddings_layer(max_length=aspect_max_length, embeddings=wv, trainable=trainable, masking=True)(input_aspect)
    aspects_emb = GaussianNoise(noise)(aspects_emb)

    h_tweets = shared_RNN(tweets_emb)
    h_tweets = Dropout(drop_text_rnn)(h_tweets)

    h_aspects = shared_RNN(aspects_emb)
    h_aspects = Dropout(drop_target_rnn)(h_aspects)
    h_aspects = MeanOverTime()(h_aspects)
    h_aspects = RepeatVector(tweet_max_length)(h_aspects)

    representation = concatenate([h_tweets, h_aspects])

    att_layer = AttentionWithContext if attention == 'context' else Attention
    representation = att_layer()(representation)
    representation = Dropout(drop_rep)(representation)

    if use_final:
        if final_type == 'maxout':
            representation = MaxoutDense(final_size)(representation)
        else:
            representation = Dense(final_size, activation=final_type)(representation)

        representation = Dropout(drop_final)(representation)

    probabilities = Dense(1 if classes == 2 else classes, activation='sigmoid' if classes == 2 else 'softmax', activity_regularizer=l2(activity_l2))(representation)

    model = Model(input=[input_aspect, input_tweet], output=probabilities)

    loss = 'binary_crossentropy' if classes == 2 else 'categorical_crossentropy'
    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr), loss=loss)

    return model

def cnn_simple(wv, sent_length, **params):
    model = Sequential()

    model.add(embeddings_layer(max_length=sent_length, embeddings=wv, masking=False))

    model.add(Conv1D(activation='relu', filters=80, kernel_size=4, padding='valid'))

    model.add(GlobalMaxPooling1D())

    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model

def cnn_multi_filters(wv, sent_length, nfilters, nb_filters, **kwargs):
    noise = kwargs.get('noise', 0)
    trainable = kwargs.get('trainable', False)
    drop_text_input = kwargs.get('drop_text_input', 0.)
    drop_conv = kwargs.get('drop_conv', 0.)
    activity_l2 = kwargs.get('activity_l2', 0.)

    input_text = Input(shape=(sent_length,), dtype='int32')

    emb_text = embeddings_layer(max_length=sent_length, embeddings=wv, trainable=trainable, masking=False)(input_text)
    emb_text = GaussianNoise(noise)(emb_text)
    emb_text = Dropout(drop_text_input)(emb_text)

    pooling_reps = []
    for i in nfilters:
        feat_maps = Convolution1D(nb_filter=nb_filters, filter_length=i, border_mode='valid', activation='relu', subsample_length=1)(emb_text)
        pool_vecs = MaxPooling1D(pool_length=2)(feat_maps)
        pool_vecs = Flatten()(pool_vecs)
        pooling_reps.append(pool_vecs)

    representation = concatenate(pooling_reps)

    representation = Dropout(drop_conv)(representation)

    probabilities = Dense(3, activation='softmax', activity_regularizer=l2(activity_l2))(representation)

    model = Model(input=input_text, output=probabilities)
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model

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

wordvector_corpus = "datastories.twitter"
wordvector_dim = 300

max_length = 50

embeddings, word_indices = get_embeddings(corpus=wordvector_corpus, dim=wordvector_dim)

nn_model = build_attention_RNN(embeddings, classes=3, max_length=max_length, unit=LSTM, layers=2, cells=150, bidirectional=True, attention="simple",
                               noise=0.3, final_layer=False, dropout_final=0.5, dropout_attention=0.5, dropout_words=0.3, dropout_rnn=0.3,
                               dropout_rnn_U=0.3, clipnorm=1, lr=0.001, loss_l2=0.0001)

#nn_model = cnn_simple(embeddings, max_length)

#nn_model = cnn_multi_filters(embeddings, max_length, [3, 4, 5], 100, noise=0.1, drop_text_input=0.2, drop_conv=0.5, )

nn_model.summary()

history = nn_model.fit(x_train, y_train, validation_split=0.2, epochs=25, batch_size=128)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()

_, accuracy = nn_model.evaluate(x_test, y_test, verbose=2)
predictions = nn_model.predict(x_test)
predictions = np.argmax(predictions, axis=1)
print(accuracy)
