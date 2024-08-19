import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D, Conv1D, ZeroPadding1D, MaxPool2D, MaxPooling1D, Conv2D, MaxPooling2D, MaxPool1D, Reshape, Concatenate, Flatten, Input, Dropout, Bidirectional, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, utils
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.optimizers import Adam


# Line 30427 in train.csv has a null value for 'text' column, sometimes.

label_encoder = LabelEncoder()

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

y_train = np.array(y_train)
y_train = label_encoder.fit_transform(y_train)
y_train = utils.to_categorical(y_train, 3)

y_test = np.array(y_test)
y_test = label_encoder.fit_transform(y_test)
y_test = utils.to_categorical(y_test, 3)

#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)

#print(x_train.head())
#print(y_train.head())
#print(x_test.head())
#print(y_test.head())

max_length = 64
embedding_dim = 300

counter_train = Counter()

for sentence in x_train['text']:
    for word in sentence.split():
        counter_train[word] += 1

num_words_train = len(counter_train)
#print(num_words_train)

tokenizer_train = Tokenizer(num_words=num_words_train)
tokenizer_train.fit_on_texts(x_train['text'])
word_index_train = tokenizer_train.word_index
train_sequences = tokenizer_train.texts_to_sequences(x_train['text'])

x_test_encoded = list()
for sentence in x_test['text']:
    x_test = [tokenizer_train.word_index[w] for word in sentence if word in tokenizer_train.word_index]
    x_test_encoded.append(x_test)

x_train = pad_sequences(train_sequences, maxlen=max_length, padding='post')
x_test = pad_sequences(x_test_encoded, maxlen=max_length, padding='post')

embeddings_index = {}
file = open('glove.6B.300d.txt', encoding='utf-8')
for line in file:
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

file.close()

print(f'Number of Tokens from GloVe: {len(embeddings_index)}')

words_not_found = []
vocab = len(tokenizer_train.word_index) + 1
embedding_matrix = np.random.uniform(-0.25, 0.25, size=(vocab, embedding_dim)) # 0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones.
for word, i in tokenizer_train.word_index.items():
    if i >= vocab:
        continue

    embedding_vector = embeddings_index.get(word)

    if (embedding_vector is not None) and (len(embedding_vector) > 0):
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)

print(f'Shape of embedding matrix: {str(embedding_matrix.shape)}')
print(f'Number of words not found in GloVe: {len(words_not_found)}')

def cnn1d_model():
    model = Sequential()

    model.add(Embedding(len(tokenizer_train.word_index) + 1, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False))

    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters=512, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.25))

    model.add(Conv1D(filters=1024, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=1024, kernel_size=2, activation='relu', padding='same'))
    model.add(Dropout(0.3))

    model.add(.Conv1D(filters=512, kernel_size=2, activation='relu', padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))
    model.add(layers.Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

def lstm_model():
    model = Sequential()

    model.add(Embedding(len(tokenizer_train.word_index) + 1, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False))

    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

def bi_lstm_model():
    model = Sequential()

    model.add(Embedding(len(tokenizer_train.word_index) + 1, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False))

    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(LSTM(128))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

def cnn1d_lstm_model():
    model = Sequential()

    model.add(Embedding(len(tokenizer_train.word_index) + 1, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False))

    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters=512, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.25))

    model.add(Conv1D(filters=1024, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=1024, kernel_size=2, activation='relu', padding='same'))
    model.add(Dropout(0.3))

    model.add(.Conv1D(filters=512, kernel_size=2, activation='relu', padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))
    model.add(layers.Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

def creative_cnn1d_model():
    inputs = Input(shape=(max_length,), dtype='int32')

    X_input = Embedding(input_dim=(len(tokenizer_train.word_index) + 1), output_dim=embedding_dim, weights=[embedding_matrix],
                        input_length=max_length, trainable=False)(inputs)

    X_input = Reshape((max_length, embedding_dim, 1))(X_input)

    X1 = Conv1D(512, kernel_size=2, padding='same', kernel_initializer='normal', activation='relu', name='conv1Filter1')(X_input)
    maxpool_1 = MaxPool1D(pool_size=2, strides=1, padding='same', name='maxpool1')(X1)

    X2 = Conv1D(512, kernel_size=4, padding='same', kernel_initializer='normal', activation='relu', name='conv1Filter2')(X_input)
    maxpool_2 = MaxPool1D(pool_size=2, strides=1, padding='same', name='maxpool2')(X2)

    X3 = Conv1D(512, kernel_size=8, padding='same', kernel_initializer='normal', activation='relu', name='conv1Filter3')(X_input)
    maxpool_3 = MaxPool1D(pool_size=2, strides=1, padding='same', name='maxpool3')(X3)

    X4 = Conv1D(512, kernel_size=16, padding='same', kernel_initializer='normal', activation='relu', name='conv1Filter3')(X_input)
    maxpool_4 = MaxPool1D(pool_size=2, strides=1, padding='same', name='maxpool4')(X3)

    concatenated_tensor = Concatenate(axis=1)([maxpool_1, maxpool_2, maxpool_3, maxpool_4])

    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(0.5)(flatten)
    output = Dense(units=3, activation='softmax', name='fully_connected_affine_layer')(dropout)

    model = Model(inputs=inputs, outputs=output, name='intent_classifier')
    adam = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

model = cnn1d_model()
model = lstm_model()
model = bi_lstm_model()
model = cnn1d_lstm_model()
model = creative_cnn1d_model()

history = model.fit(x_train, y_train, batch_size=128, validation_split=0.2, epochs=5)

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

_, accuracy = model.evaluate(x_test, y_test, verbose=2)
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)
print(accuracy)
