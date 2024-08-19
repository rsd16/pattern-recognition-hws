import numpy as np
import cv2
import os
import pandas as pd
import string
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras_tqdm import TQDMNotebookCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


with open('words.txt') as file:
    contents = file.readlines()

lines = [line.strip() for line in contents]
#print(lines)

max_label_len = 0

characters_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
#print(characters_list, len(characters_list))

num_records = 10000
images = []
labels = []
train_images = []
train_labels = []
train_input_length = []
train_label_length = []
train_original_text = []
valididation_images = []
valid_labels = []
valididation_input_length = []
valididation_label_length = []
valid_original_text = []
inputs_length = []
labels_length = []

# Encode each output word into digits:
def encode_to_labels(txt):
    digits_list = []
    for index, char in enumerate(txt):
        digits_list.append(characters_list.index(char))

    return digits_list

# Convert image to shape (32, 128, 1) and then, normalize:
def process_image(img):
    w, h = img.shape

    # Aspect Ratio Calculation
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape

    img = img.astype('float32')

    # Convert image to (32, 128, 1)
    if w < 32:
        add_zeros = np.full((32 - w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape

    if h < 128:
        add_zeros = np.full((w, 128 - h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape

    if h > 128 or w > 32:
        dim = (128, 32)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)

    img = np.expand_dims(img, axis=2)

    # Normalize
    img = img / 255

    return img

for index, line in enumerate(lines):
    splits = line.split(' ')
    status = splits[1]
    if status == 'ok':
        word_id = splits[0]
        word = ''.join(splits[8:])
        splits_id = word_id.split('-')
        filepath = 'words/{}/{}-{}/{}.png'.format(splits_id[0], splits_id[0], splits_id[1], word_id)

        # Process the image:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        try:
            img = process_image(img)
        except:
            continue

        # Process the label:
        try:
            label = encode_to_labels(word)
        except:
            continue

        if index % 10 == 0:
            valididation_images.append(img)
            valid_labels.append(label)
            valididation_input_length.append(31)
            valididation_label_length.append(len(word))
            valid_original_text.append(word)
        else:
            train_images.append(img)
            train_labels.append(label)
            train_input_length.append(31)
            train_label_length.append(len(word))
            train_original_text.append(word)

        if len(word) > max_label_len:
            max_label_len = len(word)

    if index >= num_records:
        break

train_padded_label = pad_sequences(train_labels, maxlen=max_label_len, padding='post', value=len(characters_list))
#print(train_padded_label.shape)

train_images = np.asarray(train_images)
train_input_length = np.asarray(train_input_length)
train_label_length = np.asarray(train_label_length)

print(train_images.shape)
print(train_images)

valid_padded_label = pad_sequences(valid_labels, maxlen=max_label_len, padding='post', value=len(characters_list))
#print(valid_padded_label.shape)

valididation_images = np.asarray(valididation_images)
valididation_input_length = np.asarray(valididation_input_length)
valididation_label_length = np.asarray(valididation_label_length)

#print(valididation_images.shape)

# Height=32 and Width=128
inputs = Input(shape=(32, 128, 1))

conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
batch_norm_5 = BatchNormalization()(conv_5)

conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

blstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))(blstm_1)

outputs = Dense(len(characters_list) + 1, activation='softmax')(blstm_2)

act_model = Model(inputs, outputs) # To be used at test time

act_model.summary()

true_labels = Input(name='true_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, true_labels, input_length, label_length])

model = Model(inputs=[inputs, true_labels, input_length, label_length], outputs=loss_out) # To be used at training time

batch_size = 8
epochs = 60
e = str(epochs)
optimizer_name = 'adam'

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer_name, metrics=['accuracy'])

history = model.fit(x=[train_images, train_padded_label, train_input_length, train_label_length], y=np.zeros(len(train_images)),
                    batch_size=batch_size, epochs=epochs, validation_data=([valididation_images, valid_padded_label,
                    valididation_input_length, valididation_label_length], [np.zeros(len(valididation_images))]), verbose=1)

# Predict outputs on validation images:
prediction = act_model.predict(train_images[150:170])

# use CTC decoder
decoded = K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1], greedy=True)[0][0]

out = K.get_value(decoded)

# see the results
for i, x in enumerate(out):
    print('original_text =  ', train_original_text[150 + i])
    print('predicted text = ', end = '')
    for p in x:
        if int(p) != -1:
            print(characters_list[int(p)], end = '')

    plt.imshow(train_images[150 + i].reshape(32,128), cmap=plt.cm.gray)
    plt.show()
    print('\n')
