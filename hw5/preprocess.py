import pandas as pd
import re
import string
import numpy as np
import emoji
from emoji.unicode_codes import UNICODE_EMOJI
from unidecode import unidecode
from nltk.corpus import stopwords, word_tokenize


# This file is causing lots of problems, so I have to read it before preprocessing and fix the issues...
# It has an extra column, unlike other files.
#df = pd.read_csv('Subtask_A\\twitter-2016train-A.txt', sep='\t', header=None)
#print(df.columns)
#print(df[3].value_counts())
#del df[3]
#print(df.columns)
#df.to_csv('twitter-2016test-A_altered_by_reza.txt', header=None, sep='\t', index=None)

df_train = pd.DataFrame()
df_test = pd.DataFrame()

dataset_names = ['twitter-2016train-A', 'twitter-2016dev-A', 'twitter-2016devtest-A', 'twitter-2016test-A_altered_by_reza', 'twitter-2015train-A', 'twitter-2015test-A', 'twitter-2014sarcasm-A', 'twitter-2014test-A', 'twitter-2013train-A', 'twitter-2013dev-A', 'twitter-2013test-A', 'SemEval2017-task4-test.subtask-A.english']

emoticons_str = r"""(?:[:=;][oO\-]?[D\)\]\(\]/\\OpP])"""

regex_str = [emoticons_str, r'<[^>]+>', r'(?:@[\w_]+)', r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
             r'(?:(?:\d+,?)+(?:\.?\d+)?)', r"(?:[a-z][a-z'\-_]+[a-z])", r'(?:[\w_]+)', r'(?:\S)']

contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because", "could've": "could have",
                       "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                       "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
                       "he'll've": "he will have", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have",
                       "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have","i'm": "i am", "i've": "i have",
                       "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is",
                       "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have",
                       "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                       "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",
                       "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is", "that'd": "that would", "that'd've": "that would have","that's": "that is",
                       "there'd": "there would", "there'd've": "there would have","there's": "there is", "here's": "here is",
                       "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have",
                       "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                       "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                       "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                       "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
                       "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                       "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']

for dataset_name in dataset_names:
    with open('Subtask_A\\' + dataset_name + '.txt', 'r', encoding='utf-8') as file:
        output_data = []
        for line in file:
            info = line.strip().split('\t')
            id, label, text = info[0], info[1], ' '.join(info[2:])

            # I wrote these lines so that I'd be able to fix the line causing problems... Very annoying.
            #try:
            #    id, label, text = info[0], info[1], ' '.join(info[2:])
            #except:
            #    print(id)

            # Remove URLs:
            text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', text)

            # Change quotation mark:
            text = re.sub('’', '\'', text)

            # Convert phrases that exist in that huge dictionary:
            text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(' ')])

            # Remove additional white spaces:
            text = re.sub(r'[\s]+', ' ', text)
            text = re.sub(r'[\n]+', ' ', text)

            # Trim:
            text = text.strip('\'"')

            # Only letters:
            text = re.sub(r'[^a-zA-Z]'', ' ', text)

            # Get rid of emojis:
            text = emoji.get_emoji_regexp().sub(r'', text)

            # Remove numbers:
            text = re.sub('\d+', '', text)

            # Emoticons? Nope, not anymore:
            tokens = tokens_re.findall(text)
            tokens = [token.lower() if emoticon_re.search(token) else token.lower() for token in tokens]

            # Remove stopwords and others:
            tokens = [term.lower() for term in tokens if term.lower() not in stop]

            # Remove hashtags:
            tokens = [term for term in tokens if not term.startswith('#')]

            # Remove profiles:
            tokens = [term for term in tokens if not term.startswith('@')]

            d = {'id': id, 'label': label, 'text': ' '.join(tokens)}
            output_data.append(d)

        output_data = pd.DataFrame(output_data)

        if dataset_name == 'SemEval2017-task4-test.subtask-A.english':
            df_test = pd.DataFrame(output_data)
        else:
            df_train = df_train.append(output_data, ignore_index=True)

print(df_train.shape)
print(df_test.shape)

df_train.to_csv('train.csv', index=False)
df_test.to_csv('test.csv', index=False)
