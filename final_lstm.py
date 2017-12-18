import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model,Sequential,load_model
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy
def sample(preds, temperature=1.0):
  if temperature == 0:
      temperature = 1
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)
def f5(seq, idfun=None): 
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = idfun(item)
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result
def build_model(bs,sequence_length, chars):
    model = Sequential()
    model.add(LSTM(bs,input_shape=(chars[1],chars[2])))
    model.add(Dense(chars[2]))
    # model.add(Dense(100))
    model.add(Activation('softmax'))

    #optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model
def extract_unique_words(ttt):
    kkk = [j for i in ttt for j in i]
    return sorted(list(set(kkk)))
def create_sequences(ttt, sequence_length, step):
    sequences = []
    next_words = []
    text = [j for i in ttt for j in i]
    #text =' '.join(text)
    for i in range(0, len(text) - sequence_length, step):
        sequences.append(text[i: i + sequence_length])
        next_words.append(text[i + sequence_length])
    return sequences, next_words
def get_words_index_dicts(chars):
    return dict((c, i) for i, c in enumerate(chars)), dict((i, c) for i, c in enumerate(chars))
def vectorize(sequences, sequence_length, chars, char_to_index, next_chars):
    X = np.zeros((len(sequences), sequence_length, len(chars)), dtype=np.bool)
    y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sequences):
        for t, char in enumerate(sentence):
            X[i, t, char_to_index[char]] = 1
        y[i, char_to_index[next_chars[i]]] = 1
    return X, y

dd = pd.read_csv('../lstm_data.csv')
es = dd['lyrics'].tolist()
es = [f5(i.split(' ')) for i in es]
hms = 20
sl = 10
words =  extract_unique_words(es)
wti, iw = get_words_index_dicts(words)
sequences, nw = create_sequences(es, sl, hms)
X, y = vectorize(sequences, sl, words, wti, nw)
batchSize = 32
epch=10

model = build_model(batchSize,sl, X.shape)
history = model.fit(X, y, batch_size = batchSize, epochs = epch)

#britney spears 9 words 
sentence_1="I'm just a girl with a crush on you"
#britney spears 3 words
sentence_2="crush on you"
#eminem 10 words
sentence_3 = "I was born to brew up storms, stir up shit,"
#eminem 3 words
sentence_4 = "stir up shit"
list_sentences = [sentence_1,sentence_2,sentence_3,sentence_4]
columns = ['aritst','genre', 'lyrics']
df_new = pd.DataFrame(columns=columns)
df_new.loc[0]=['britney-spears','9','1']
df_new.loc[1]=['britney-spears','3','2']
df_new.loc[2]=['eminem','10','3']
df_new.loc[3]=['eminem','3','4']

DIVERSITY=0.3
for sl in range(len(list_sentences)):
  generated = ''
  generated += list_sentences[sl]
  sentence= list_sentences[sl].split()
  for i in range(200):
    x = np.zeros((len(sentence), sl, len(words)))
    for t, wrd in enumerate(sentence):
      x[t, 0, wti[wrd]] = 1.
    predictions = model.predict(x, verbose=0)[0]
    next_index = sample(predictions)
    next_char = iw[next_index]
    generated += ' '+next_char
    sentence = sentence+[next_char]
  df_new.ix[sl,'lyrics'] = generated
df_new.to_csv("../lyrics_generated.csv",index=False)

