# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:34:36 2020

@author: ACER
"""


import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from tensorflow.keras.layers import Embedding
from keras.layers import Dense,Conv1D,MaxPooling1D,Flatten
from keras.optimizers import Adam
import numpy as np
import pickle

def preprocess_train_data():
    max_words = 20000
    sentence_length = 50
    labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    x_train = train['comment_text'].fillna("_na_").values
    x_test = test['comment_text'].fillna("_na_").values
    tokenizer = Tokenizer(num_words = max_words)
    tokenizer.fit_on_texts(list(x_train))
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_t = pad_sequences(x_train, maxlen = sentence_length)
    x_te = pad_sequences(x_test, maxlen = sentence_length)
    y = train[labels].values
    print(x_t)
    print(x_te)
    print(y)
    print(tokenizer)
    return x_t,x_te,y,tokenizer

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

def embedding_weights(tokenizer):
    max_words = 20000
    embed_size = 50
    embedding_indices = dict(get_coefs(*o.strip().split()) for o in open("glove.6B.50d.txt",encoding = "utf8"))
    stacked_emb = np.stack(embedding_indices.values())
    mean = stacked_emb.mean()
    std = stacked_emb.std()
    t_words = tokenizer.word_index
    num_words = min(max_words,len(t_words))
    embed_matrix = np.random.normal(mean,std,(num_words,embed_size))
    
    for word,i in t_words.items():
        if i>= max_words: continue
        emb_weights = embedding_indices.get(word)
        if emb_weights is not None:
            embed_matrix[i] = emb_weights
            
    return embed_matrix

def get_model(embed_matrix):
    max_words = 20000
    sentence_length = 50
    embed_size = 50
    model = Sequential()
    model.add(Embedding(max_words,embed_size,weights=[embed_matrix],input_length = sentence_length))
    model.add(Conv1D(64, 3, activation = 'relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(32,3,activation = 'relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(250,activation = 'relu'))
    model.add(Dense(6, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-4),metrics=['accuracy'])
    return model
    
    

def main():
    x_t,x_te,y,tokenizer = preprocess_train_data()
    embed_matrix = embedding_weights(tokenizer)
    model = get_model(embed_matrix)
    model.fit(x_t,y,batch_size = 128, epochs = 10,validation_split = 0.25)
    model.save("sentiment_model.h5")
    pickle.dump(tokenizer, open('tokenizer.pickle','wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":
    main()
    
    