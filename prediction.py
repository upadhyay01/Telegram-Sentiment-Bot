# -*- coding: utf-8 -*-

import pickle
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model

tokenizer = pickle.load(open("tokenizer.pickle","rb"))
model = load_model("sentiment_model.h5")

def input_text(x):
    return [x]

def preprocess_text(x,tokenizer = tokenizer,max_words = 20000,sentence_length = 50):
    tokenized_text = tokenizer.texts_to_sequences(x)
    x_t = pad_sequences(tokenized_text, maxlen = sentence_length)
    return x_t

def predict(x):
    x_t = preprocess_text(input_text(x))
    labels = ["Toxic", "Severely Toxic", "Obscene", "Threat", "Insult", "Identity Hate"] 
    output = str(dict(zip(labels,model.predict([x_t,]).flatten())))
    return output

#o = predict("motherfucker")
#print(o)
    
