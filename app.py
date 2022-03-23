from os import remove
import sys
sys.path.append("../klasifikasi")

import pandas as pd
import numpy as np
import pickle
import cnn_model
from chi_square import *
from preprocess import *
from helper import *
from flask import Flask, redirect, render_template, request
from keras.models import load_model
from gensim.models.word2vec import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

"""
{'bola': 0, 
'health': 1, 
'lifestyle': 2, 
'news': 3, 
'otomotif': 4, 
'teknologi': 5}
"""

classes = ['bola', 'health', 'lifestyle', 'news', 'otomotif', 'teknologi']
# load classifier
model = load_model("./asset/models/model_2.h5")
word2vec = Word2Vec.load("./asset/models/word2vec.model")
chi_results = pd.read_csv("./asset/chi/new-chi-khusus.csv")
INPUT_LEN = 400

def preprocess_data(text):
    text = case_folding_sentence(text)
    text = remove_punctuation_sentence(text)
    text = remove_number(text)
    text = remove_single_character(text)
    text = stemming_sentence(text)
    text = tokenize_split_sentence(text)
    text = stop_word_removal_sentence(text)
    text = " ".join(text)
    
    return text

def get_prediction_2(text):
    # clean data
    clean_text = preprocess_data(text)

    # feature selection
    with open('./asset/models/vectorizer.pickle', 'rb') as handle:
        vectorizer = pickle.load(handle)
        transform_text = vectorizer.transform([clean_text]).toarray()
        selected_text = select_features_nWords(transform_text, vectorizer.vocabulary_, chi_results, INPUT_LEN)
        new_selected_text = np.array(selected_text)

        # predict
        pred = model.predict(new_selected_text)
        y_predict = np.argmax(pred, axis=1)
        category = classes[y_predict[0]]
    
    return (category, pred[0])

def get_prediction(text):
    # clean data
    clean_text = preprocess_data(text)

    # feature selection
    tokenize_data = clean_text.split()
    selected_text = select_features([tokenize_data], chi_results, 0.25)
    join_selected_features = [" ".join(selected_text[0])]

    # tokenization and sequences
    with open('./asset/models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        x_test_sequences = tokenizer.texts_to_sequences(join_selected_features)
        x_test_sequences = pad_sequences(x_test_sequences, INPUT_LEN)
        
        pred = model.predict(x_test_sequences)
        y_predict = np.argmax(pred, axis=1)
        category = classes[y_predict[0]]
        # print(category)
        # print(pred)
    
    return (category, pred[0])

def make_query(category, probabilities):
    query = "?category=" + category
    for index, label in enumerate(classes):
        query += "&" + label + "=" + str(probabilities[index])

    return query

app = Flask(__name__)

@app.route("/")
def home():
    data = {
        'category': request.args.get("category", ''),
        'teknologi': request.args.get("teknologi", ''),
        'bola': request.args.get("bola", ''),
        'news': request.args.get("news", ''),
        'otomotif': request.args.get("otomotif", ''),
        'health': request.args.get("health", ''),
        'lifestyle': request.args.get("lifestyle", ''),
        'text': request.args.get("text", '')
    }
    return render_template('single.html',data=data)

@app.route("/categorylist")
def list():
    return render_template("category_list.html")

@app.route("/detail")
def detail():
    return render_template("detail.html")

@app.route("/multiple")
def multiple():
    return render_template("multiple.html")

@app.route("/post/singleClassify", methods = ['POST'])
def singleClassify():
    
    # get data
    text = request.form.get("text")
    query = ""
    if len(text) > 0:
        # get prediction
        category, probabilities = get_prediction_2(text)

        # make query
        query = make_query(category, probabilities)
        query += "&text=" + text

    return redirect("/" + query)

if __name__ == '__main__':
	app.run(debug=True)