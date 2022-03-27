from distutils import filelist
import sys
sys.path.append("../klasifikasi")

import os
import shutil
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import pickle
# import cnn_model
from chi_square import *
from preprocess import *
from helper import *
from flask import Flask, flash, redirect, render_template, request, send_from_directory
# from werkzeug.utils import secure_filename
from keras.models import load_model
# from gensim.models.word2vec import Word2Vec
# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

"""
{'bola': 0, 
'health': 1, 
'lifestyle': 2, 
'news': 3, 
'otomotif': 4, 
'teknologi': 5}
"""
# load classifier
classes = ['bola', 'health', 'lifestyle', 'news', 'otomotif', 'teknologi']
model = load_model("./asset/models/model-uji_15-folds-4.h5")
chi_results = pd.read_csv("./asset/chi/new-chi-khusus-1.csv")
# INPUT_LEN = 400
# word2vec = Word2Vec.load("./asset/models/word2vec.model")

UPLOAD_FOLDER = os.path.join(os.getcwd(), "static", "upload")
DOWNLOAD_FOLDER = os.path.join(os.getcwd(), "static", "download")
USER_DOWNLOAD_FOLDER = os.path.join(os.getcwd(), "static", "user_download")
DATABASE_FOLDER = os.path.join(os.getcwd(), "static", "database")
ALLOWED_EXTENSIONS = {'txt'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['USER_DOWNLOAD_FOLDER'] = USER_DOWNLOAD_FOLDER
app.config['DATABASE_FOLDER'] = DATABASE_FOLDER

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
    clean_text = [preprocess_data(text)]
    clean_text_tokenize = tokenize_split(clean_text)
    selected_text = select_features(clean_text_tokenize, chi_results, 0.4)

    # feature selection
    with open('./asset/models/vectorizer-rasio40.pickle', 'rb') as handle:
        vectorizer = pickle.load(handle)
        VOCAB_LEN = len(vectorizer.vocabulary_)
        transform_text = vectorizer.transform(selected_text).toarray()
        new_selected_text = transform_text.reshape(len(selected_text), 1, VOCAB_LEN)

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

def allowed_file(files):
    for file in files:
        filename = file.filename
        allow =  '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
        if not allow:
            return False
    
    return True

def save_files(files):
    for file in files:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

def get_files_data():
    upload_path = app.config['UPLOAD_FOLDER']
    onlyfiles = [f for f in listdir(upload_path) if isfile(join(upload_path, f)) and f.endswith(".txt")]

    file_data = []
    for file in onlyfiles:
        with open(join(upload_path, file), 'r') as f:
            text = f.read()
            file_data.append({
                "filename":file,
                "text":text
            })

    return file_data

def prepare_download(data_category):
    download_path = app.config['DOWNLOAD_FOLDER']
    upload_path = app.config['UPLOAD_FOLDER']

    # move file
    for key, listdata in data_category.items():
        if len(data_category[key]) > 0:
            os.makedirs(join(download_path, key), exist_ok=True)
            for data in listdata:
                source = join(upload_path, data['filename'])
                destination = join(download_path, key, data['filename'])
                shutil.move(source, destination)
            
    # zip file
    zip_name = "classify"
    zip_file = join(app.config['USER_DOWNLOAD_FOLDER'], zip_name)
    shutil.make_archive(zip_file, 'zip', download_path)
    
    # delete file in download folder
    for folder in listdir(download_path):
        shutil.rmtree(join(download_path, folder))

    return zip_name + ".zip"

def create_database(data_category):
    database = []
    for key, listdata in data_category.items():
        for data in listdata:
            data['category'] = key
            data['bola'] = data['prob'][0]
            data['health'] = data['prob'][1]
            data['lifestyle'] = data['prob'][2]
            data['news'] = data['prob'][3]
            data['otomotif'] = data['prob'][4]
            data['teknologi'] = data['prob'][5]
            data['query'] = make_query(key, data['prob']) + "&text=" + data['text']
            database.append(data)
    
    db_path = app.config['DATABASE_FOLDER']
    df = pd.DataFrame(database)
    save_to_csv(df, join(db_path, "database.csv"))
    # print(df)


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
    category = request.args.get("ctg")
    db_path = join(app.config['DATABASE_FOLDER'], "database.csv")
    database = pd.read_csv(db_path)
    filelist = database[database['category'] == category]
    print(filelist)
    return render_template("category_list.html", filelist=filelist, category=category)

@app.route("/detail")
def detail():
    data = {
        'category': request.args.get("category", ''),
        'teknologi': request.args.get("teknologi", ''),
        'bola': request.args.get("bola", ''),
        'news': request.args.get("news", ''),
        'otomotif': request.args.get("otomotif", ''),
        'health': request.args.get("health", ''),
        'lifestyle': request.args.get("lifestyle", ''),
        'text': request.args.get("text", ''),
        'filename': request.args.get("filename", '')
    }
    return render_template("detail.html", data=data)

@app.route("/multiple")
def multiple():
    data_category = {
            'bola': [], 
            'health': [], 
            'lifestyle': [], 
            'news': [], 
            'otomotif': [], 
            'teknologi': []
        }
    count = {
        'bola': 0, 
        'health': 0, 
        'lifestyle': 0, 
        'news': 0, 
        'otomotif': 0, 
        'teknologi': 0
    }

    download_link = 'nofile'

    return render_template("multiple.html",data=data_category, count=count, download_link=download_link)


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

@app.route("/post/multipleClassify", methods = ['POST', 'GET'])
def multipleClassify():

    data_category = {
            'bola': [], 
            'health': [], 
            'lifestyle': [], 
            'news': [], 
            'otomotif': [], 
            'teknologi': []
        }
    count = {
        'bola': 0, 
        'health': 0, 
        'lifestyle': 0, 
        'news': 0, 
        'otomotif': 0, 
        'teknologi': 0
    }

    download_link = 'nofile'

    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        files = request.files.getlist('file')

        # check if no file uploaded
        if files[0].filename == '':
            return redirect(request.url)

        # check allowed files
        if not allowed_file(files):
            print("there are files not allowed files")
            return redirect(request.url)
        
        # save files
        save_files(files)

        # get file data
        data = get_files_data()

        # classify file
        for file in data:
            category, probabilities = get_prediction_2(file['text'])
            file['prob'] = probabilities
            data_category[category].append(file)
            count[category] = len(data_category[category])

        # prepare folder for downloaded data
        download_link = prepare_download(data_category)     

        # create database
        create_database(data_category)

    return render_template("multiple.html",data=data_category, count=count, download_link=download_link)

@app.route('/downloads/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    download = app.config['USER_DOWNLOAD_FOLDER']
    full_path = join(download, filename)
    print(full_path)
    if os.path.exists(full_path):
        return send_from_directory(path=full_path, directory=download, filename=filename)
    else:
        return redirect("/multiple")

if __name__ == '__main__':
	app.run(debug=True)