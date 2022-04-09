import string
import re
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical

def case_folding_sentence(data):
  return data.lower()

def remove_punctuation_sentence(data):
  punc = string.punctuation #!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
  data = data.replace("\n", " ")
  data = data.replace("\xa0", " ")
  new_text = ''
  for letter in data:
    if letter not in punc:
      new_text += letter
  # new_text = new_text.replace("\n", " ")
  # new_text = new_text.replace("\xa0", " ")
  return new_text

def remove_punctuation_sentence_space(data):
  punc = string.punctuation #!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
  data = data.replace("\n", " ")
  data = data.replace("\xa0", " ")
  new_text = ''
  for letter in data:
    if letter not in punc:
        new_text += letter
    else:
        new_text += " "
  # new_text = new_text.replace("\n", " ")
  # new_text = new_text.replace("\xa0", " ")
  return new_text

def tokenize_sentence(data):
    sentence = []
    word = ""
    for character in data:
      if character == " ":
        if word != "":
          sentence.append(word)
          word = ""
        continue
      word += character
    
    return sentence

def tokenize_split_sentence(data):
  return data.split()

def stop_word_removal_sentence(data):
    stop_words = set(stopwords.words('indonesian'))
    sentence = []
    
    for word in data:
      if word not in stop_words:
        sentence.append(word)
    
    return sentence

def stemming_sentence(data):
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  return stemmer.stem(data)

def categorical(y):
  le = preprocessing.LabelEncoder()
  y_categorical = le.fit_transform(y)
  y_categorical = to_categorical(y_categorical)

  return y_categorical

def tokenize_split(data):
  return [ text.split() for text in data ]

def remove_number(text):
  return re.sub(r'[0-9]+', ' ', text)

def remove_single_character(text):
  return ' '.join( [w for w in text.split() if len(w)>1] )