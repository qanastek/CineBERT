import re
import random
import argparse
from lxml import etree
from operator import itemgetter

import numpy as np
from tqdm import tqdm
from transformations import unit_vector

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import tensorflow as tf
from tensorflow.keras import *
from keras.models import Sequential
from tensorflow.keras.layers import *
from keras.preprocessing import sequence
from tensorflow.keras.optimizers import *
from keras.layers.embeddings import Embedding
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Flatten, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

parser = argparse.ArgumentParser(description='Classification')
parser.add_argument('--input', type=str, default="./", help='The input file')
parser.add_argument('--output', type=str, default="out/models/", help='The output root directory for the model')
parser.add_argument('--mode', type=str, default="tf-idf", help='bow / tf_idf')
parser.add_argument('--epochs', type=int, default=1, help='Number of Epochs')
args = parser.parse_args()

fr_stopwords = open("stopwords-fr.txt","r").read().split("\n")

NORMALIZE = True

le = LabelEncoder()
mlb = LabelEncoder()

np.random.seed(42)

# colonnes : numéro de ligne ; terme ; # positif ; # neutre ; # négatif 
polarity_fr = open("polarity_fr.txt", "r").read().split("\n")

# vocab_15k = open("15K_vocab.txt", "r").read().split("\n")

mot_sentiment = {}
vocab_bis = []

for line in polarity_fr:
    line = line.replace("\"", "")
    line = line.lower()
    infos = line.split(";")
    if len(infos[1].split(" "))<=1:
        maxValue = max(infos[2:])
        sentiment = ""
        if (maxValue == infos[2]):
            mot_sentiment[infos[1]] = 0 # positif
        elif (maxValue == infos[3]):
            mot_sentiment[infos[1]] = 1 # neutre
        elif (maxValue == infos[4]):
            mot_sentiment[infos[1]] = 2 # negatif
        
        words = infos[1].split(" ")
        vocab_bis.extend(words)

print("len(mot_sentiment)")
print(len(mot_sentiment))

vocab_bis = list(set(vocab_bis))

vectorizer = TfidfVectorizer(
        max_features=10000,
        use_idf=True,
        stop_words=stopwords.words('french'),
        # max_df=0.98,
        # min_df=0.02,
        strip_accents="ascii",
        # analyzer="char_wb",
        # binary=True,
        # ngram_range=(1,3),
    )

UNK = "<unk>"
PAD = "<pad>"

def tokenizer(text):

    text = text.lower()
    
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    return text

def getSentiment3(mot_sentiment,sen):
    
    vec = [0.0, 0.0, 0.0,]

    phrase = sen.split(" ")
    for s in phrase:
        if s in mot_sentiment:
            vec[mot_sentiment[s]] += 1

    return vec


def getCorpora(data, mode):

    sentences = []

    if mode != "test":
        # data = data[0:100]
        data = data[0:7000]
        
    # For each row
    for i in tqdm(data):

        commentaire_text = i.find('commentaire').text
        commentaire_review_id = i.find('review_id').text
        if mode != "test":
            commentaire_note = i.find('note').text
        else:
            commentaire_note = ""

        if commentaire_text != None and commentaire_note != None:
            s = (tokenizer(commentaire_text), [commentaire_note], commentaire_review_id)
        else:
            s = ("", [commentaire_note], commentaire_review_id)

        sentences.append(s)  
    
    random.shuffle(sentences)

    print("len(sentences)")
    print(len(sentences))

    com_text, com_note, com_id = zip(*sentences)
    com_text, com_note, com_id = list(com_text), list(com_note), list(com_id)

    old_text = com_text

    print("Before transform")

    # Transform note to one hot vector
    if mode == "train":
        com_text = vectorizer.fit_transform(com_text).toarray()
    else:
        com_text = vectorizer.transform(com_text).toarray()

    print("Before sentiment")

    res = []
    for text in tqdm(old_text):
        emotion_vec = getSentiment3(mot_sentiment,text)
        res.append(emotion_vec)

    print("after sentiment")

    if NORMALIZE:
        res = np.nan_to_num(unit_vector(res, axis=1), nan=0.0)
    
    merged = []
    for a,b in zip(res,com_text):

        merged.append(np.concatenate((a, b), axis=None).astype(np.float32))
        
    print("after merged")
    print(merged[0])
    
    return merged, com_note, com_id

print("Load " + args.input + "train.xml")
TRAIN = etree.parse(args.input + "train.xml")
contentTrain = TRAIN.xpath("//comment")
print("XML file Train loaded!")
print(len(contentTrain))

print("Start building train...")
x_train, y_train, ids_train = getCorpora(contentTrain,"train")
print("Train loaded!")
print("lenght: " + str(len(x_train[0])))

svm = LinearSVC(
    max_iter = 1000,
)
multilabel_classifier = svm.fit(x_train, y_train)

print("Load " + args.input + "test.xml")
TEST  = etree.parse(args.input + "test.xml")
contentTest = TEST.xpath("//comment")

x_test, y_test, ids_test = getCorpora(contentTest,"test")
print("Test loaded!")

labels = ["0,5","1,0","1,5","2,0","2,5","3,0","3,5","4,0","4,5","5,0"]
nbr_labels = len(labels)
print("labels:", labels)

import pickle
filename = 'finalized_model.sav'

pickle.dump(multilabel_classifier, open(filename, 'wb'))

y_real_test_pred = multilabel_classifier.predict(x_test)
output_labels = [i for i in y_real_test_pred]

# Create the output file for submition
output_file_results = open("results_labrak_zheng_1.txt","w")
for a, p in zip(ids_test, output_labels):
    output_file_results.write(str(a) + " " + str(p) + "\n")
output_file_results.close()

