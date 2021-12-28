import os
import re
import pickle
import argparse
import itertools
from lxml import etree
from itertools import repeat
from datetime import datetime
from operator import itemgetter
from collections import Counter
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count

from sklearn.ensemble import *
from sklearn.utils import class_weight
from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay

import torch

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow as tf
from tensorflow.keras import *
from keras.models import Sequential
from tensorflow.keras.layers import *
from keras.preprocessing import sequence
from tensorflow.keras.optimizers import *
from keras.layers.embeddings import Embedding
from tensorflow.keras.utils import to_categorical
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, Flatten, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint

from transformers import CamembertModel, CamembertTokenizer

import numpy as np
from tqdm import tqdm

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

parser = argparse.ArgumentParser(description='Multilabel Classification')
parser.add_argument('--input', type=str, default="./raw_data/", help='The input file')
parser.add_argument('--output', type=str, default="out/models/", help='The output root directory for the model')
parser.add_argument('--epochs', type=int, default=1, help='Number of Epochs')
args = parser.parse_args()

fr_stopwords = open("stopwords-fr.txt","r").read().split("\n")

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
camembert = CamembertModel.from_pretrained("camembert-base")

# mlb = LabelEncoder()
# mlb = MultiLabelBinarizer()

CamemBERT_EMBED_SIZE = 768
CamemBERT_MAX_TOKENS = 512

EQUAL_CLASSES = True
MAX_ELEMENTS = 700

count_notes = {}

def initNotesCounter():
    for l in ['0,5','1,0','1,5','2,0','2,5','3,0','3,5','4,0','4,5','5,0']:
        count_notes[l] = 0

def normalize(text):

    text = text.lower()

    # Remove the URL and Lowercase
    text = re.sub(r"http\S+", "", text)

    # Lower + Remove date
    text = re.sub("\d\d/\d\d/\d\d\d\d", " ", text)

    # # Tokenize
    text = text.replace("#"," ").replace("."," . ").replace(","," , ").replace(":"," : ").replace(";"," ; ").replace("/"," / ").replace("'"," ' ").replace('"',' " ').replace("+"," + ")
    text = text.replace("("," ( ").replace(")"," ) ").replace("["," [ ").replace("]"," ] ").replace("{"," { ").replace("}"," } ")
    text = text.replace("!"," ! ").replace("?"," ? ").replace("*"," * ").replace("@"," @ ").replace("|"," ")
    text = text.replace("é","e").replace("è","e").replace("ê","e").replace("â","a") # 23% avec remove et 21% sans remove
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    return text

def getEmbedding768(sentence):
    enc = torch.tensor(
        tokenizer.encode(sentence, max_length=CamemBERT_MAX_TOKENS, truncation=True)
    ).unsqueeze(0)
    return camembert(enc)[0][0].mean(dim=0).detach().numpy()

def getBow(documents, mode):

    # Documents
    # docs = [text for text, labels in documents]
    docs = list(map(itemgetter(0), documents))

    # Labels
    # all_tags = [labels for text, labels in documents]
    all_tags = list(map(itemgetter(1), documents))

    all_ids = list(map(itemgetter(2), documents))

    bow = None

    print("Before getEmbedding")

    bow = []
    for doc in tqdm(docs):
        bow.append(getEmbedding768(doc))
    bow = np.array(bow)
    # print(type(bow[0]))
    # exit(0)

    # bow = vectorizer.fit_transform(docs).toarray()

    # Apply Transformation
    if mode == "train":
        bow_tags = all_tags
        # bow_tags = mlb.fit_transform(all_tags)
    elif mode == "dev":
        bow_tags = all_tags
        # bow_tags = mlb.transform(all_tags)
    elif mode == "test":
        bow_tags = None

    if mode != "test":
        print("Vector size:", str(len(bow[0])))
        bow_tags = list(itertools.chain(*bow_tags))
        classes_in = list(set(bow_tags))
        print(bow_tags[0:2])
        # classes_in = list(mlb.classes_)
        print(classes_in)

    return bow, bow_tags, classes_in, all_ids, all_tags

def getCorpora(data, mode, nbr_elements):

    initNotesCounter()

    sentences = []

    # pool = Pool(cpu_count())
    # inputs_pool = list(zip(data, repeat(mode)))
    # print(inputs_pool[0:3])
    # sentences = pool.starmap(process, inputs_pool)
    # pool.close()
    # pool.join()
    
    # sentences = Parallel(n_jobs=cpu_count())(delayed(process)(d,m) for d,m in zip(data, repeat(mode)))

    if mode != "test":
        data = data[0:nbr_elements]
        
    # For each row
    for i in tqdm(data):

        # ----- RATING -----
        if mode != "test":
            commentaire_note = i.find('note').text
        else:
            commentaire_note = ""
       
        # ----- EQUALIZE -----
        if mode == "test" :
            pass
        elif EQUAL_CLASSES == True and mode != "test" and count_notes[commentaire_note] >= MAX_ELEMENTS:
            continue
        else:
            count_notes[commentaire_note] += 1

        # ----- PROCESS -----
        commentaire_text = i.find('commentaire').text
        commentaire_review_id = i.find('review_id').text

        if commentaire_text != None and commentaire_note != None:
            # Create a sentence & Tokenize
            s = (normalize(commentaire_text), [commentaire_note], commentaire_review_id)
        else:
            s = ("", [commentaire_note], commentaire_review_id)

        sentences.append(s)
            
    print("len(sentences)")
    print(len(sentences))
    return getBow(sentences, mode)

# nbr_elements = 2
nbr_elements = 25000

train_path_pickle = "corporas/camembert_mean_pooling_corpora_" + "train" + "_" + str(nbr_elements) + ".pickle"
dev_path_pickle  = "corporas/camembert_mean_pooling_corpora_" + "dev" + "_" + str(nbr_elements) + ".pickle"
test_path_pickle  = "corporas/camembert_mean_pooling_corpora_" + "test" + "_" + str(nbr_elements) + ".pickle"

if os.path.isfile(train_path_pickle) == False and os.path.isfile(dev_path_pickle) == False and os.path.isfile(test_path_pickle) == False:

    print("Load " + args.input + "train.xml")
    TRAIN = etree.parse(args.input + "train.xml")
    contentTrain = TRAIN.xpath("//comment")
    print("CSV file Train loaded!")
    print(len(contentTrain))

    print("Load " + args.input + "dev.xml")
    DEV  = etree.parse(args.input + "dev.xml")
    contentDev = DEV.xpath("//comment")
    print("CSV file Dev loaded!")
    print(len(contentDev))

    print("Load " + args.input + "test.xml")
    TEST  = etree.parse(args.input + "test.xml")
    contentTest = TEST.xpath("//comment")
    print("CSV file Test loaded!")
    print(len(contentTest))

    # print("Load " + args.input + "test.xml")
    # TEST  = etree.parse(args.input + "test.xml")
    # contentReal_Test = TEST.xpath("//comment")
    # print("CSV file Test loaded!")
    # print(len(contentReal_Test))

    print("°"*50)
    print("Enter in PROCESS embeddings")
    print("°"*50)

    # TF-IDF and labels (text, [labels]) for Train
    print("Get train")
    X, Y, labels, ids_train, all_tags_train = getCorpora(contentTrain,"train",nbr_elements)
    print("End train")

    with open(train_path_pickle, 'wb') as handle:
        pickle.dump((X, Y, labels, ids_train, all_tags_train), handle, protocol=pickle.HIGHEST_PROTOCOL)

    # TF-IDF and labels (text, [labels]) for Test
    X_dev, y_dev, labels_dev, ids_dev, all_tags_dev = getCorpora(contentDev,"dev",nbr_elements)
    print("After dev")

    with open(dev_path_pickle, 'wb') as handle:
        pickle.dump((X_dev, y_dev, labels_dev, ids_dev, all_tags_dev), handle, protocol=pickle.HIGHEST_PROTOCOL)

    # TF-IDF and labels (text, [labels]) for Test
    X_test, y_test, labels_test, ids_test, all_tags_test = getCorpora(contentTest,"test",nbr_elements)
    print("After test")

    with open(test_path_pickle, 'wb') as handle:
        pickle.dump((X_test, y_test, labels_test, ids_test, all_tags_test), handle, protocol=pickle.HIGHEST_PROTOCOL)

else:

    print("°"*50)
    print("Enter in LOAD embeddings")
    print("°"*50)

    with open(train_path_pickle, 'rb') as handle:
        (X, Y, labels, ids_train, all_tags_train) = pickle.load(handle)

    with open(dev_path_pickle, 'rb') as handle:
        (X_dev, y_dev, labels_dev, ids_dev, all_tags_dev) = pickle.load(handle)

    with open(test_path_pickle, 'rb') as handle:
        (X_test, y_test, labels_test, ids_test, all_tags_test) = pickle.load(handle)

# # TF-IDF and labels (text, [labels]) for Test
# X_Real_Test, y_Real_Test, labels_Real_Test, ids_real_test = getCorpora(contentReal_Test,"real_test")
# print("After real test")

# Split into training and testing data
X_train, y_train = X, Y
print("split train finished")

raw_labels = ['0,5','1,0','1,5','2,0','2,5','3,0','3,5','4,0','4,5','5,0']
raw_labels.sort()
nb_classes = len(raw_labels)
print("nb_classes: ", nb_classes)
print("classes: ", raw_labels)

# Id2Labels & Labels2Ids
y_train = [raw_labels.index(a) for a in y_train]
# class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
y_train = to_categorical(y_train, nb_classes)

y_dev = [raw_labels.index(a) for a in y_dev]
y_dev = to_categorical(y_dev, nb_classes)

# nbr_classes = len(y_train[0])
nbr_classes = len(raw_labels)
input_shape = len(X_train[0])

print("Bow embedding dimension:", input_shape)
print("Number of output classes:", nbr_classes)

model = Sequential()
model.add(Dense(2048, input_shape=(CamemBERT_EMBED_SIZE,), activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(nbr_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

print("X_train / y_train:")
print(len(X_train))
print(len(y_train))

CURRENT_DATE = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
os.makedirs('models_cnn_camembert/' + str(CURRENT_DATE), exist_ok=True)
modelCallback = ModelCheckpoint(filepath='models/best_model_MLP_mean_pooling_camembert.h5')

# unique, counts = np.unique([labels.index(a) for a in list(itertools.chain(*all_tags_train))], axis=0, return_counts=True)
# myDict = dict(zip(unique, counts))

# classWeight = compute_class_weight('balanced', np.unique(y_train, axis=0), y_train)
# print(classWeight)
# classWeight = dict(enumerate(classWeight))
# print(classWeight)
# class_weight=classWeight

model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=150, batch_size=2048, verbose=2, callbacks=[modelCallback])
# model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=3, batch_size=1024, verbose=2, callbacks=[modelCallback])
scores = model.evaluate(X_dev, y_dev, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# -----------------------------------------------------------

# model = models.load_model('models/best_model_MLP_mean_pooling_camembert.h5')
# print(model.summary())

predictions = model.predict(X_test)
real_preds = list(np.argmax(predictions, axis=1))

predictions = open("MLP_mean_pooling_camembert_results.txt","w")
for c, id in tqdm(zip(real_preds, ids_test)):
    pred = labels[c]
    predictions.write(str(id) + " " + str(pred) + "\n")
predictions.close()










