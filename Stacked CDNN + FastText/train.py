import os
import re
import random
import argparse
from lxml import etree
from datetime import datetime

import numpy as np
from tensorflow.python.keras.layers import embeddings
from tqdm import tqdm

import fasttext

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from keras.layers.embeddings import Embedding
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

parser = argparse.ArgumentParser(description='Classification')
parser.add_argument('--input', type=str, default="./raw_data/", help='The input file')
parser.add_argument('--output', type=str, default="out/models/", help='The output root directory for the model')
parser.add_argument('--mode', type=str, default="tf-idf", help='bow / tf_idf')
parser.add_argument('--epochs', type=int, default=10, help='Number of Epochs')
args = parser.parse_args()

np.random.seed(42)

CURRENT_DATE = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

TRAIN = etree.parse(args.input + "train.xml")
contentTrain = TRAIN.xpath("//comment")
print("Train elements loaded: ", len(contentTrain))

DEV  = etree.parse(args.input + "dev.xml")
contentDev = DEV.xpath("//comment")
print("Dev elements loaded: ", len(contentDev))

TEST  = etree.parse(args.input + "test.xml")
contentTest = TEST.xpath("//comment")
print("Test elements loaded: ", len(contentTest))

random.seed(0)

PAD = 0
MAX_LENGTH = 300
NORMALIZED = True
EQUAL_CLASSES = False
MAX_ELEMENTS = 5000

count_notes = {}

def initNotesCounter():
    for l in ['0,5','1,0','1,5','2,0','2,5','3,0','3,5','4,0','4,5','5,0']:
        count_notes[l] = 0

def normalize(text):

    # text = text.lower()

    # Remove the URL and Lowercase
    text = re.sub(r"http\S+", "", text)

    # Lower + Remove date
    text = re.sub("\d\d/\d\d/\d\d\d\d", " ", text)

    # # Tokenize
    text = text.replace("#"," ").replace("."," . ").replace(","," , ").replace(":"," : ").replace(";"," ; ").replace("/"," / ").replace("'"," ' ").replace('"',' " ').replace("+"," + ")
    text = text.replace("("," ( ").replace(")"," ) ").replace("["," [ ").replace("]"," ] ").replace("{"," { ").replace("}"," } ")
    text = text.replace("!"," ! ").replace("?"," ? ").replace("*"," * ").replace("@"," @ ").replace("|"," ")
    # text = text.replace("é","e").replace("è","e").replace("ê","e").replace("â","a")
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    return text

def getCorpora(data, mode):

    initNotesCounter()

    sentences = []
    labels = []
    ids = []

    if mode != "test":
        data = data[0:150000]
        # data = data[0:50000]
        # data = data[0:7000]

    # For each row
    for i in tqdm(data):

        if mode != "test" and i.find('note').text != None:
            commentaire_note = i.find('note').text
        else:
            commentaire_note = None

        # ----- EQUALIZE -----
        if mode == "test" :
            pass
        elif EQUAL_CLASSES == True and mode != "test" and count_notes[commentaire_note] >= MAX_ELEMENTS:
            continue
        else:
            count_notes[commentaire_note] += 1

        commentaire_text = i.find('commentaire').text
        commentaire_review_id = i.find('review_id').text

        if i.find('commentaire').text == None:
            commentaire_text = ""
        elif NORMALIZED:
            commentaire_text = normalize(commentaire_text)

        sentences.append(commentaire_text)
        labels.append(commentaire_note)
        ids.append(commentaire_review_id)

    return sentences, labels, ids

x_train, y_train, ids_train = getCorpora(contentTrain,"train")
x_dev, y_dev, ids_dev = getCorpora(contentDev,"dev")
x_test, y_test, ids_test = getCorpora(contentTest,"test")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)

x_train_res = []
for element in tokenizer.texts_to_sequences(x_train):
    if len(element) >= MAX_LENGTH:
        x_train_res.append(element[0:MAX_LENGTH])
    else:
        x_train_res.append(element + [PAD for i in range(MAX_LENGTH - len(element))])

x_train = np.array(x_train_res)

x_dev_res = []
for element in tokenizer.texts_to_sequences(x_dev):
    if len(element) >= MAX_LENGTH:
        x_dev_res.append(element[0:MAX_LENGTH])
    else:
        x_dev_res.append(element + [PAD for i in range(MAX_LENGTH - len(element))])
x_dev = np.array(x_dev_res)

x_test_res = []
for element in tokenizer.texts_to_sequences(x_test):
    if len(element) >= MAX_LENGTH:
        x_test_res.append(element[0:MAX_LENGTH])
    else:
        x_test_res.append(element + [PAD for i in range(MAX_LENGTH - len(element))])
x_test = np.array(x_test_res)

labelEncoder = LabelEncoder()
y_train = labelEncoder.fit_transform(y_train)
y_train = to_categorical(y_train)

y_dev = labelEncoder.transform(y_dev)
y_dev = to_categorical(y_dev)

labels = labelEncoder.classes_
print("labels: ", labels)

embeddings_index = fasttext.load_model('fasttext/cc.fr.300.bin')

embedding_dim = 300
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Create a empty embeddings for each tokens of the vocabulary
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

# For each vocabulary tokens
for word, i in word_index.items():
    embedding_vector = embeddings_index.get_word_vector(word)
    # If exist
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


inputs = Input(shape=(MAX_LENGTH,), dtype=tf.int64)
emb = Embedding(vocab_size, embedding_dim, input_length=MAX_LENGTH, weights=[embedding_matrix], trainable=True)(inputs)

x1 = Conv1D(embedding_dim*2, 1, activation='relu',padding='valid', strides=1)(emb)
x1 = Conv1D(embedding_dim*2, 1, activation='relu',padding='valid', strides=1)(x1)
x1 = MaxPooling1D()(x1)

x2 = Conv1D(embedding_dim*2, 2, activation='relu',padding='valid', strides=1)(emb)
x2 = Conv1D(embedding_dim*2, 1, activation='relu',padding='valid', strides=1)(x2)
x2 = MaxPooling1D()(x2)

x3 = Conv1D(embedding_dim*2, 3, activation='relu',padding='valid', strides=1)(emb)
x3 = Conv1D(embedding_dim*2, 1, activation='relu',padding='valid', strides=1)(x3)
x3 = MaxPooling1D()(x3)

x4 = Conv1D(embedding_dim*2, 4, activation='relu',padding='valid', strides=1)(emb)
x4 = Conv1D(embedding_dim*2, 1, activation='relu',padding='valid', strides=1)(x4)
x4 = MaxPooling1D()(x4)

x5 = Conv1D(embedding_dim*2, 5, activation='relu',padding='valid', strides=1)(emb)
x5 = Conv1D(embedding_dim*2, 1, activation='relu',padding='valid', strides=1)(x5)
x5 = MaxPooling1D()(x5)

x = Concatenate(axis=1)([x1,x2,x3,x4,x5])
x = Flatten()(x)

x = Dense(128)(x)
x = Dropout(0.25)(x)
x = ReLU()(x)
x = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x, name="StackedCRDNN")
optim = Adam(lr=0.0001, decay=1e-6)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

OUTPUT_DIR = 'models/CNN-FastText/' + str(CURRENT_DATE) + '/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
BEST_PATH = OUTPUT_DIR + 'best_model.h5'
modelCallback = ModelCheckpoint(filepath=BEST_PATH)

history = model.fit(x_train, y_train, validation_data=(x_dev, y_dev), epochs=args.epochs, batch_size=256, shuffle=True, verbose=2, callbacks=[modelCallback])

# Load best model
model = models.load_model(BEST_PATH)

scores = model.evaluate(x_dev, y_dev, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

loss, accuracy = model.evaluate(x_train, y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(x_dev, y_dev, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# -----------------------------------------------------------

predictions = model.predict(x_test)
real_preds = list(np.argmax(predictions, axis=1))

predictions = open("CNN+FastText_results.txt","w")
for c, id in tqdm(zip(real_preds, ids_test)):
    pred = labels[c]
    predictions.write(str(id) + " " + str(pred) + "\n")
predictions.close()


