import re
import random
import argparse
from lxml import etree
from operator import itemgetter

import numpy as np
from tensorflow.python.keras.layers import embeddings
from tqdm import tqdm

import fasttext

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
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, one_hot
from keras.layers import Dense, Flatten, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

parser = argparse.ArgumentParser(description='Classification')
parser.add_argument('--input', type=str, default="./raw_data/", help='The input file')
parser.add_argument('--output', type=str, default="out/models/", help='The output root directory for the model')
parser.add_argument('--mode', type=str, default="tf-idf", help='bow / tf_idf')
parser.add_argument('--epochs', type=int, default=10, help='Number of Epochs')
args = parser.parse_args()

np.random.seed(42)

print("Load " + args.input + "train.xml")
TRAIN = etree.parse(args.input + "train.xml")
contentTrain = TRAIN.xpath("//comment")
print("XML file Train loaded!")
print(len(contentTrain))

print("Load " + args.input + "dev.xml")
DEV  = etree.parse(args.input + "dev.xml")
contentDev = DEV.xpath("//comment")
print("XML file Dev loaded!")
print(len(contentDev))

print("Load " + args.input + "test.xml")
TEST  = etree.parse(args.input + "test.xml")
contentTest = TEST.xpath("//comment")
print("XML file Test loaded!")
print(len(contentTest))

random.seed(0)

MAX_LENGTH = 200

PAD = 0

NORMALIZED = True

EQUAL_CLASSES = False
# EQUAL_CLASSES = False
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
    text = text.replace("é","e").replace("è","e").replace("ê","e").replace("â","a")
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
        # labels.append([commentaire_note])
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

        # shape = np.shape(element)
        # padded_array = np.zeros((1, MAX_LENGTH))
        # padded_array[:shape[0],:shape[1]] = element
        # print(padded_array)
        # print(padded_array.shape)
        # exit(0)
        x_train_res.append(element + [PAD for i in range(MAX_LENGTH - len(element))])

x_train = np.array(x_train_res)
# x_train = np.array(tokenizer.texts_to_sequences(x_train))
print("x_train has None ? ", np.isnan(x_train).any())

x_dev_res = []
for element in tokenizer.texts_to_sequences(x_dev):
    if len(element) >= MAX_LENGTH:
        x_dev_res.append(element[0:MAX_LENGTH])
    else:
        x_dev_res.append(element + [PAD for i in range(MAX_LENGTH - len(element))])
x_dev = np.array(x_dev_res)
# x_dev   = np.array(tokenizer.texts_to_sequences(x_dev))

print("X_dev has None ? ", np.isnan(x_dev).any())

x_test_res = []
for element in tokenizer.texts_to_sequences(x_test):
    if len(element) >= MAX_LENGTH:
        x_test_res.append(element[0:MAX_LENGTH])
    else:
        x_test_res.append(element + [PAD for i in range(MAX_LENGTH - len(element))])
x_test = np.array(x_test_res)
# x_test  = np.array(tokenizer.texts_to_sequences(x_test))
print("x_test has None ? ", np.isnan(x_test).any())


labelEncoder = LabelEncoder()
y_train = labelEncoder.fit_transform(y_train)
y_train = to_categorical(y_train)

y_dev = labelEncoder.transform(y_dev)
y_dev = to_categorical(y_dev)

# y_test = labelEncoder.transform(y_test)
# y_test = to_categorical(y_test)

labels = labelEncoder.classes_
print("labels: ", labels)

# embeddings_index = pickle.load(open("fasttext/cc.fr.300.bin", "rb"))
embeddings_index = fasttext.load_model('fasttext/cc.fr.300.bin')

embedding_dim = 300
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Prepare embedding matrix from pre-trained model
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get_word_vector(word)
    # embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

inputs = Input(shape=(MAX_LENGTH,), dtype=tf.int64)
# inputs = Input(shape=(MAX_LENGTH,), dtype=tf.int64)
emb = Embedding(vocab_size, embedding_dim, input_length=MAX_LENGTH, weights=[embedding_matrix], trainable=True)(inputs)
# model.add(Embedding(len(vocab), EMBEDDING_SIZE, input_length=EMBEDDING_SIZE))

x = Bidirectional(GRU(embedding_dim))(emb)
# x = Bidirectional(LSTM(embedding_dim))(emb)
# x = Bidirectional(LSTM(128))(emb)
x = Dropout(0.25)(x)

x = Flatten()(x)
# print(x.shape)

x = Dense(128)(x)
# x = Dense(128)(x)
x = Dropout(0.25)(x)
x = ReLU()(x)
x = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x, name="BiGRU")
optim = Adam(lr=0.0001, decay=1e-6)
# optim = Adam(lr=0.0005, decay=1e-6)
# optim = Adadelta(lr=1.0,rho=0.95,epsilon=1e-06)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

print(x_train.shape)
print(x_dev.shape)
print(x_test.shape)

history = model.fit(x_train, y_train, validation_data=(x_dev, y_dev), epochs=args.epochs, batch_size=512, shuffle=True, verbose=2)
scores = model.evaluate(x_dev, y_dev, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

loss, accuracy = model.evaluate(x_train, y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(x_dev, y_dev, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# -----------------------------------------------------------

# model = models.load_model('models/best_model_MLP_mean_pooling_camembert.h5')
# print(model.summary())

predictions = model.predict(x_test)
real_preds = list(np.argmax(predictions, axis=1))

predictions = open("LSTM+FastText_results.txt","w")
for c, id in tqdm(zip(real_preds, ids_test)):
    pred = labels[c]
    predictions.write(str(id) + " " + str(pred) + "\n")
predictions.close()


