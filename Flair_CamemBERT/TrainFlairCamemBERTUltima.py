import re
import os
import json
import random
import argparse
from datetime import datetime
from collections import Counter

import pandas as pd
from tqdm import tqdm
from lxml import etree
from operator import itemgetter

from flair.embeddings import *
from flair.data import Corpus
from flair.data import Sentence
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.datasets import SentenceDataset
from flair.embeddings import TransformerDocumentEmbeddings

import torch.optim as optim

parser = argparse.ArgumentParser(description='Defi 2')
parser.add_argument('--input', type=str, default="./raw_data/", help='The input file')
parser.add_argument('--output', type=str, default="./out/models/", help='The output root directory for the model')
parser.add_argument('--epochs', type=int, default=1, help='Number of Epochs')
args = parser.parse_args()

# Training Configuration
LR = 0.02
MIN_BATCH_SIZE = 12
EPOCHS = args.epochs

# Current date formatted
CURRENT_DATE = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")

output = os.path.join(args.output, "Flair_Classifier".upper() + "_Allocine+NER_" + "en".upper() + "/" + str(args.epochs) + "_" + CURRENT_DATE)
print(output)

contentTrain = etree.parse(args.input + "train.xml").xpath("//comment")
print("CSV file Train loaded!")
contentDev = etree.parse(args.input + "dev.xml").xpath("//comment")
print("CSV file Dev loaded!")
contentTest = etree.parse(args.input + "test.xml").xpath("//comment")
print("CSV file Test loaded!")

films = json.loads(open("./scrapping/films_dict.json","r").read())
LOADED_WITH_METADATA = 0

TASK_NAME = 'defi_2'

EQUAL_CLASSES = False
# EQUAL_CLASSES = True

# [3724, 2973, 2900, 2100, 1873, 1819, 1668, 1165, 972, 806]
MAX_ELEMENTS = 8000

count_notes = {}

def initNotesCounter():
    for l in ['0,5','1,0','1,5','2,0','2,5','3,0','3,5','4,0','4,5','5,0']:
        count_notes[l] = 0

# Tokenize the input text
def tokenizer(text):
    
    # Lower + Remove date
    text = re.sub("\d\d/\d\d/\d\d\d\d", " ", text.lower())

    # Tokenize
    text = text.replace("•"," | ").replace("#"," ").replace("."," . ").replace(","," , ").replace(":"," : ").replace(";"," ; ").replace("/"," / ").replace("'"," ' ").replace('"',' " ').replace("+"," + ")
    text = text.replace("("," ( ").replace(")"," ) ").replace("["," [ ").replace("]"," ] ").replace("{"," { ").replace("}"," } ")
    text = text.replace("!"," ! ").replace("?"," ? ").replace("*"," * ").replace("@"," @ ").replace("|"," ")
    text = text.replace("é","e").replace("è","e").replace("ê","e")

    return text

def loadCorpora(data, mode):

    movies_ids = []
    ids = []
    notes = []
    sentences = []

    if mode != "test":
        data = data[0:150000]

    # For each row
    for i in tqdm(data):

        commentaire_text = i.find('commentaire').text
        commentaire_movies_ids = i.find('movie').text
        commentaire_review_id = i.find('review_id').text

        if mode != "test":
            commentaire_note = i.find('note').text
        else:
            commentaire_note = ""

        if commentaire_text != None and commentaire_note != None:
            sentences.append(commentaire_text)                    
            notes.append(commentaire_note)                    
            ids.append(commentaire_review_id)   
            movies_ids.append(commentaire_movies_ids)   

        else:
            sentences.append("rien à dire")                    
            notes.append(commentaire_note)                    
            ids.append(commentaire_review_id)    
            movies_ids.append(commentaire_movies_ids)   

    return sentences, notes, ids, movies_ids

def getCorpora(data,mode):

    global LOADED_WITH_METADATA

    sentences = []

    initNotesCounter()

    sents, notes, ids, movies_ids = loadCorpora(data,mode)

    # For each row
    for comment, label, id, movie in zip(sents, notes, ids, movies_ids):

        s = Sentence(comment)

        if mode != "test":
            s.add_label(TASK_NAME, label)

        sentences.append(s)                    

    print(len(sentences))

    return sentences

all = getCorpora(contentTrain, "train")
print("Corpora processed for Train!")

allDev = getCorpora(contentDev, "dev")
print("Corpora processed for Dev!")

print("LOADED_WITH_METADATA: ", LOADED_WITH_METADATA)

# Both Corpora
train = all
dev   = allDev

# Split corpora
train, dev = SentenceDataset(train), SentenceDataset(dev)

# Make a corpus with train and test split
corpus = Corpus(train=train, dev=dev, test=dev)
print(corpus.obtain_statistics())

label_dict = corpus.make_label_dictionary(label_type=TASK_NAME)

# document_embeddings = TransformerDocumentEmbeddings("tblard/tf-allocine", fine_tune=True)
document_embeddings = TransformerDocumentEmbeddings("camembert-base", fine_tune=True)

# Load base TARS
model = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type=TASK_NAME)
model.multi_label = False

# Initialize the text classifier trainer with your corpus
trainer = ModelTrainer(model, corpus)

# Train model
trainer.train(
    base_path=output,
    learning_rate=5e-5,
    mini_batch_size=MIN_BATCH_SIZE,
    max_epochs=EPOCHS,
    train_with_dev=True,

    optimizer=optim.AdamW,
    shuffle=True,
    use_swa=True,
    checkpoint=True,
    use_amp=True,
    amp_opt_level="O2",
)
