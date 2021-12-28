import re
import argparse
from lxml import etree
from operator import itemgetter

from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import classification_report

from tqdm import tqdm

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

parser = argparse.ArgumentParser(description='Multilabel Classification')
parser.add_argument('--input', type=str, default="./", help='The input file')
parser.add_argument('--output', type=str, default="out/models/", help='The output root directory for the model')
parser.add_argument('--mode', type=str, default="bow", help='bow / tf_idf')
parser.add_argument('--model', type=str, default="svm", help='hgb / gb / svm')
parser.add_argument('--epochs', type=int, default=1, help='Number of Epochs')
args = parser.parse_args()

fr_stopwords = open("stopwords-fr.txt","r").read().split("\n")

mlb = LabelEncoder()

TRAIN = etree.parse(args.input + "train.xml")
contentTrain = TRAIN.xpath("//comment")

DEV  = etree.parse(args.input + "dev.xml")
contentTest = DEV.xpath("//comment")

TEST  = etree.parse(args.input + "test.xml")
contentReal_Test = TEST.xpath("//comment")

vectorizer = CountVectorizer(
    max_features = 15000,
    max_df=0.99,
    min_df=0.01,
    lowercase=True,
    stop_words = stopwords.words('french'),
    ngram_range=(1,3),
    strip_accents="ascii",
    # analyzer="char_wb",
)

def tokenizer(text):

    text = text.lower()

    # Remove the URL and Lowercase
    text = re.sub(r"http\S+", "", text)

    # Lower + Remove date
    text = re.sub("\d\d/\d\d/\d\d\d\d", " ", text)

    # Tokenize
    text = text.replace("#"," ").replace("."," . ").replace(","," , ").replace(":"," : ").replace(";"," ; ").replace("/"," / ").replace("'"," ' ").replace('"',' " ').replace("+"," + ")
    text = text.replace("("," ( ").replace(")"," ) ").replace("["," [ ").replace("]"," ] ").replace("{"," { ").replace("}"," } ")
    text = text.replace("!"," ! ").replace("?"," ? ").replace("*"," * ").replace("@"," @ ").replace("|"," ")
    text = text.replace("é","e").replace("è","e").replace("ê","e").replace("â","a") # 23% avec remove et 21% sans remove
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    return text

def getBow(documents, mode):

    # Documents
    docs = list(map(itemgetter(0), documents))

    # Labels
    all_tags = list(map(itemgetter(1), documents))
    all_ids = list(map(itemgetter(2), documents))

    bow = None

    # Apply Transformation
    if mode == "train":
        bow = vectorizer.fit_transform(docs).toarray()
        bow_tags = mlb.fit_transform(all_tags)
    elif mode == "test":
        bow = vectorizer.transform(docs).toarray()
        bow_tags = mlb.transform(all_tags)
    elif mode == "real_test":
        bow = vectorizer.transform(docs).toarray()
        bow_tags = None

    return bow, bow_tags, list(mlb.classes_), all_ids

def getCorpora(data, mode):

    sentences = []

    if mode != "real_test":
        # data = data[0:1000]
        data = data[0:100000]

    # For each row
    for i in tqdm(data):

        commentaire_text = i.find('commentaire').text

        commentaire_review_id = i.find('review_id').text

        if mode != "real_test":
            commentaire_note = i.find('note').text
        else:
            commentaire_note = ""

        if commentaire_text != None and commentaire_note != None:

            # Create a sentence & Tokenize
            s = (tokenizer(commentaire_text), [commentaire_note], commentaire_review_id)

        else:
            s = ("", [commentaire_note], commentaire_review_id)

        sentences.append(s)                    

    return getBow(sentences, mode)

# TF-IDF and labels (text, [labels]) for Train
bows, bow_tags, labels, ids_train = getCorpora(contentTrain,"train")
X, Y = bows, bow_tags

# TF-IDF and labels (text, [labels]) for Test
X_test, y_test, labels_test, ids_test = getCorpora(contentTest,"test")

# TF-IDF and labels (text, [labels]) for Test
X_Real_Test, y_Real_Test, labels_Real_Test, ids_real_test = getCorpora(contentReal_Test,"real_test")

# Split into training and testing data
X_train, y_train = X, Y

svm = LinearSVC(
    max_iter = 1000,
    # learning_rate = 0.01,
    # n_jobs = -1,
)

# Fit the data to the Multilabel classifier
multilabel_classifier = svm.fit(X_train, y_train)

print("Score train: ", svm.score(X_train, y_train))
print("Score test: ", svm.score(X_test, y_test))

# Get predictions for test data
y_test_pred = multilabel_classifier.predict(X_test)

y_real_test_pred = multilabel_classifier.predict(X_Real_Test)
output_labels = [labels_Real_Test[i] for i in y_real_test_pred]

# Create the output file for submition
output_file_results = open("results_labrak_zheng_" + args.model + ".txt","w")
for a, p in zip(ids_real_test, output_labels):
    output_file_results.write(str(a) + " " + str(p) + "\n")
output_file_results.close()

f1_score = classification_report(y_test, y_test_pred, target_names=labels)
print(f1_score)

# Get Vocabulary
vocab = vectorizer.get_feature_names_out()
output_file = open("vocab_" + args.model + ".txt","w")
output_file.write("\n".join(vocab))
output_file.close()