import re
import json
import argparse
from datetime import datetime

from lxml import etree
from tqdm import tqdm

from flair.models import TextClassifier
from flair.data import Corpus
from flair.embeddings import *
from flair.data import Sentence
from flair.datasets import SentenceDataset

parser = argparse.ArgumentParser(description='Defi 2')
parser.add_argument('--input', type=str, default="./raw_data/", help='The input file')
args = parser.parse_args()

films = json.loads(open("./scrapping/films_dict.json","r").read())

TASK_NAME = 'defi_2'

MIN_BATCH_SIZE = 6

# Current date formatted
CURRENT_DATE = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")

contentReal_Test = etree.parse(args.input + "test.xml").xpath("//comment")
print("CSV file Test loaded!")

# Tokenize the input text
def tokenizer(text):
    
    # Lower + Remove date
    text = re.sub("\d\d/\d\d/\d\d\d\d", " ", text.lower())

    # Tokenize
    text = text.replace("•"," | ").replace("#"," ").replace("."," . ").replace(","," , ").replace(":"," : ").replace(";"," ; ").replace("/"," / ").replace("'"," ' ").replace('"',' " ').replace("+"," + ")
    text = text.replace("("," ( ").replace(")"," ) ").replace("["," [ ").replace("]"," ] ").replace("{"," { ").replace("}"," } ")
    text = text.replace("!"," ! ").replace("?"," ? ").replace("*"," * ").replace("@"," @ ").replace("|"," ")
    text = text.replace("é","e").replace("è","e").replace("ê","e")
    
    # # Remove years
    # text = re.sub(years, " ", text)

    return text

def loadCorpora(data, mode):

    ids = []
    notes = []
    sentences = []

    # if mode != "test":
    #     data = data[0:20000]
    # data = data[0:10]

    # For each row
    for i in tqdm(data):

        commentaire_text = i.find('commentaire').text
        commentaire_review_id = i.find('review_id').text

        if mode != "test":
            commentaire_note = i.find('note').text
        else:
            commentaire_note = "0,5"

        if commentaire_text != None and commentaire_note != None:
            sentences.append(commentaire_text)                    
            notes.append(commentaire_note)                    
            ids.append(commentaire_review_id)   

        else:
            sentences.append("rien à dire")                    
            notes.append(commentaire_note)                    
            ids.append(commentaire_review_id)    

    print("len(sentences)")
    print(len(sentences))

    return sentences, notes, ids

def getCorpora(data,mode):

    sentences = []

    sents, notes, ids = loadCorpora(data,mode)
    print("Corpora loaded!")

    # For each row
    for comment in tqdm(sents):

        s = Sentence(tokenizer(comment))

        # if id in films:
        #     # Create a sentence
        #     s = Sentence(
        #         str(id) + " . " + str(films[id]["duree"]) + " . " + str(films[id]["title"]) + " . " + str(films[id]["realisateur"]) + " . " + str(films[id]["categories"]) + " . " + str(films[id]["acteurs"]) + " . " + tokenizer(comment)
        #     )
        # else:
        #     s = Sentence(
        #         str(id) + " . " + tokenizer(comment)
        #     )

        # # Create a sentence
        # s = Sentence(
        #     tokenizer(comment)
        # )

        sentences.append(s)                    

    print("Corpora converted!")
    print(len(sentences))

    return sentences, notes, ids

allTest, notes, ids = getCorpora(contentReal_Test, "test")
print("Corpora processed for Test!")

# test = SentenceDataset(allTest)
# print("Sentence Dataset Done!")

model = TextClassifier.load("/home/nas-wks01/users/uapv1701666/defi2/out/models/distilbert-base-uncased-finetuned-sst-2-english_FLAIR_CLASSIFIER_Allocine+NER_EN/3_2021-12-25-17-04-31/final-model.pt")
# model = TextClassifier.load("/home/nas-wks01/users/uapv1701666/defi2/out/models/illuin/camembert-base-fquad_FLAIR_CLASSIFIER_Allocine+NER_EN/3_2021-12-25-04-53-10/final-model.pt")
# model = TextClassifier.load("/home/nas-wks01/users/uapv1701666/defi2/out/models/flaubert/flaubert_large_cased_FLAIR_CLASSIFIER_Allocine+NER_EN/3_2021-12-25-02-35-07/final-model.pt")
# model = TextClassifier.load("/home/nas-wks01/users/uapv1701666/defi2/out/models/flaubert/flaubert_base_cased_FLAIR_CLASSIFIER_Allocine+NER_EN/3_2021-12-25-00-05-01/final-model.pt")
# model = TextClassifier.load("/home/nas-wks01/users/uapv1701666/defi2/out/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2_FLAIR_CLASSIFIER_Allocine+NER_EN/3_2021-12-24-17-08-48/final-model.pt")

# model = TextClassifier.load("/home/nas-wks01/users/uapv1701666/defi2/out/models/setu4993/LaBSE_FLAIR_CLASSIFIER_Allocine+NER_EN/3_2021-12-24-02-12-24/final-model.pt")
# model = TextClassifier.load("/home/nas-wks01/users/uapv1701666/defi2/out/models/BaptisteDoyen/camembert-base-xnli_FLAIR_CLASSIFIER_Allocine+NER_EN/3_2021-12-24-03-32-21/final-model.pt")

# model = TextClassifier.load("/home/nas-wks01/users/uapv1701666/defi2/out/models/bhadresh-savani/distilbert-base-uncased-emotion_FLAIR_CLASSIFIER_Allocine+NER_EN/3_2021-12-23-22-09-01/final-model.pt")
# model = TextClassifier.load("/home/nas-wks01/users/uapv1701666/defi2/out/models/FLAIR_CLASSIFIER_Allocine+NER_EN/a11/final-model.pt")
# model = TextClassifier.load("/home/nas-wks01/users/uapv1701666/defi2/out/models/FLAIR_CLASSIFIER_Allocine+NER_EN/3_2021-12-19-16-56-55/final-model.pt") # 3 epochs AlloCine
# model = TextClassifier.load("/home/nas-wks01/users/uapv1701666/defi2/out/models/FLAIR_CLASSIFIER_Allocine+NER_EN/3_2021-12-19-14-30-01/final-model.pt") # 3 epochs
# model = TextClassifier.load("/home/nas-wks01/users/uapv1701666/defi2/out/models/FLAIR_CLASSIFIER_Allocine+NER_EN/3_2021-12-19-04-09-58/final-model.pt") # 4 epochs
# model = TextClassifier.load("/home/nas-wks01/users/uapv1701666/defi2/out/models/FLAIR_CLASSIFIER_Allocine+NER_EN/3_2021-12-19-04-09-58/final-model.pt") # 3 epochs
# model = TextClassifier.load("/home/nas-wks01/users/uapv1701666/defi2/out/models/FLAIR_CLASSIFIER_Allocine+NER_EN/2_2021-12-19-02-24-34/final-model.pt") # 2 epochs
# model = TextClassifier.load("/home/nas-wks01/users/uapv1701666/defi2/out/models/FLAIR_CLASSIFIER_Allocine+NER_EN/3_2021-12-18-23-11-29/final-model.pt") # 3 epochs
# model = TextClassifier.load("/home/nas-wks01/users/uapv1701666/defi2/out/models/FLAIR_CLASSIFIER_Allocine+NER_EN/8_2021-12-18-14-38-27/final-model.pt")
# model = TextClassifier.load("/home/nas-wks01/users/uapv1701666/defi2/out/models/FLAIR_CLASSIFIER_Allocine+NER_EN/4_2021-12-18-01-47-57/final-model.pt")
# model = TextClassifier.load("/users/ylabrak/Defi_2/defi_2/out/models/FLAIR_CLASSIFIER_BC7-LitCovid_Joined+NER_EN/10_2021-12-06-01-17-29/final-model.pt")
# model = TextClassifier.load("/users/ylabrak/Defi_2/defi_2/out/models/FLAIR_CLASSIFIER_BC7-LitCovid_Joined+NER_EN/5_2021-12-06-00-10-46/best-model.pt")
# model = TextClassifier.load("/users/ylabrak/Defi_2/defi_2/out/models/FLAIR_CLASSIFIER_BC7-LitCovid_Joined+NER_EN/5_2021-12-05-19-05-17/best-model.pt")

file_out = open("./results/predictions_model_camembert_allocine_" + str(CURRENT_DATE) + ".txt","w")
for sentence, id in tqdm(zip(allTest, ids)):
    model.predict(sentence)
    res = str(id) + " " + str(sentence.labels[0].value) + "\n"
    file_out.write(res)
file_out.close()

# print("Evaluate!")
# result, score = model.evaluate(
#     test,
#     gold_label_type='defi_2',
#     mini_batch_size=6,
#     out_path=f"./results/predictions_model_camembert.txt",
# )
# print(result)
# print(result.detailed_results)
# print(score)


