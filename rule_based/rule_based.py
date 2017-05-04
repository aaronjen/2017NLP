import numpy as np
from loader import get_test
import json

with open('data/aspect_term.txt') as f:
    lines = [l.split() for l in f]
    aspects = {l[0]: l for l in lines}

#Database of polarity
positive_list = []
negative_list = []

#positive processing
with open('data/NTUSD_pos.txt', encoding='UTF-8') as pos_file:
    for line in pos_file:
        line = line.strip("\n")
        positive_list.append(line)

#negative processing
with open('data/NTUSD_neg.txt', encoding='UTF-8') as neg_file:
    for line in neg_file:
        line = line.strip("\n")
        negative_list.append(line)

def get_aspect(sentence):
    for key in aspects:
        for term in aspects[key]:
            if term in sentence:
                return key
    return None

predicts = {}
reviews = get_test()

for ob in reviews:
    id = ob['id']
    review = ob['review']

    polarity_dict = {"環境":0,"服務":0,"價格":0,"交通":0,"餐廳":0}
    for sent in review:
        aspect = get_aspect(sent)
        if not aspect or polarity_dict[aspect] != 0:
            continue

        polarity = 0
        for pos in positive_list:
            if (pos in sent):
                polarity = 1
                break
        for neg in negative_list:
            if (neg in sent):
                polarity = -1
                break
        polarity_dict[aspect] = polarity    
    predicts[id] = polarity_dict

json.dump(predicts, open('predicts.json', 'w'))

