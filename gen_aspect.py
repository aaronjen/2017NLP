import gensim
from gensim.models.keyedvectors import KeyedVectors
import json

w2v = KeyedVectors.load_word2vec_format('./my_data/med250.model.bin', binary=True)

aspect = {}
for line in open('data/aspect_term.txt'):
    s = line.split()
    aspect[s[0]] = set(s)

for key in aspect:
    seed = aspect[key].copy()
    term = aspect[key]
    for i in seed:
        similars = w2v.most_similar(i, topn=10)
        term.update([i[0] for i in similars])
for key, value in aspect.items():
    aspect[key] = list(value)

json.dump(aspect, open('my_data/term.txt', 'w'))