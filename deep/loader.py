import jieba
import pickle
import os
from collections import Counter
import numpy as np

class Loader():
    def __init__(self, seq_len=20, n_word = 20000):
        self.seq_len = seq_len
        self.n_word = n_word

        self.data = self._build()

    def _build(self):
        seq_len = self.seq_len
        n_word = self.n_word

        stopword = open('my_data/stopwords.txt').read().split()
        stopword = set(stopword)
        self.stopword = stopword

        reviews = []

        # bad: 0
        # netural: 1
        # good: 2
        polaritys = []
        for i in open('data/polarity_review.txt'):
            seg = i.strip().split('\t')
            polarity = int(seg[0]) + 1
            review = list(jieba.cut(seg[1]))
            review = [i for i in review if not i in stopword]
            reviews.append(review)
            polaritys.append(polarity)
        polaritys = np.array(polaritys)
            
        words = [j for i in reviews for j in i]
        counter = Counter(words)
        
        word_dict = {i[0]: ind+2 for ind, i in enumerate(counter.most_common(n_word))}
        word_dict['<pad>'] = 0
        word_dict['<unk>'] = 1

        self.word_dict = word_dict
        
        n_data = len(reviews)

        review_len = np.zeros([n_data], dtype=int)
        ind_reviews = np.zeros([n_data, seq_len], dtype=int)
        for i, review in enumerate(reviews):
            review = review[:seq_len]
            ind_reviews[i, :len(review)] = [word_dict[r] if r in word_dict else 1 for r in review]
            review_len[i] = len(review)

        return n_data, review_len, ind_reviews, polaritys

    def train(self, batch_size=32, valid=0):
        n_data, review_len, ind_reviews, polaritys = self.data
        n_data = int(n_data*(1-valid))

        review_len = review_len[:n_data]
        ind_reviews = ind_reviews[:n_data]
        polaritys = polaritys[:n_data]

        n_batch = int((n_data-1)/batch_size + 1)

        order = np.arange(n_batch)
        np.random.shuffle(order)

        for i in order:
            batch_len = review_len[i*batch_size:(i+1)*batch_size]
            batch_review = ind_reviews[i*batch_size:(i+1)*batch_size]
            batch_pol = polaritys[i*batch_size:(i+1)*batch_size]

            yield batch_len, batch_review, batch_pol

    def valid(self, batch_size):
        return

    def to_data(self, s):
        if not hasattr(self, 'word_dict'):
            print('Model not trained')

        seq_len = self.seq_len
        word_dict = self.word_dict
        stopword = self.stopword

        inp = np.zeros([len(s), seq_len])
        inp_len = []
        for ind, line in enumerate(s):
            line = [i for i in list(jieba.cut(line)) if not i in stopword]
            line = line[:seq_len]
            line = [word_dict[i] if i in word_dict else 1 for i in line]

            line_len = len(line)
            inp_len.append(line_len)
            inp[ind, :line_len] = line

        return inp_len, inp
