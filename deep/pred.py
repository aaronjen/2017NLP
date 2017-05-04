from model import predict
import pandas as pd
import numpy as np
import json

# with open('data/aspect_term.txt') as f:
#     lines = [l.split() for l in f]
#     aspects = {l[0]: l for l in lines}

aspects = json.load(open('my_data/term.txt'))

def parse_sent(sent):
    sent = sent.strip()
    sent = sent.replace(',',' ').replace('，',' ').replace('.',' ').replace('。',' ').replace('!',' ').replace('?',' ')
    sent = sent.replace('？',' ').replace('！',' ').replace('...',' ').replace('：','').replace(':','').replace('~',' ').replace('\\', '')
    sent = sent.split()
    return sent

def get_aspect(sentence):
    for key in aspects:
        for term in aspects[key]:
            if term in sentence:
                return key
    return None

def parse_review(review):
    ob = {}
    last_aspect = None
    for s in review:
        aspect = get_aspect(s)
        if aspect:
            if aspect in ob:
                ob[aspect] += ' ' + s
            else:
                ob[aspect] = s
        else:
            if last_aspect:
                ob[last_aspect] += ' ' + s
        last_aspect = aspect
    return ob

def get_valid():
    reviews = {}
    review_ans = {}
    f = open('data/aspect_review.txt')
    while True:
        try:
            review_id = int(next(f))
            review = f.readline().strip()
            review = parse_sent(review)

            pos = f.readline().split()
            neg = f.readline().split()
            ans = {"環境":0,"服務":0,"價格":0,"交通":0,"餐廳":0}

            for i in pos:
                ans[i] = 1
            for i in neg:
                ans[i] = -1

            review_ans[review_id] = ans
            reviews[review_id] = parse_review(review)
        except StopIteration:
            break
    f.close()
    return reviews, review_ans

def valid():
    reviews, review_ans = get_valid()
    sents = []
    for id, r in reviews.items():
        for asp in r:
            sents.append(r[asp])

    pols = predict(sents)
    
    count = 0
    corr = []
    for id, r in reviews.items():
        ans = {"環境":0,"服務":0,"價格":0,"交通":0,"餐廳":0}
        for k in ans:
            if k in r:
                ans[k] = pols[count]
                count += 1
        true_ans = review_ans[id]

        for k in true_ans:
            corr.append(true_ans[k] == ans[k])

    print(len(sents))
    print(np.mean(corr))

def get_test():
    reviews = {}
    f = open('data/test_review.txt')
    while True:
        try:
            review_id = int(next(f))
            review = f.readline().strip()
            review = parse_sent(review)

            reviews[review_id] = parse_review(review)
        except StopIteration:
            break
    f.close()
    return reviews

def test():
    reviews = get_test()

    def map_sent(x):
        r_id = x['Review_id']
        aspect = x['Aspect']
        ob = reviews[r_id]

        if aspect not in ob:
            return
        return ob[aspect]
        

    df = pd.read_csv('data/test.csv')
    df['Sent'] = df.apply(map_sent, 1)

    valid_sents = [i for i in df['Sent'] if i != None]
    pols = predict(valid_sents)

    labels = []
    count = 0
    for i in df['Sent']:
        if i == None:
            labels.append(0)
        else:
            labels.append(pols[count])
            count += 1
        
    df['Label'] = labels

    df.to_csv('deep.csv', columns=['Id', 'Label'], index=False)

# valid()
test()