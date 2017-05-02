def parse_sent(sent):
    sent = sent.strip()
    sent = sent.replace(',',' ').replace('，',' ').replace('.',' ').replace('。',' ').replace('!',' ').replace('?',' ')
    sent = sent.replace('？',' ').replace('！',' ').replace('...',' ').replace('：',' ').replace(':',' ').replace('~',' ')
    sent = sent.split()
    return sent

def get_train():
    reviews = []
    with open('data/aspect_review.txt') as f:
        while True:
            try:
                review_id = int(f.readline())
                review = f.readline().strip()
                review = parse_sent(review)
                pos = f.readline().split()
                neg = f.readline().split()

                polarity_dict = {"環境":0,"服務":0,"價格":0,"交通":0,"餐廳":0}
                for i in pos:
                    polarity[i] = 1
                for i in neg:
                    polarity[i] = -1
                ob = {
                    'id': review_id,
                    'review': review,
                    'label': polarity_dict
                }
                reviews.append(ob)
            except:
                break
    return reviews

def get_test():
    reviews = []
    f = open('data/test_review.txt')
    while True:
        try:
            review_id = int(f.readline())
            review = f.readline().strip()
            review = parse_sent(review)
            
            ob = {
                'id': review_id,
                'review': review
            }
            reviews.append(ob)
        except:
            break
    f.close()
    return reviews

