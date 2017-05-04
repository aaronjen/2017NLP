import tensorflow as tf
import numpy as np
from loader import Loader
import os
import pickle

batch_size = 32
n_word = 20000
seq_len = 20
lstm_size = 256
n_layer = 3
learning_rate = 0.001

review = tf.placeholder(tf.int32, [None, seq_len])
review_len = tf.placeholder(tf.int32, [None])

target = tf.placeholder(tf.int32, [None])

embedding = tf.Variable(tf.random_uniform([n_word, 300], -0.1, 0.1))

embedded_review = tf.nn.embedding_lookup(embedding, review)

basic_lstm = tf.contrib.rnn.BasicLSTMCell(256)

cell = tf.contrib.rnn.MultiRNNCell([basic_lstm] * n_layer)

_, state = tf.nn.dynamic_rnn(cell, embedded_review, review_len, dtype=tf.float32)

lstm_out = state[-1].h

w = tf.Variable(tf.random_uniform([256, 3], -0.1, 0.1))
b = tf.Variable(tf.zeros([3]))

logit = tf.matmul(lstm_out, w) + b

cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=target)
cost = tf.reduce_mean(cost)

pred = tf.cast(tf.argmax(logit, 1), tf.int32)
correct = tf.cast(tf.equal(pred, target), tf.float32)
acc = tf.reduce_mean(correct)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

loader_path = 'deep/loader.p'

if not os.path.exists(loader_path):
    print('New Loader...')
    loader = Loader(seq_len, n_word)
    print('Caching Loader...')
    pickle.dump(loader, open(loader_path, 'wb'))
else:
    print('Using Cached Loader...')
    loader = pickle.load(open(loader_path, 'rb'))

model_path = 'deep/model/'
if not os.path.exists(model_path):
    os.mkdir(model_path)

def train():
    n_epoch = 3

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(max_to_keep=100)

        for epo in range(n_epoch): 
            total_cost = 0
            accs = []
            for it, (d_review_len, d_review, d_pol) in enumerate(loader.train(batch_size)):
                feed = {
                    review : d_review,
                    review_len : d_review_len,
                    target: d_pol
                }

                a, c, _ = sess.run([acc, cost, optimizer], feed_dict=feed)
                total_cost += c
                accs.append(a)
                if (it+1) % 100 == 0:
                    print("iter %d: cost %f acc %f" % (it+1, total_cost/(it+1), np.mean(accs)))
            saver.save(sess, model_path + 'epoch%d-%.2f' % (epo+1, np.mean(accs)))

def predict(s):
    inp_len, inp = loader.to_data(s)

    with tf.Session() as sess:
        saver = tf.train.Saver()

        path = tf.train.latest_checkpoint(model_path)

        saver.restore(sess, path)


        p = pred.eval({review: inp, review_len: inp_len})
        p = [i-1 for i in p]
    return p
