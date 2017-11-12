# need comments!
import csv
with open('train.csv', "rt", encoding='utf-8', errors='ignore') as csvfile:
    twtreader = csv.reader(csvfile)

    trainlistoriginal = list(twtreader)
    trainlistoriginal = trainlistoriginal[1:]

    traintar = [None] * len(trainlistoriginal)
    trainlist = [None] * len(trainlistoriginal)

    for i in range(0, len(trainlistoriginal)):
        traintar[i] = trainlistoriginal[i][0]
        trainlist[i] = trainlistoriginal[i][1]

    # a = len("https://t.co/MAqtFwyM0R")
    # print(a)

    str = "http"
    for i in range(0, len(trainlist)):
        strcount = trainlist[i].count(str)
        if strcount > 0:
            for j in range(0, strcount):
                mystr = trainlist[i]
                ind = mystr.index(str, 0)
                trainlist[i] = trainlist[i][0:ind-1] + trainlist[i][ind + 23-1: -1]

    trainumtar = [[0 for i in range(2)] for i in range(len(traintar))]
    for i in range(0, len(traintar)):
        if traintar[i] == "HillaryClinton":
            trainumtar[i][0] = 1
        else:
            trainumtar[i][1] = 1

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
maxlen = 0
trainlistallwords = set()
for i in range(0, len(trainlist)):
    trainlist[i] = tokenizer.tokenize(trainlist[i])
    trainlistallwords.update(trainlist[i])

maxlen = 0
for tweet in trainlist:
    if len(tweet) > maxlen:
        maxlen = len(tweet)
#print(maxlen)

trainlistallwords = list(trainlistallwords)
N = len(trainlistallwords)

dict = {}
wordnum = 0
for word in trainlistallwords:
    dict[word] = wordnum
    wordnum += 1

# import numpy as np
# OneHotmatrix = np.eye(len(dict))

# def getsentonehotm(index):
#     sentonehot = [[0 for i in range(len(trainlistallwords))]for i in range(1)]
#     for i in range(len(trainlist[index])):
#         newmatrix = np.array(OneHotmatrix[dict[trainlist[index][i]]]).T
#         newmatrix = newmatrix.reshape(1, -1)
#         sentonehot = np.concatenate((sentonehot, newmatrix), axis=0)
#     if len(sentonehot) < maxlen:
#         allzeros = [[0 for i in range(len(trainlistallwords))] for i in range(1)]
#         for i in range(len(sentonehot), maxlen):
#             sentonehot = np.concatenate((sentonehot, allzeros), axis=0)
#     np.delete(sentonehot, 1, axis=0)
#     return sentonehot

def getbatchlabel(index):
    batch_label = [0] * maxlen
    for i in range(len(trainlist[index])):
        batch_label[i] = dict[trainlist[index][i]]
    return batch_label

traindata = []
for i in range(len(trainlist)):
    traindata.append(getbatchlabel(i))

def gettrainbatch(batchi, batch_size):
    batch_x = []
    batch_y = []
    for i in range(batch_size):
        batch_x.append(traindata[batchi * batch_size + i])
        batch_y.append(trainumtar[batchi * batch_size + i])
    batch_len = []
    for i in range(batch_size):
        batch_len.append(len(trainlist[batchi * batch_size + i]))
    return batch_x, batch_y, batch_len

#sentenceonehot = getsentonehotm(0)
# for i in range(len(result)):
#     for j in range(len(result[i])):
#         if result[i, j] == 1:
#             print(i, j)
# print(dict)

import tensorflow as tf
vocalbulary_size = len(trainlistallwords)
embedding_size = 20
lstm_size = 20
senten_words_num = maxlen
class_num = 2
batch_size = 128
# batch_num = len(trainlist)

X = tf.placeholder(tf.int32, [batch_size, maxlen])
Y = tf.placeholder(tf.float32, [batch_size, 2])
sentense_length = tf.placeholder(tf.int32, [batch_size])

weights = tf.Variable(tf.truncated_normal([embedding_size, class_num], stddev=0.1, dtype=tf.float32))
bias = tf.Variable(tf.constant(0.1, shape=[class_num], dtype=tf.float32))

embeddings = tf.Variable(tf.random_uniform([vocalbulary_size, embedding_size], -1.0, 1.0))
inputs = tf.nn.embedding_lookup(embeddings, X)

lstm_cells = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_size)
outputs, state = tf.nn.dynamic_rnn(cell=lstm_cells, inputs=inputs, dtype=tf.float32, sequence_length=sentense_length)
one_batch_predict = tf.nn.softmax(tf.matmul(outputs[:, -1, :], weights) + bias)

cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=one_batch_predict, labels=Y))

train_method = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_loss)

accurate_position = tf.equal(tf.argmax(Y, 1), tf.argmax(one_batch_predict, 1))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batch_num = 37
    for train_step in range(10):
        for batchi in range(batch_num):
            batch_x, batch_y, batch_len = gettrainbatch(batchi, batch_size)
            print(train_step, batchi)
            print(sess.run(weights, feed_dict={X: batch_x, Y: batch_y, sentense_length : batch_len}))
            loss = sess.run(cross_entropy_loss, feed_dict={X: batch_x, Y: batch_y, sentense_length: batch_len})
            print(loss)















