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
    #eliminate retweets
    str = "http"
    for i in range(0, len(trainlist)):
        strcount = trainlist[i].count(str)
        if strcount > 0:
            for j in range(0, strcount):
                mystr = trainlist[i]
                ind = mystr.index(str, 0)
                trainlist[i] = trainlist[i][0:ind-1] + trainlist[i][ind + 23-1: -1]
    #classify training target
    trainumtar = [[0 for i in range(2)] for i in range(len(traintar))]
    for i in range(0, len(traintar)):
        if traintar[i] == "HillaryClinton":
            trainumtar[i][0] = 1
        else:
            trainumtar[i][1] = 1
# words tokenize
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
for i in range(0, len(trainlist)):
    trainlist[i] = tokenizer.tokenize(trainlist[i])

# find sentenses' max length
maxlen = 0
for tweet in trainlist:
    if len(tweet) > maxlen:
        maxlen = len(tweet)

trainlistallwords = set()
for i in range(0, len(trainlist)):
    trainlistallwords.update(trainlist[i])

trainlistallwords = list(trainlistallwords)
N = len(trainlistallwords)
#make a dictionary
dict = {}
wordnum = 0
for word in trainlistallwords:
    dict[word] = wordnum
    wordnum += 1

# change the words batch to number batch
def getbatchlabel(index):
    batch_label = [0] * maxlen
    for i in range(len(trainlist[index])):
        batch_label[i] = dict[trainlist[index][i]]
    return batch_label

traindata = []
for i in range(len(trainlist)):
    traindata.append(getbatchlabel(i))

# get test data
with open('test.csv', "rt", encoding='utf-8', errors='ignore') as csvfile:
    twtreader = csv.reader(csvfile)

    testlistoriginal = list(twtreader)
    testlistoriginal = testlistoriginal[1:]

    testlist = [None] * len(testlistoriginal)

    for i in range(0, len(testlistoriginal)):
        testlist[i] = testlistoriginal[i][1]

    str = "http"
    for i in range(0, len(testlist)):
        strcount = testlist[i].count(str)
        if strcount > 0:
            for j in range(0, strcount):
                mystr = testlist[i]
                ind = mystr.index(str, 0)
                testlist[i] = testlist[i][0:ind - 1] + testlist[i][ind + 23 - 1: -1]

    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    for i in range(0, len(testlist)):
        testlist[i] = tokenizer.tokenize(testlist[i])

def gettestdata(testlist):
    testlist_processed = []
    for i in range(len(testlist)):
        new_test_list = []
        for j in range(len(testlist[i])):
            if dict.get(testlist[i][j]) != None:
                new_test_list.append(testlist[i][j])
        testlist_processed.append(new_test_list)

    test_data = []
    for i in range(len(testlist_processed)):
        test_label = [0] * maxlen
        for j in range(min(len(testlist_processed[i]), maxlen)):
            test_label[j] = dict[testlist_processed[i][j]]
        test_data.append(test_label)

    test_len = []
    for i in range(len(testlist)):
        test_len.append(min(len(testlist_processed[i]), maxlen))
    return test_data, test_len

# get train data
def gettrainbatch(batchi, batch_size):
    batch_x = []
    batch_y = []
    batch_len = []
    if (batchi + 1) * batch_size > len(traindata):
        out_length = len(traindata) - batchi * batch_size
        for i in range(out_length):
            batch_x.append(traindata[batchi * batch_size + i])
            batch_y.append(trainumtar[batchi * batch_size + i])
        for i in range(out_length):
            batch_len.append(len(trainlist[batchi * batch_size + i]))
    else:
        for i in range(batch_size):
            batch_x.append(traindata[batchi * batch_size + i])
            batch_y.append(trainumtar[batchi * batch_size + i])
        for i in range(batch_size):
            batch_len.append(len(trainlist[batchi * batch_size + i]))
    return batch_x, batch_y, batch_len

import tensorflow as tf
vocalbulary_size = len(trainlistallwords)
embedding_size = 25
lstm_size = 25
senten_words_num = maxlen
class_num = 2

X = tf.placeholder(tf.int32, [None, maxlen])
Y = tf.placeholder(tf.float32, [None, 2])
sentense_length = tf.placeholder(tf.int32, [None])

import numpy as np
# weights = tf.Variable(tf.truncated_normal([embedding_size, class_num], stddev=1, dtype=tf.float32))
# bias = tf.Variable(tf.constant(0.0, shape=[class_num], dtype=tf.float32))

embeddings = tf.Variable(tf.random_uniform([vocalbulary_size, embedding_size], -1.0, 1.0))
inputs = tf.nn.embedding_lookup(embeddings, X)
def forwardNN(last_rnn_output):
    lay_1_num = 50
    lay_2_num = 100
    weights_1 = tf.Variable(tf.truncated_normal([embedding_size, lay_1_num], stddev=np.sqrt(2.0/lay_1_num), dtype=tf.float32))
    bias_1 = tf.Variable(tf.constant(0.0, shape=[lay_1_num], dtype=tf.float32))
    result_1 = tf.matmul(last_rnn_output, weights_1) + bias_1
    weights_2 = tf.Variable(tf.truncated_normal([lay_1_num, lay_2_num], stddev=np.sqrt(2.0/lay_2_num), dtype=tf.float32))
    bias_2 = tf.Variable(tf.constant(0.0, shape=[lay_2_num], dtype=tf.float32))
    result_2 = tf.matmul(result_1, weights_2) + bias_2
    out_weights = tf.Variable(tf.truncated_normal([lay_2_num, class_num], stddev=np.sqrt(2.0/class_num), dtype=tf.float32))
    out_bias = tf.Variable(tf.constant(0.0, shape=[class_num], dtype=tf.float32))
    out_result = tf.matmul(result_2, out_weights) + out_bias
    l2_loss = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(out_weights) + tf.nn.l2_loss(weights_2)
    return out_result, l2_loss

#lstm_cells = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_size)
#outputs, state = tf.nn.dynamic_rnn(cell=lstm_cells, inputs=inputs, dtype=tf.float32, sequence_length=sentense_length)
#outputs = tf.nn.dropout(outputs, keep_prob=0.6)
#last_output_idx = tf.range(tf.shape(outputs)[0]) * tf.shape(outputs)[1] + sentense_length - 1
#last_rnn_output = tf.gather(tf.reshape(outputs, [-1, lstm_size]), last_output_idx)

#================cnn===============
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, maxlen, embedding_size, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 2 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
last_cnn_output = conv_net(inputs, weights, biases, keep_prob)
one_batch_predict = forwardNN(last_cnn_output)[0]
probability = tf.nn.softmax(one_batch_predict)
l2_loss = forwardNN(last_cnn_output)[1]
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=one_batch_predict, labels=Y)) + 0.001 * l2_loss
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 2000, 0.96, staircase=True)
learning_step = (tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss))


accurate_position = tf.equal(tf.argmax(Y, 1), tf.argmax(one_batch_predict, 1))
accuracy = tf.reduce_mean(tf.cast(accurate_position, tf.float32))

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    sess.run(init)
    import math
    import random
    batch_size = 128
    train_batch_num = math.ceil(len(trainlist) / batch_size)
    for i in range(1, 2001):
        for batchi in range(train_batch_num):
            batch_x, batch_y, batch_len = gettrainbatch(batchi, batch_size)
            sess.run(learning_step, feed_dict={X: batch_x, Y: batch_y, sentense_length: batch_len})
        if i % 100 == 0 or i == 1:
            batch_x, batch_y, batch_len = gettrainbatch(random.randint(0, train_batch_num-1), batch_size)
            acc, loss = sess.run([accuracy, cross_entropy_loss], feed_dict={X: batch_x, Y: batch_y,
                                                                  sentense_length: batch_len})
            print(acc, loss)
    print("train over!")
    saver.save(sess, "Model/model.ckpt")

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "./Model/model.ckpt")
#Validation part
    # val_x = []
    # val_y = []
    # val_len = []
    # for i in range((train_batch_num - 5) * batch_size, len(trainlist)):
    #     val_x.append(trainlist[i])
    #     val_y.append(trainumtar[i])
    # val_x, val_len = gettestdata(val_x)
    # val_acc, val_loss = sess.run([accuracy, cross_entropy_loss], feed_dict={X: val_x, Y: val_y, sentense_length: val_len})
    # print('val_acc:%f, val_loss:%f' % (val_acc, val_loss))

#test part
    testdata, test_len = gettestdata(testlist)
    test_result = sess.run(probability, feed_dict={X: testdata, sentense_length: test_len})

    csvFile = open("test_result.csv", "w", newline='')
    writer = csv.writer(csvFile)

    fileheader = ["id", "realDonaldTrump", "HillaryClinton"]

    writer.writerow(fileheader)

    for i in range(len(test_result)):
        writer.writerow([i, test_result[i][1], test_result[i][0]])

    csvFile.close()