from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import random

import numpy as np


## read text data from file

import csv
import string

labelList = []
tweetList = []
tweetList_test = []

with open('train.csv', encoding='utf-8') as trainfile:
    reader = csv.reader(trainfile)
    for row in reader:
        if row[0] == 'HillaryClinton':
            labelList.append(1)
        elif row[0] == 'realDonaldTrump':
            labelList.append(0)
            
        if row[1] != 'tweet':
            tweetList.append(row[1])
            
with open('test.csv', encoding='utf-8') as testfile:
    reader = csv.reader(testfile)
    for row in reader: 
        if row[1] != 'tweet':
            tweetList_test.append(row[1])

N = len(tweetList)
#print(len(tweetList))
#print(len(labelList))

N_test = len(tweetList_test)
#print(len(tweetList_test))

## corpus
import string

def refineWordsList(tweetList):
    ret = []
    for tweet in tweetList:
        step1 = tweet.split('http',1)[0]
        step2 = step1.lower()
        step3 = step2.translate(str.maketrans('.',' '))
        step4 = step3.translate(str.maketrans('','',string.punctuation))
        ret.append(step4)
    return ret

tweetList = refineWordsList(tweetList)
tweetList_test = refineWordsList(tweetList_test)

corpus = set()

for tweet in tweetList:
    words = tweet.split();
    corpus.update(words)

corpus = list(corpus)
M = len(corpus)


## Label Encoding

hashEncoder = {}
i = 0
for word in corpus:
    hashEncoder[word] = i
    i += 1

# train data
tweetList_encoded = []
for tweet in tweetList:
    tweet_encoded = []
    for word in tweet.split():
        word_encoded = hashEncoder[word]
        tweet_encoded.append(word_encoded)
    tweetList_encoded.append(tweet_encoded)
    
# test data
tweetList_test_encoded = []
for tweet in tweetList_test:
    tweet_encoded = []
    for word in tweet.split():
        if hashEncoder.get(word) != None :
            word_encoded = hashEncoder[word]
            tweet_encoded.append(word_encoded)
    tweetList_test_encoded.append(tweet_encoded)

def decodeTweet(tweet_encoded):
    decodeList = []
    for word_encoded in tweetList_encoded[5]:
        decodeList.append(list(hashEncoder.keys())[list(hashEncoder.values()).index(word_encoded)])
    return decodeList
    
#print(len(tweetList_encoded))
#print(len(tweetList_test_encoded))
#print()
#print(tweetList[5])
#print(tweetList_encoded[5])
#print(decodeTweet(tweetList_encoded[5]))
#print(len(tweetList_encoded[5]))
#print()
#print(tweetList_test[5])
#print(tweetList_test_encoded[5])

#print(tweetList_encoded)
#print('Shape of x: {}'.format(np.array(tweetList_encoded).shape))



#===============lstm train==============
""" Dynamic Recurrent Neural Network.
TensorFlow implementation of a Recurrent Neural Network (LSTM) that performs
dynamic computation over sequences with variable length. This example is using
a toy dataset to classify linear sequences. The generated sequences have
variable length.
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""




# ====================
#  TOY DATA GENERATOR
# ====================
class ToySequenceData(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, n_samples=len(tweetList_encoded), max_seq_len=31, min_seq_len=1,
                 max_value=10000):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            #tweetList_encoded
            # Random sequence length
            len1 = len(tweetList_encoded[i])
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(len1)
            #data
            s = [[tweetList_encoded[i][j]] for j in range(len1)]
            s += [[0] for j in range(max_seq_len - len1 )]
            self.data.append(s)
            self.labels.append([labelList[i]])
        self.batch_id = 0
        #print(self.data)
        #print('Shape of data: {}'.format(np.array(self.data).shape))

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen

# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
seq_max_len = 31 # Sequence max length
n_hidden = 31 #todo?? hidden layer num of features
n_classes = 1 # linear sequence or not

state_size = 1#20

trainset = ToySequenceData(n_samples=len(tweetList_encoded), max_seq_len=seq_max_len)
testset = ToySequenceData(n_samples=len(tweetList_encoded), max_seq_len=seq_max_len)#todo

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, state_size])
y = tf.placeholder("float", [None, n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Embedding layer
#embeddings = tf.get_variable('embedding_matrix', [seq_max_len, state_size])
#rnn_inputs = tf.nn.embedding_lookup(embeddings, x)


# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def dynamicRNN(x, seqlen, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)
    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(seq_max_len)#todo

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

pred = dynamicRNN(x, seqlen, weights, biases)#todoo

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps + 1):
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen})

        if step % display_step == 0 or step == 1:
            # Calculate batch accuracy & loss
            acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y,
                                                seqlen: batch_seqlen})
            print("Step " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                      seqlen: test_seqlen}))