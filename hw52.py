import csv
with open('train.csv', "rt", encoding='utf-8', errors='ignore') as csvfile:
    twtreader = csv.reader(csvfile)

    trainlistoriginal = list(twtreader)
    trainlistoriginal = trainlistoriginal[1:]

    traintar = [None] * len(trainlistoriginal) #raw label
    trainlist = [None] * len(trainlistoriginal) #raw data

    for i in range(0, len(trainlistoriginal)):
        traintar[i] = trainlistoriginal[i][0]
        trainlist[i] = trainlistoriginal[i][1]

    str = "http"
    for i in range(0, len(trainlist)):
        strcount = trainlist[i].count(str)
        if strcount > 0:
            for j in range(0, strcount):
                mystr = trainlist[i]
                ind = mystr.index(str, 0)
                trainlist[i] = trainlist[i][0:ind-1] + trainlist[i][ind + 23-1: -1]

    #trainset label
    trainumtar = [[0 for i in range(2)] for i in range(len(traintar))]
    for i in range(0, len(traintar)):
        if traintar[i] == "HillaryClinton":
            trainumtar[i][0] = 1
        else:
            trainumtar[i][1] = 1

#tokenize, update 'trainlist'
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
trainlistallwords = set()
for i in range(0, len(trainlist)):
    trainlist[i] = tokenizer.tokenize(trainlist[i])
    trainlistallwords.update(trainlist[i])

#max length of all tweets
maxlen = 0
for tweet in trainlist:
    if len(tweet) > maxlen:
        maxlen = len(tweet)

trainlistallwords = list(trainlistallwords)
N = len(trainlistallwords) # N: numbers of words

dict = {}
wordnum = 0
for word in trainlistallwords:
    dict[word] = wordnum
    wordnum += 1

#return sub-dataset of each batch
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

    testlist_processed = []
    for i in range(len(testlist)):
        new_test_list = []
        for j in range(len(testlist[i])):
            if dict.get(testlist[i][j]) != None:
                new_test_list.append(testlist[i][j])
        testlist_processed.append(new_test_list)

def gettestlabel(index):
    test_label = [0] * maxlen
    for i in range(min(len(testlist_processed[index]), maxlen)):
        test_label[i] = dict[testlist_processed[index][i]]
    return test_label

testdata = []
for i in range(len(testlist)):
    testdata.append(gettestlabel(i))

test_len = []
for i in range(len(testlist)):
    test_len.append(min(len(testlist_processed[i]), maxlen))

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
embedding_size = 30
lstm_size = 30
senten_words_num = maxlen
class_num = 2
learning_rate = 1
iter_num = 500
iter_display_step = 10
#nn
nn1 = 30
nn2 = 30

#batch_size = 128
# batch_num = len(trainlist)

X = tf.placeholder(tf.int32, [None, maxlen])
Y = tf.placeholder(tf.float32, [None, class_num])
sentense_length = tf.placeholder(tf.int32, [None])

#weight and bias
weights = {
    'w1' : tf.Variable(tf.random_normal([embedding_size, nn1])),
    'w2' : tf.Variable(tf.random_normal([nn1, nn2])),
    'sigmoid': tf.Variable(tf.random_normal([nn2, class_num], stddev=1, dtype=tf.float32))
}
bias = {
    'b1' : tf.Variable(tf.random_normal(shape=[nn1])),
    'b2' : tf.Variable(tf.random_normal(shape=[nn2])),
    'sigmoid' : tf.Variable(tf.random_normal(shape=[class_num], dtype=tf.float32))
}
embeddings = tf.Variable(tf.random_uniform([vocalbulary_size, embedding_size], -1.0, 1.0))
inputs = tf.nn.embedding_lookup(embeddings, X)

#lstm
lstm_cells = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_size)
outputs, state = tf.nn.dynamic_rnn(cell=lstm_cells, inputs=inputs, dtype=tf.float32, sequence_length=sentense_length)
last_output_idx = tf.range(tf.shape(outputs)[0]) * tf.shape(outputs)[1] + sentense_length - 1
last_rnn_output = tf.gather(tf.reshape(outputs, [-1, lstm_size]), last_output_idx)

#nn layer
nn_layer_1 = tf.add(tf.matmul(last_rnn_output, weights['w1']), bias['b1'])
nn_layer_2 = tf.add(tf.matmul(nn_layer_1, weights['w2']), bias['b2'])

#sigmoid layer
one_batch_predict = tf.nn.sigmoid(tf.matmul( last_rnn_output, weights['sigmoid']) + bias['sigmoid'])

#loss
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=one_batch_predict, labels=Y))

#backward
train_method = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)

accurate_position = tf.equal(tf.argmax(Y, 1), tf.argmax(one_batch_predict, 1))
accuracy = tf.reduce_mean(tf.cast(accurate_position, tf.float32))

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=1)




#train
loss_list=[]
iter_list=[]
with tf.Session() as sess:
    sess.run(init)
    import math
    batch_size = 128
    train_batch_num = math.ceil(len(trainlist) / batch_size)

    #valid data
    val_x = []
    val_y = []
    val_len = []
    for i in range((train_batch_num - 5) * batch_size, len(trainlist)):
        val_x.append(traindata[i])
        val_y.append(trainumtar[i])
        val_len.append(len(trainlist[i]))

    for i in range(iter_num):
        for batchi in range(train_batch_num):
            batch_x, batch_y, batch_len = gettrainbatch(batchi, batch_size)
            sess.run(train_method, feed_dict={X: batch_x, Y: batch_y, sentense_length: batch_len})
        
        #check
        if((i%iter_display_step==0) or i==1):
            loss = sess.run(cross_entropy_loss,  feed_dict={X: batch_x, Y: batch_y, sentense_length: batch_len})
            loss_list.append(loss)
            iter_list.append(i)
            val_loss = sess.run([accuracy, cross_entropy_loss], feed_dict={X: val_x, Y: val_y, sentense_length: val_len})
            print('step {}. trainset: loss={}, testset: loss={}'.format(i,loss,val_loss))

    #plot convergence speed
    import matplotlib.pyplot as plt
    plt.title('convergence speed')
    plt.xlabel('iteration times')
    plt.ylabel('train error')
    plt.plot(iter_list, loss_list, color="red")
    plt.grid(True)
    plt.legend()
    plt.show()



    saver.save(sess, "Model/model.ckpt")
    saver.restore(sess, "./Model/model.ckpt")

    val_acc, val_loss = sess.run([accuracy, cross_entropy_loss], feed_dict={X: val_x, Y: val_y, sentense_length: val_len})
    print('val_acc:%f, val_loss:%f' % (val_acc, val_loss))
    test_result = sess.run(one_batch_predict, feed_dict={X: testdata, sentense_length: test_len})

csvFile = open("test_result.csv", "w")
writer = csv.writer(csvFile)

fileheader = ["id", "realDonaldTrump", "HillaryClinton"]

writer.writerow(fileheader)

for i in range(len(test_result)):

    onerow = [i, test_result[i][1], test_result[i][0]]
    writer.writerow(onerow)

csvFile.close()