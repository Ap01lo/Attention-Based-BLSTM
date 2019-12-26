from tensorflow.nn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.nn import dynamic_rnn as rnn
from tensorflow.contrib.rnn import BasicLSTMCell as Cell
# from tensorflow.contrib.rnn import GRUCell as Cell
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import glob
import os
import time
import gc
import random

class ABLSTM(object):
	"""docstring for ABLSTM"""
	def __init__(self, config):
		#super(ABLSTM, self).__init__()
		self.time_step = config["time_step"]
		self.hidden_size = config["hidden_size"]
		# self.vocab_size = config["vocab_size"]
		# self.embedding_size = config["embedding_size"]
		self.n_class = config["n_class"]
		self.learning_rate = config["learning_rate"]

		# placeholser
		self.x = tf.compat.v1.placeholder(tf.float32, [None, self.time_step, 90])
		self.label = tf.compat.v1.placeholder(tf.int32, [None, self.n_class])
		self.keep_prob = tf.compat.v1.placeholder(tf.float32)
		
	def build_graph(self):
		print("building graph")
        
		rnn_outputs, _ = bi_rnn(Cell(self.hidden_size), Cell(self.hidden_size), inputs=self.x, dtype=tf.float32)

		fw_output, bw_output = rnn_outputs

		H = fw_output + bw_output  # (batch_size, time_step, hidden_size)
		
		# attention
		# att_size = H.shape[2].value
		W = tf.Variable(tf.random_normal([self.hidden_size], stddev=0.1))
		M = tf.tanh(H)  # (batch_size, time_step, hidden_size)

		self.alpha = tf.nn.softmax(tf.tensordot(M, W, axes=1))
		r = tf.reduce_sum(H*tf.expand_dims(self.alpha, -1), 1)
		h_star = tf.tanh(r)  #(batch, hidden_size)

		# Dropout layer
		h_drop = tf.nn.dropout(h_star, self.keep_prob)

		# Fully connected layer
		# FC_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.n_class], stddev=0.1))
  #       FC_b = tf.Variable(tf.constant(0., shape=[self.n_class]))
  #       y_hat = tf.nn.xw_plus_b(h_drop, FC_W, FC_b)
		y_hat = tf.layers.dense(h_drop, self.n_class, kernel_initializer=None)
		self.prediction = tf.argmax(tf.nn.softmax(y_hat), 1)

        # Calculate mean cross-entropy loss
		self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.label, logits=y_hat)

        # optimization
		# loss_to_minimize = self.loss
		# tvars = tf.trainable_variables()
		# gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
		# grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

		# self.global_step = tf.Variable(0, name="global_step", trainable=False)
		# # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		# self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
		# self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
		print("graph built successfully!")

# class BiLSTM(object):
# 	"""docstring for LSTM"""
# 	def __init__(self, config):
# 		# super(LSTM, self).__init__()
# 		self.time_step = config["time_step"]
# 		self.hidden_size = config["hidden_size"]
# 		self.n_class = config["n_class"]
# 		self.learning_rate = config["learning_rate"]

# 		#placeholser
# 		self.x = tf.placeholder(tf.float32, [None, self.time_step, 90])
# 		self.label = tf.placeholder(tf.int32, [None, self.n_class])
# 		self.keep_prob = tf.placeholder(tf.float32)

# 	def build_graph(self):
# 		print("building graph")

# 		run_outputs, _ = rnn(Cell(self.hidden_size), inputs=self.x, dtype=tf.float32)
# 		# fw_output, bw_output = run_outputs
# 		# outputs = fw_output+bw_output
# 		r = tf.reduce_mean(run_outputs, 1)

# 		h_star = tf.tanh(r)
# 		#dropout layer
# 		h_drop = tf.nn.dropout(h_star, self.keep_prob)
# 		#FC layer
# 		y_hat = tf.layers.dense(h_drop, self.n_class, kernel_initializer=None)
# 		self.prediction = tf.argmax(tf.nn.softmax(y_hat), 1)

# 		self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat, labels=self.label)
# 		self.train_op = tf.train.PMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
# 		print("graph built successfully!")

def fill_feed_dict(data_X, data_Y, batch_size):
    """Generator to yield batches"""
    # Shuffle data first.
    shuffled_X, shuffled_Y = shuffle(data_X, data_Y)
    # print("fill_shuffled successfully!")
    # print("before shuffle: ", data_Y[:10])
    # print(data_X.shape[0])
    # perm = np.random.permutation(data_X.shape[0])
    # data_X = data_X[perm]
    # shuffled_Y = data_Y[perm]
    # print("after shuffle: ", shuffled_Y[:10])
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = shuffled_X[batch_size * idx: batch_size * (idx + 1)]
        y_batch = shuffled_Y[batch_size * idx: batch_size * (idx + 1)]
        yield idx, x_batch, y_batch
    del shuffled_X, shuffled_Y
    gc.collect()

# def run_train_step(model, sess, batch):
#     feed_dict = {model.x: batch[0], model.label: batch[1], model.keep_prob: 0.5}
#     to_return = {
#         'train_op': model.train_op,
#         'loss': model.loss,
#         # 'global_step': model.global_step,
#     }
#     return sess.run(to_return, feed_dict)


def get_attn_weight(model, sess, batch):
    feed_dict = {model.x: batch[0], model.label: batch[1], model.keep_prob: 0.5}
    return sess.run(model.alpha, feed_dict)


def run_eval_step(model, sess, batch):
    feed_dict = {model.x: batch[0], model.label: batch[1], model.keep_prob: 1.0}
    prediction = sess.run(model.prediction, feed_dict)
#     print(prediction.shape)
#     print(tf.math.argmax(batch[1], 1).shape)
    acc = sess.run(tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(batch[1], 1)), tf.float32)))
#     print(acc.shape)
    return acc

def dataimport(path1, path2):
    window_size = 1000 #time_step
    threshold = 60
    slide_size = 200 #less than window_size!!!
    
    xx = np.empty([0,window_size,90],float)
    yy = np.empty([0,8],float)

	###Input data###
	#data import from csv
    input_csv_files = sorted(glob.glob(path1))
    for f in input_csv_files:
        print("input_file_name=",f)
        data = [[ float(elm) for elm in v] for v in csv.reader(open(f, "r"))]
        tmp1 = np.array(data)
        x2 =np.empty([0,window_size,90],float)
# 		print(tmp1.shape)

		#data import by slide window
        k = 0
        while k <= (len(tmp1) + 1 - 2 * window_size):
            x = np.dstack(np.array(tmp1[k:k+window_size, 1:91]).T)
# 			print(x.shape)
            x2 = np.concatenate((x2, x),axis=0)
# 			print(x2.shape)
            k += slide_size

        xx = np.concatenate((xx,x2),axis=0)
# 		print(xx.shape)
# 	xx = xx.reshape(len(xx),-1)
# 	print(xx.shape)


	###Annotation data###
	#data import from csv
    annotation_csv_files = sorted(glob.glob(path2))
    for ff in annotation_csv_files:
        print("annotation_file_name=",ff)
        ano_data = [[ str(elm) for elm in v] for v in csv.reader(open(ff,"r"))]
        tmp2 = np.array(ano_data)

        #data import by slide window
        y = np.zeros(((len(tmp2) + 1 - 2 * window_size)//slide_size+1,8))
        k = 0
        while k <= (len(tmp2) + 1 - 2 * window_size):
            y_pre = np.stack(np.array(tmp2[k:k+window_size]))
            bed = 0
            fall = 0
            walk = 0
            pickup = 0
            run = 0
            sitdown = 0
            standup = 0
            noactivity = 0
            for j in range(window_size):
                if y_pre[j] == "bed":
                    bed += 1
                elif y_pre[j] == "fall":
                    fall += 1
                elif y_pre[j] == "walk":
                    walk += 1
                elif y_pre[j] == "pickup":
                    pickup += 1
                elif y_pre[j] == "run":
                    run += 1
                elif y_pre[j] == "sitdown":
                    sitdown += 1
                elif y_pre[j] == "standup":
                    standup += 1
                else:
                    noactivity += 1

            if bed > window_size * threshold / 100:
                y[int(k/slide_size),:] = np.array([0,1,0,0,0,0,0,0])
            elif fall > window_size * threshold / 100:
                y[int(k/slide_size),:] = np.array([0,0,1,0,0,0,0,0])
            elif walk > window_size * threshold / 100:
                y[int(k/slide_size),:] = np.array([0,0,0,1,0,0,0,0])
            elif pickup > window_size * threshold / 100:
                y[int(k/slide_size),:] = np.array([0,0,0,0,1,0,0,0])
            elif run > window_size * threshold / 100:
                y[int(k/slide_size),:] = np.array([0,0,0,0,0,1,0,0])
            elif sitdown > window_size * threshold / 100:
                y[int(k/slide_size),:] = np.array([0,0,0,0,0,0,1,0])
            elif standup > window_size * threshold / 100:
                y[int(k/slide_size),:] = np.array([0,0,0,0,0,0,0,1])
            else:
                y[int(k/slide_size),:] = np.array([2,0,0,0,0,0,0,0])
            k += slide_size

        yy = np.concatenate((yy, y),axis=0)
    print(xx.shape,yy.shape)
    return (xx, yy)

def split_dataset(x,y,dev_ratio,test_ratio):
    x_size = len(x)
    train_dev_size = int(x_size * (1-test_ratio))
    x_train_dev = x[:train_dev_size]
    x_test = x[train_dev_size:]
    y_train_dev = y[:train_dev_size]
    y_test = y[train_dev_size:]

    train_size = int(x_size * (1-dev_ratio-test_ratio))
# 	print(train_size)
    x_train = x_train_dev[:train_size]
# 	print(x_train.shape)
    x_dev = x_train_dev[train_size:]
    y_train = y_train_dev[:train_size]
# 	print(y_train.shape)
    y_dev = y_train_dev[train_size:]

    return x_train, x_dev, x_test, y_train, y_dev, y_test



if __name__ == '__main__':
    # load data
    xx = np.empty([0,1000,90],float)
    yy = np.empty([0,8],float)
    for i, label in enumerate (["bed", "walk", "run", "sitdown", "standup", "fall", "pickup"]):
#         print(label,":")
        filepath1 = "/home/yan/Dataset/Data/input_" + str(label) + "*.csv"
        filepath2 = "/home/yan/Dataset/Data/annotation_" + str(label) + "*.csv"
        x, y = dataimport(filepath1, filepath2)
#         print("x:", x.shape, "y:", y.shape)
        xx = np.concatenate((xx, x),axis=0)
        yy = np.concatenate((yy, y),axis=0)
#         print(xx.shape, yy.shape)

    # xx, yy = shuffle(xx, yy)
    # print("shuffle successfully!")
    index = [i for i in range(len(xx))]
    random.shuffle(index)
    xx = xx[index]
    yy = yy[index]

    x_train, x_dev, x_test, y_train, y_dev, y_test = split_dataset(xx,yy,dev_ratio=0.1,test_ratio=0.1)
#     print("x_train:", x_train.shape, "x_dev:", x_dev.shape, "x_test:", x_test.shape, "y_train:", y_train.shape, "y_test:", y_test.shape, "y_dev:", y_dev.shape)
    del xx, yy
    gc.collect()

    config = {
        "time_step": 1000,
        "hidden_size": 200,
        # "vocab_size": vocab_size,
        # "embedding_size": 128,
        "n_class": 8,
        "learning_rate": 1e-3,
        "batch_size": 256,
        "train_epoch": 1
    }

    classifier = ABLSTM(config)
    classifier.build_graph()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # start = time.time()

    for e in range(config["train_epoch"]):
        
        t0 = time.time()
        print("Epoch %d start!" %(e+1))
        for c, x_batch, y_batch in fill_feed_dict(x_train, y_train, config["batch_size"]):
            # train_op, train_loss = run_train_step(classifier, sess, (x_batch, y_batch))
            train_loss, train_op = sess.run([classifier.loss, classifier.train_op], feed_dict={classifier.x: x_batch, classifier.label: y_batch, classifier.keep_prob: 0.5})
            print("batch train successfully!")
            # train_loss = tf.reduce_mean(loss)
            # attn = get_attn_weight(classifier, sess, (x_batch, y_batch))
            if c%10==0:
            	cou = 0
            	dev_acc = 0
            	for d, x_batch, y_batch in fill_feed_dict(x_dev, y_dev, config["batch_size"]):
            		accuracy = run_eval_step(classifier, sess, (x_batch, y_batch))
            		dev_acc += accuracy
            		cou += 1
            	print("Train step=%d, train loss=%.3f, validation accuracy: %.3f " % (c, train_loss, dev_acc/cou))
        t1 = time.time()
        print("Train Epoch time:  %.3f s" % (t1 - t0))
#        print(dev_acc.shape)
#	print("validation accuracy: ", dev_acc)

    # print("Training finished, time consumed : ", time.time() - start, " s")
    print("Start evaluating:  \n")
    cnt = 0
    test_acc = 0
    for e, x_batch, y_batch in fill_feed_dict(x_test, y_test, config["batch_size"]):
        acc = run_eval_step(classifier, sess, (x_batch, y_batch))
        test_acc += acc
        cnt += 1

    print("Test accuracy : %f %%" % (test_acc / cnt * 100))
