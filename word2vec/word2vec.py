# Copyright 2018 luozhouyang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

import numpy as np
import tensorflow as tf

from .data import SkipGramDataSet

dataset = SkipGramDataSet(os.path.join(os.path.curdir, "word2vec/test.txt"))

VOCAB_SIZE = dataset.vocab_size
EMBEDDING_SIZE = 128
SKIP_WINDOW = 2

NUM_SAMPLED = 64

BATCH_SIZE = 32
WINDOW_SIZE = 2
LOG_DIR = "/tmp/word2vec"

TRAIN_STEPS = 10000

LEARNING_RATE = 0.1


class Word2Vec(object):

  def __init__(self):
    self.graph = tf.Graph()
    with self.graph.as_default():
      with tf.name_scope("inputs"):
        self.x = tf.placeholder(shape=(None, VOCAB_SIZE), dtype=tf.float32)
        self.y = tf.placeholder(shape=(None, VOCAB_SIZE), dtype=tf.float32)

      with tf.name_scope("layer1"):
        self.W1 = tf.Variable(
          tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE], -1, 1),
          dtype=tf.float32)
        self.b1 = tf.Variable(tf.random_normal([EMBEDDING_SIZE]),
                              dtype=tf.float32)
      hidden = tf.add(self.b1, tf.matmul(self.x, self.W1))

      with tf.name_scope("layer2"):
        self.W2 = tf.Variable(
          tf.random_uniform([EMBEDDING_SIZE, VOCAB_SIZE], -1, 1),
          dtype=tf.float32)
        self.b2 = tf.Variable(tf.random_normal([VOCAB_SIZE]),
                              dtype=tf.float32)

      self.prediction = tf.nn.softmax(
        tf.add(tf.matmul(hidden, self.W2), self.b2))

      log = self.y * tf.log(self.prediction)
      self.loss = tf.reduce_mean(
        -tf.reduce_sum(log, reduction_indices=[1], keepdims=True))

      self.opt = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
        self.loss)

  def _one_hot_input(self, dataset):
    features, labels = dataset.gen_batch_inputs(BATCH_SIZE, WINDOW_SIZE)
    f, l = [], []
    for w in features:
      tmp = np.zeros([VOCAB_SIZE])
      tmp[w] = 1
      f.append(tmp)
    for w in labels:
      tmp = np.zeros(VOCAB_SIZE)
      tmp[w] = 1
      l.append(tmp)
    return f, l

  def train(self, dataset, n_iters, ):
    with tf.Session(graph=self.graph) as sess:
      sess.run(tf.global_variables_initializer())
      for i in range(n_iters):
        features, labels = self._one_hot_input(dataset)

        predi, loss = sess.run([self.prediction, self.loss],
                               feed_dict={
                                 self.x: features,
                                 self.y: labels
                               })
        print("loss:%s" % loss)

  def predict(self):
    pass

  def nearest(self, n):
    pass

  def similarity(self, a, b):
    pass


word2vec = Word2Vec()
word2vec.train(dataset, TRAIN_STEPS)
