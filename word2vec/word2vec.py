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
import math

import tensorflow as tf
from .configs import Configs

VOCAB_SIZE = 100000
EMBEDDING_SIZE = 128
SKIP_WINDOW = 2

NUM_SAMPLED = 64

BATCH_SIZE = 32

LOG_DIR = "/tmp/word2vec"

TRAIN_STEPS = 100001


class SkipGramModel(object):

  def __init__(self, configs=Configs):
    self._build_graph()

  def _build_graph(self):
    self.graph = tf.Graph()
    with self.graph.as_default():
      with tf.name_scope("inputs"):
        self.train_inputs = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE])
        self.train_labels = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE])

      with tf.device("/cpu:0"):
        with tf.name_scope("embeddings"):
          self.embeddings = tf.Variable(
            tf.random_uniform([BATCH_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
          self.embedded_inputs = tf.nn.embedding_lookup(
            self.embeddings, self.train_inputs)

        with tf.name_scope("weights"):
          self.nce_weights = tf.Variable(
            tf.truncated_normal([VOCAB_SIZE, EMBEDDING_SIZE],
                                stddev=1.0 / math.sqrt(EMBEDDING_SIZE)))
          with tf.name_scope("bias"):
            self.nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]))

      with tf.name_scope("loss"):
        self.loss = tf.reduce_mean(
          tf.nn.nce_loss(
            weights=self.nce_weights,
            biases=self.nce_bias,
            labels=self.train_labels,
            inputs=self.embedded_inputs,
            num_sampled=NUM_SAMPLED,
            num_classes=VOCAB_SIZE))

      tf.summary.scalar("loss", self.loss)

      with tf.name_scope("optimizer"):
        self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(
          self.loss)

  def _generate_batch_inputs(self):
    raise NotImplementedError()

  def train(self):
    merged_summary = tf.summary.merge_all()
    with tf.Session(graph=self.graph) as sess:
      writer = tf.summary.FileWriter(LOG_DIR, graph=sess.graph)

      sess.run(tf.global_variables_initializer())

      avg_loss = 0
      for step in range(TRAIN_STEPS):
        inputs, labels = self._generate_batch_inputs()

        feed_dict = {self.train_inputs: inputs,
                     self.train_labels: labels}

        _, summary, loss = sess.run([self.optimizer, merged_summary, self.loss],
                                    feed_dict=feed_dict)

        avg_loss += loss
        writer.add_summary(summary, step)
