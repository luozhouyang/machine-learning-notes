# Copyright 2018 luozhouyang
#
# This file is most copied from https://github.com/Kyubyong/transformer/blob/master/modules.py,
#  with the Apache License:
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
import numpy as np


def normalize(inputs, epsilon=1e-8, scope="norm", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]  # 最后一行

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / math.sqrt(variance + epsilon)
        outputs = gamma * normalized + beta

    return outputs


def embedding(inputs,
              vocab_size,
              embedding_size,
              zero_pad=True,
              scale=True,
              scope="embedding",
              reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable(
            "lookup_table",
            dtype=tf.float32,
            shape=[vocab_size, embedding_size],
            initializer=tf.contrib.layers.xavier_initializer())

        if zero_pad:
            # 在lookup_table第一行加上一行的0
            lookup_table = tf.concat((tf.zeros(shape=[1, embedding_size]),
                                      lookup_table[1:, :]), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (math.sqrt(embedding_size))

    return outputs


def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    batch_size, time_steps = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        # a=tf.range(time_steps)创建一个shape=(T)的矢量
        # b=tf.expand_dims(a,axis=0)在行上扩展，变成2维张量，shape=(1,T)
        # c=tf.tile(c,[batch_size,1])在c的第一个维度上，复制batch_size份，
        #   在c的第二个维度上，复制1份（不变）。此时张量shape=(batch_size,T)
        position_index = tf.tile(tf.expand_dims(tf.range(time_steps), 0),
                                 [batch_size, 1])

        # 根据position生成对应的数字，即position encoding
        position_encoding = np.array([pos / np.power(10000, 2. * i / num_units)
                                      for i in range(num_units)]
                                     for pos in range(time_steps))
        # 对奇数列使用cosine，对偶数列使用sin
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])  # dim 2i
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])  # dim 2i+1

        lookup_table = tf.convert_to_tensor(position_encoding)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]),
                                     0)
        # 根据位置position，就可以查出对应的数值
        # num_units应该不小于batch size??
        output = tf.nn.embedding_lookup(lookup_table, position_index)

        if scale:
            output = output * num_units ** 0.5

        return output
