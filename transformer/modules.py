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

import numpy as np
import tensorflow as tf


def normalize(inputs, epsilon=1e-8, scope="norm", reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]  # 最后一行

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.Variable(tf.zeros(params_shape))
    gamma = tf.Variable(tf.ones(params_shape), dtype=tf.float32)
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
    position_encoding[:, 0::2] = tf.sin(position_encoding[:, 0::2])  # dim 2i
    position_encoding[:, 1::2] = tf.cos(position_encoding[:, 1::2])  # dim 2i+1

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


def dot_product_attention(queries,
                          keys,
                          values,
                          mode,
                          mask=None,
                          dropout=0.0):
  dot = tf.matmul(queries, keys, transpose_b=True)
  if mask:
    dot = dot * mask + ((1.0 - mask) * dot.dtype.min)

  attention = tf.nn.softmax(dot)
  attention = tf.layers.dropout(
    attention, rate=dropout, training=mode == tf.estimator.ModeKeys.TRAIN)
  context = tf.matmul(attention, values)
  return context, attention


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    # Set the fall back option for num_units
    if num_units is None:
      num_units = queries.get_shape().as_list[-1]

    # Linear projections
    Q = tf.layers.dense(queries, num_units,
                        activation=tf.nn.relu)  # (N, T_q, C)
    K = tf.layers.dense(keys, num_units,
                        activation=tf.nn.relu)  # (N, T_k, C)
    V = tf.layers.dense(keys, num_units,
                        activation=tf.nn.relu)  # (N, T_k, C)

    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2),
                   axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2),
                   axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2),
                   axis=0)  # (h*N, T_k, C/h)

    # Multiplication
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

    # Scale
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

    # Key Masking
    key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
    key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
    key_masks = tf.tile(tf.expand_dims(key_masks, 1),
                        [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(key_masks, 0), paddings,
                       outputs)  # (h*N, T_q, T_k)

    # Causality = Future blinding
    if causality:
      diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
      tril = tf.contrib.linalg.LinearOperatorTriL(
        diag_vals).to_dense()  # (T_q, T_k)
      masks = tf.tile(tf.expand_dims(tril, 0),
                      [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

      paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
      outputs = tf.where(tf.equal(masks, 0), paddings,
                         outputs)  # (h*N, T_q, T_k)

    # Activation
    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

    # Query Masking
    query_masks = tf.sign(
      tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
    query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
    query_masks = tf.tile(tf.expand_dims(query_masks, -1),
                          [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
    outputs *= query_masks  # broadcasting. (N, T_q, C)

    # Dropouts
    outputs = tf.layers.dropout(outputs, rate=dropout_rate,
                                training=tf.convert_to_tensor(is_training))

    # Weighted sum
    outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0),
                        axis=2)  # (N, T_q, C)

    # Residual connection
    outputs += queries

    # Normalize
    outputs = normalize(outputs)  # (N, T_q, C)

  return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    # Inner layer
    params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
              "activation": tf.nn.relu, "use_bias": True}
    outputs = tf.layers.conv1d(**params)

    # Readout layer
    params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
              "activation": None, "use_bias": True}
    outputs = tf.layers.conv1d(**params)

    # Residual connection
    outputs += inputs

    # Normalize
    outputs = normalize(outputs)

  return outputs


def label_smoothing(inputs, epsilon=0.1):
  K = inputs.get_shape().as_list()[-1]  # number of channels
  return ((1 - epsilon) * inputs) + (epsilon / K)
