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
import tensorflow as tf
import collections
import numpy as np
import random


class DataSet(object):

  def __init__(self, file):
    self.file = file

    self.data_index = 0

    self._build_dataset()

  def _build_dataset(self):
    if not os.path.exists(self.file):
      raise ValueError("File does not exist --> %s" % self.file)
    f = open(self.file, mode='rt', encoding='utf8')
    self.data = tf.compat.as_str(f.read()).split()
    if f:
      f.close()

    c = collections.Counter(self.data).most_common()
    self.vocab_size = len(c)

    self.counter = c.insert(0, ('UNK', -1))
    self.vocab_size += 1

    self.word2id = dict()
    self.id2word = dict()
    for word, _ in c:
      self.word2id[word] = len(self.word2id)
      self.id2word[len(self.id2word)] = word

  def gen_batch_inputs(self, batch_size, num_skips, skip_window):
    raise NotImplementedError()


class SkipGramDataSet(DataSet):

  def gen_batch_inputs(self, batch_size, num_skips, skip_window):
    """generate a batch of features and labels.

    self.data_index is the first word index in self.data of a span,
    the target word is the central word of a span,
    context words are pre and post words of the central word.
    A span looks like:
    [context_word_0, context_word_1, target, context_word_2, context_word_3]
          |                             |
          self.data_index            target word
    """

    assert batch_size % num_skips == 0
    # We randomly choose words from a range [0, 2*skip_windows+1], and we can not choose the one in (2*skip_window+1)/2
    assert num_skips <= 2 * skip_window

    features = np.ndarray(shape=(batch_size,), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,), dtype=np.int32)

    span = 2 * skip_window + 1  # [skip_windows target skip_window]
    buffer = collections.deque(maxlen=span)

    if self.data_index + span > len(self.data):
      self.data_index = 0

    # fetch a list of words of span length
    buffer.extend(self.word2id[w] for w in self.data[self.data_index:self.data_index + span])
    self.data_index += span

    for i in range(batch_size // num_skips):
      context_words_ids = [w for w in range(span) if w != skip_window]
      # This will create num_skip words, so we need do `batch_size // num_skip` to
      # ensure return features and labels in batch_size
      use_words = random.sample(context_words_ids, num_skips)

      for j, context_word_id in enumerate(use_words):
        # position offset from 0.
        features[i * num_skips + j] = buffer[skip_window]
        labels[i * num_skips + j] = buffer[context_word_id]

      # When come to the last word fo data, fetch words of size span from the beginning of data.
      # Then move index to the beginning offset span.
      if self.data_index == len(self.data):
        buffer.extend(self.data[0:span])
        self.data_index = span
      else:
        buffer.append(self.data[self.data_index])
        self.data_index += 1

    self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
    return features, labels
