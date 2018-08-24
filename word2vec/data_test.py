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

import unittest

from .data import SkipGramDataSet
import os

_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


class TestDataSet(unittest.TestCase):

  def testGenBatchInputs(self):
    ds = SkipGramDataSet(file=os.path.join(_CURRENT_DIR, "__init__.py"))
    print(ds.word2id)
    print(ds.id2word)
    print(ds.data)

    print(ds.vocab_size)

    print('luozhouyang' in ds.word2id)
    print(ds.word2id['luozhouyang'])

    features, labels = ds.gen_batch_inputs(8, 2, 1)

    for i in range(8):
      print("%s --> %s" % (ds.id2word[features[i]], ds.id2word[labels[1]]))


if __name__ == "__main__":
  unittest.main()
