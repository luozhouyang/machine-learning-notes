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
import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):

  def __init__(self, attention_dropout=0.0, scale=True):
    super(ScaledDotProductAttention, self).__init__()
    self.scale = scale
    self.dropout = nn.Dropout(attention_dropout)
    self.softmax = nn.Softmax(dim=2)

  def forward(self, q, k, v, mask=None):
    """Forward pass.

    Args:
      q: Queries tensor, with shape of [B, L_q, D_q]
      k: Keys tensor, with shape of [B, L_k, D_k]
      v: Values tensor, with shape of [B, L_v, D_v]
      mask: A ByteTensor, binary mask. If not None, do mask.
    """
    attention = torch.bmm(q, k.transpose(1, 2))
    if self.scale:
      d_k = k.size(-1)  # get model's dimension or num_units
      attention = attention / np.sqrt(d_k)
    if mask:
      attention = attention.masked_fill_(mask, -np.inf)
    attention = self.softmax(attention)
    attention = self.dropout(attention)
    context = torch.bmm(attention, v)
    return context, attention


class LayerNorm(nn.Module):
  """Layer Normalization. Pytorch has already implemented this `nn.LayerNorm`"""

  def __init__(self, features, epsilon=1e-6):
    """Init.

    Args:
      features: Number of features,
        or the last dim of input x with shape of [B, L, D]
      epsilon: A small number to avoid numeric error.
    """
    super(LayerNorm, self).__init__()
    # weights
    self.gamma = nn.Parameter(torch.ones(features))
    # bias
    self.beta = nn.Parameter(torch.zeros(features))
    self.epsilon = epsilon

  def forward(self, x):
    """Forward pass.

    Args:
      x: Input tensor, with shape of [B, L, D]
    """
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


class PositionalEncoding(nn.Module):

  def __init__(self, d_model, max_seq_len):
    """Init.

    Args:
      d_model: The model's dimension,
        the last dimension of input x with shape of [B, L, D]
      max_seq_len: The maximum sequence length, or the time-steps,
        the second dimension of input x with shape of [B, L, D]
    """
    super(PositionalEncoding, self).__init__()

    position_encoding = np.array([
      [pos / np.pow(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
      for pos in range(max_seq_len)])
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

    pad_row = torch.zeros([1, d_model])
    position_encoding = torch.cat((pad_row, position_encoding))
    # additional PAD position index
    self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)

    self.position_encoding.weight = nn.Parameter(position_encoding,
                                                 requires_grad=False)

  def forward(self, input_len):
    """Forward pass.

    Args:
      input_len: A tensor with shape [B, 1]. Each element's value in ths tensor
        is the length of a sequence from a mini batch.

    Returns:
      Position encoding(or position embedding) of a mini batch sequence.
    """
    max_len = torch.max(input_len)
    tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
    # we start from 1 because 0 is for PAD
    # we pad a sequence with PAD(0) to the max length
    input_pos = tensor(
      [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
    return self.position_encoding(input_pos)
