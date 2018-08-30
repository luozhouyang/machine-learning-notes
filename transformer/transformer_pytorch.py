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

import numpy as np
import torch
import torch.functional as F
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):

  def __init__(self, attention_dropout=0.0, scale=True):
    super(ScaledDotProductAttention, self).__init__()
    self.scale = scale
    self.dropout = nn.Dropout(attention_dropout)
    self.softmax = nn.Softmax(dim=2)

  def forward(self, q, k, v, padding_mask=None):
    """Forward pass.

    Args:
      q: Queries tensor, with shape of [B, L_q, D_q]
      k: Keys tensor, with shape of [B, L_k, D_k]
      v: Values tensor, with shape of [B, L_v, D_v]
      padding_mask: A binary masking tensor, with shape of [B, L_q, L_k]
    """
    attention = torch.bmm(q, k.transpose(1, 2))
    if self.scale:
      d_k = k.size(-1)  # get model's dimension or num_units
      attention = attention / np.sqrt(d_k)
    if padding_mask:
      # Mask out attention, set a big negative number to where were padded a `PAD`
      attention = attention.masked_fill_(padding_mask, -np.inf)
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


class MultiHeadAttention(nn.Module):
  """Multi-Head attention as described in paper <Attention Is All You Need>."""

  def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
    """Init.

    Args:
      model_dim: Model's dimension, default is 512 according to the paper
      num_heads: Number of heads, default is 8 according to the paper
      dropout: Dropout rate for dropout layer
    """
    super(MultiHeadAttention, self).__init__()

    self.dim_per_head = model_dim // num_heads
    self.num_heads = num_heads
    self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
    self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
    self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

    self.dot_product_attention = ScaledDotProductAttention(dropout, scale=True)
    self.linear_final = nn.Linear(model_dim, model_dim)
    self.dropout = nn.Dropout(dropout)
    self.layer_norm = nn.LayerNorm(model_dim)

  def forward(self, key, value, query, padding_mask=None, seq_mask=None):
    """Forward pass.

    Args:
      key: Key tensor, with shape of [B, L_k, D]
      value: Value tensor, with shape of [B, L_v, D]
      query: Query tensor, with shape of [B, L_q, D]
      padding_mask: Binary mask indicating which position is padding values, with shape of [B, L_q, L_k]
      seq_mask: Binary mask indicating which keys has non-zero attention, with shape of [B, L, L]
    """
    residual = query

    dim_per_head = self.dim_per_head
    num_heads = self.num_heads
    batch_size = key.size(0)

    # linear projection
    key = self.linear_k(key)
    value = self.linear_v(value)
    query = self.linear_q(query)

    # split by heads
    key = key.view(batch_size * num_heads, -1, dim_per_head)
    value = value.view(batch_size * num_heads, -1, dim_per_head)
    query = query.view(batch_size * num_heads, -1, dim_per_head)

    # TODO(luozhouyang) decide where to do seq_mask

    if padding_mask:
      # TODO(luozhouyang): check the shape of mask
      padding_mask = padding_mask.repeat(num_heads, 1, 1)
    # scaled dot product attention
    context, attention = self.dot_product_attention(query, key, value, padding_mask)

    # concat heads
    context = context.view(batch_size, -1, dim_per_head * num_heads)

    # final linear projection
    output = self.linear_final(context)

    # dropout
    output = self.dropout(output)

    # add residual and norm layer
    output = self.layer_norm(residual + output)

    return output, attention


class PositionalWiseFeedForward(nn.Module):
  """Positional-wise feed forward network."""

  def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
    """Init.

    Args:
      model_dim: Model's dimension, default is 512 according to the paper.
      ffn_dim: Hidden size of the feed forward network,
        default is 2048 according to the paper.
      dropout: Dropout rate.
    """
    super(PositionalWiseFeedForward, self).__init__()
    self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
    self.w2 = nn.Conv1d(model_dim, ffn_dim, 1)
    self.dropout = nn.Dropout(dropout)
    self.layer_norm = nn.LayerNorm(model_dim)

  def forward(self, x):
    """Forward pass.

    Args:
      x: Input tensor, with shape of [B, L, D]

    Returns:
      A tensor with shape of [B, L, D], without residual value and normalization
    """
    output = x.transpose(1, 2)
    output = self.w2(F.relu(self.w1(output)))
    output = self.dropout(output.transpose(1, 2))

    # add residual and norm layer
    output = self.layer_norm(x + output)
    return output


class EncoderLayer(nn.Module):
  """A encoder block, with tow sub layers."""

  def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
    super(EncoderLayer, self).__init__()

    self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
    self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

  def forward(self, inputs, mask=None):
    """Forward pass.

    Args:
      inputs: embedded inputs
      mask: mask

    Returns:
      Output and attention tensors of encoder layer.
    """
    context, attention = self.attention(inputs, inputs, inputs, mask)
    output = self.feed_forward(context)
    return output, attention


class Encoder(nn.Module):

  def __init__(self,
               vocab_size,
               max_seq_len,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.0):
    super(Encoder, self).__init__()

    self.num_layers = num_layers
    self.encoder_layers = nn.ModuleList(
      [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])

    self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
    self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

  def forward(self, inputs, inputs_len, mask=None):
    """Forward pass.

    Args:
      inputs: embedded inputs
      inputs_len: length of input sequence
      mask: mask

    Returns:
      An output of encoder block.
      An attention list contains each attention tensor of each encoder layer.
    """
    output = self.seq_embedding(inputs)
    output += self.pos_embedding(inputs_len)
    attentions = []
    for i in range(self.num_layers):
      output, attention = self.encoder_layers[i](output, mask)
      if attention:
        attentions.append(attention)
    return output, attentions


class DecoderLayer(nn.Module):

  def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
    super(DecoderLayer, self).__init__()

    self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
    self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

  def forward(self, inputs, keys, values, seq_mask=None, padding_mask=None):
    """Forward pass.

    Args:
      inputs: Embedded input tensor
      keys: Keys tensor from encoder, used for enc-dec attention
      values: Values tensor form encoder, used for enc-dec attention
      seq_mask: Sequence mask tensor, with shape of [B, L, L]
      padding_mask: Padding mask tensor, with shape of [B, L_k, L_q]
    """
    context, self_attention = self.attention(inputs, inputs, inputs, mask)

    context, enc_dec_attention = self.attention(keys, values, context, mask)

    output = self.feed_forward(context)

    return output, self_attention, enc_dec_attention


class Decoder(nn.Module):

  def __init__(self,
               vocab_size,
               max_seq_len,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.0):
    super(Decoder, self).__init__()

    self.num_layers = num_layers

    self.decoder_layers = nn.ModuleList(
      [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])

    self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
    self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

  def forward(self, inputs, inputs_len, keys, values, mask=None):
    output = self.seq_embedding(inputs)
    output += self.pos_embedding(inputs_len)

    self_attentions = []
    context_attentions = []
    for decoder in self.decoder_layers:
      output, self_attn, context_attn = decoder(output, keys, values, mask)
      self_attentions.append(self_attn)
      context_attentions.append(context_attn)

    return output, self_attentions, context_attentions


def sequence_mask(seq):
  """Sequence mask to masking out sub-sequence info.

  Args:
    seq: Sequence tensor, with shape [B, L]

  Returns:
    A masking tensor, with shape [B, L, L]
  """
  batch_size, seq_len = seq.size()
  mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
  mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
  return mask


def padding_mask(seq_k, seq_q):
  """For masking out the padding part of the keys sequence.

  Args:
    seq_k: Keys tensor, with shape [B, L_k]
    seq_q: Query tensor, with shape [B, L_q]

  Returns:
    A masking tensor, with shape [B, L_1, L_k]
  """
  len_q = seq_q.size(1)
  # `PAD` is 0
  pad_mask = seq_k.eq(0)
  pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
  return pad_mask


class Transformer(nn.Module):

  def __init__(self,
               src_vocab_size,
               src_max_len,
               tgt_vocab_size,
               tgt_max_len,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.0):
    super(Transformer, self).__init__()

    self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, model_dim, num_heads, ffn_dim, dropout)
    self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim, num_heads, ffn_dim, dropout)

    self.linear = nn.Linear(model_dim, tgt_vocab_size + 1, bias=False)
    self.softmax = nn.Softmax(dim=2)

  def forward(self, src_seq, src_len, tgt_seq, tgt_len):
    # TODO(luozhouyang) create mask
    mask = None
    output, _ = self.encoder(src_seq, src_len, mask)
    output, _, _ = self.decoder(tgt_seq, tgt_len, mask)

    output = self.linear(output)
    output = self.softmax(output)
    return output
