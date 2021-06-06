import numpy as np
import torch
import torch.nn as nn
from abc import ABC


class DotProductAttention(nn.Module, ABC):
    """ Dot product attention """

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, #queries, `d`)
    # Shape of `keys`: (`batch_size`, #key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, #key-value pairs, value dimension)
    # Shape of `valid_lends`: (`batch_size`, #queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module, ABC):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, query, key, value, valid_len):
        # For self-attention, `query`, `key`, and `value` shape: (`batch_size`, `seq_len`, `dim`),
        # where `seq_len` is the length of input sequence.
        # `valid_len` shape is either (`batch_size`, ) or (`batch_size`, `seq_len`)

        # Project and transpose `query`, `key`, and `value` from
        # (`batch_size`, `seq_len`, `num_hiddens`) to
        # (`batch_size` * `num_heads`, `seq_len`, `num_hiddens` / `num_heads`)
        query = transpose_qkv(self.W_q(query), self.num_heads)
        key = transpose_qkv(self.W_k(key), self.num_heads)
        value = transpose_qkv(self.W_v(value), self.num_heads)

        if valid_len is not None:
            if valid_len.ndim == 1:
                valid_len = valid_len.repeat(self.num_heads)
            else:
                valid_len = valid_len.repeat(self.num_heads, 1)

        # For self-attention, `output` shape:
        # (`batch_size` * `num_heads`, `seq_len`, `num_hiddens` / `num_heads`)
        output = self.attention(query, key, value, valid_len)

        # `output_concat` shape:
        # (`batch_size`, `seq_len`, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class PositionWiseFFN(nn.Module, ABC):
    def __init__(self, ffn_num_input, ffn_num_hiddens,
                 pw_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, pw_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module, ABC):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class PositionalEncoding(nn.Module, ABC):
    def __init__(self, num_hiddens, dropout, max_len=1000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        # create a long enough `P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(0, max_len, dtype=torch.float32).reshape(-1, 1) \
            / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X += self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class EncoderBlock(nn.Module, ABC):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.add_norm_1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.add_norm_2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_len):
        Y = self.add_norm_1(X, self.attention(X, X, X, valid_len))
        return self.add_norm_2(Y, self.ffn(Y))


class DecoderBlock(nn.Module, ABC):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i  # the `i`-th block in the decoder
        self.attention_1 = MultiHeadAttention(key_size, query_size, value_size,
                                              num_hiddens, num_heads, dropout)
        self.add_norm_1 = AddNorm(norm_shape, dropout)
        self.attention_2 = MultiHeadAttention(key_size, query_size, value_size,
                                              num_hiddens, num_heads, dropout)
        self.add_norm_2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.add_norm_3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        encoder_outputs, encoder_valid_len = state[0], state[1]
        # `state[2][i]` contains the past queries for this block
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values

        if self.training:
            batch_size, seq_len, _ = X.shape
            valid_len = torch.arange(1, seq_len + 1, device=X.device).\
                repeat(batch_size, 1)
        else:
            valid_len = None

        X2 = self.attention_1(X, key_values, key_values, valid_len)
        Y = self.add_norm_1(X, X2)
        Y2 = self.attention_2(Y, encoder_outputs, encoder_valid_len)
        Z = self.add_norm_2(Y, Y2)
        return self.add_norm_3(Z, self.ffn(Z)), state


class TransformerEncoder(nn.Module, ABC):
    def __init__(self, input_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blocks = nn.Sequential()
        for i in range(self.num_layers):
            encoder_block = EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                         norm_shape, ffn_num_input, ffn_num_hiddens,
                                         num_heads, dropout, use_bias)
            self.blocks.add_module('block' + str(i), encoder_block)

    def forward(self, X, valid_len, *args):
        X = self.pos_encoding(self.embedding(X) * np.sqrt(self.num_hiddens))
        for block in self.blocks:
            X = block(X, valid_len)
        return X


class TransformerDecoder(nn.Module, ABC):
    def __init__(self, input_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blocks = nn.Sequential()
        for i in range(self.num_layers):
            decoder_block = DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                         norm_shape, ffn_num_input, ffn_num_hiddens,
                                         num_heads, dropout, i)
            self.blocks.add_module('block' + str(i), decoder_block)
        self.dense = nn.Linear(num_hiddens, input_size)

    def init_state(self, encoder_output, encoder_valid_len, *args):
        return [encoder_output, encoder_valid_len, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * np.sqrt(self.num_hiddens))
        for block in self.blocks:
            X, state = block(X, state)
        return self.dense(X), state


class EncoderDecoder(nn.Module, ABC):
    """ The base class for the encoder-decoder architecture. """
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input, decoder_input, *args):
        encoder_output = self.encoder(encoder_input, *args)
        decoder_state = self.decoder.init_state(encoder_output, *args)
        return self.decoder(decoder_input, decoder_state)


def masked_softmax(X, valid_lens):
    """ Perform softmax operation by masking elements on the last axis. """
    # `X`: 3-D tensor
    # `valid_lens`: 1-D or 2-D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


def sequence_mask(X, valid_len, value=0.):
    """ Mask irrelevant entries in sequences. """
    max_len = X.size(1)
    mask = torch.arange(max_len, dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def transpose_qkv(X, num_heads):
    # Input `X` shape: (`batch_size`, `seq_len`, `num_hiddens`).
    # Output `X` shape: (`batch_size`, `seq_len`, `num_heads`, `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # `X` shape: (`batch_size`, `num_heads`, `seq_len`, `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # `output` shape: (`batch_size` * `num_heads`, `seq_len`, `num_hiddens` / `num_heads`)
    output = X.reshape(-1, X.shape[2], X.shape[3])
    return output


def transpose_output(X, num_heads):
    # A reversed version of `transpose_qkv`
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


# X = torch.ones((2, 100, 32, 32))
# encoder_blk = EncoderBlock(32, 32, 32, 32, [100, 32, 32], 32, 64, 8, 0.5)
# encoder_blk.eval()
# valid_len = torch.tensor([2, 3])


#
# encoder = TransformerEncoder(input_size=None,
#                              key_size=None,
#                              query_size=None,
#                              value_size=None,
#                              num_hiddens=None,
#                              norm_shape=None,
#                              ffn_num_input=None,
#                              ffn_num_hiddens=None,
#                              num_heads=None,
#                              num_layers=None,
#                              dropout=None)
#
# decoder = TransformerDecoder(input_size=None,
#                              key_size=None,
#                              query_size=None,
#                              value_size=None,
#                              num_hiddens=None,
#                              norm_shape=None,
#                              ffn_num_input=None,
#                              ffn_num_hiddens=None,
#                              num_heads=None,
#                              num_layers=None,
#                              dropout=None)
