#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(".")
from explab import *

from nmtlab import Vocab, MTDataset
from nnlab import nn
import nnlab.nn.functional as F
import tensorflow as tf

ss = tf.InteractiveSession()

vocab_path = "{}/iwslt14.tokenized.de-en/iwslt14.en.bpe20k.vocab".format(MAINLINE_ROOT)
vocab = Vocab(vocab_path)
index_table = vocab.get_index_table()

text_path = "{}/iwslt14.tokenized.de-en/valid.en.bpe20k".format(MAINLINE_ROOT)


d = MTDataset(text_path, text_path, vocab_path, vocab_path)
it, a, b = d.get_iterator()
ss.run(a.init)
ss.run(b.init)
ss.run(it.initializer)
print(it.get_next()[1].eval())

# embedding = tf.get_variable("emb", [vocab.size(), 300], tf.float32)
# result = tf.nn.embedding_lookup(embedding, it.get_next()[1])
# ss.run(tf.global_variables_initializer())
src, tgt = it.get_next()
src_len = (tf.reduce_sum(tf.to_int32(src > 0), axis=1))
tgt_len = (tf.reduce_sum(tf.to_int32(tgt > 0), axis=1))
src_emb_layer = nn.Embedding(vocab.size(), 256)
tgt_emb_layer = nn.Embedding(vocab.size(), 256)
inp_emb = src_emb_layer.get(F.tensor_from_tf(tgt)).tf
tgt_emb = tgt_emb_layer.get(F.tensor_from_tf(tgt)).tf
with tf.variable_scope("enc"):
    fw_cell = tf.nn.rnn_cell.LSTMCell(256)
    bw_cell = tf.nn.rnn_cell.LSTMCell(256)
    
    (fw_states, bw_states), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inp_emb, sequence_length=src_len, swap_memory=True, dtype=tf.float32)
with tf.variable_scope("dec"):
    tgt_fw_cell = tf.nn.rnn_cell.LSTMCell(256)
    tgt_bw_cell = tf.nn.rnn_cell.LSTMCell(256)
    (tgt_fw_states, tgt_bw_states), _ = tf.nn.bidirectional_dynamic_rnn(tgt_fw_cell, tgt_bw_cell, tgt_emb, sequence_length=tgt_len, swap_memory=True, dtype=tf.float32)

last_enc_fw_state = fw_states[:, -1]
first_enc_bw_state = bw_states[:, 0]
last_dec_fw_state = tgt_fw_states[:, -1]
first_dec_bw_state = tgt_bw_states[:, 0]
dense1 = nn.Dense(256 * 2, 256)
dense2 = nn.Dense(256 * 2, 256)
vec1 = dense1.forward(F.tensor_from_tf(tf.concat([last_enc_fw_state, first_enc_bw_state], axis=1)))
vec2 = dense2.forward(F.tensor_from_tf(tf.concat([last_dec_fw_state, first_dec_bw_state], axis=1)))
logit = tf.nn.sigmoid((vec1 * vec2).sum(axis=1))


import pdb; pdb.set_trace()
# print(ss.run(result).shape)
# print(result)


