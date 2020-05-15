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
em = nn.Embedding(vocab.size(), 300)
result = em.get(F.tensor_from_tf(it.get_next()[1]))
# print(ss.run(result).shape)
print(result)


