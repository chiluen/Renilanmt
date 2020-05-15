#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(".")
from explab import *

from nmtlab import Vocab, MTDataset
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
import pdb; pdb.set_trace()
# print(tf.string_split(text).values.eval())
raise SystemExit
