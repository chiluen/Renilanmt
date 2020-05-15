#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
from six.moves import xrange, zip

import os, sys
import tensorflow as tf
import time
import numpy as np

from nmtlab import Vocab


class NCEDataset(object):
    _PARALLEL_N = 4
    _NULL_ID = 0
    _BOS_ID = 1
    _EOS_ID = 2
    _UNK_ID = 3
    _SHUFFLE_SEED = 3
    _BUF_SZ = 30000
    
    def __init__(self, src_fp, tgt_fp, src_vocab, tgt_vocab, max_lines=None, gpu_id=0, gpu_num=1):
        if type(src_vocab) == str:
            assert os.path.exists(src_vocab)
            src_vocab = Vocab(src_vocab)
        if type(tgt_vocab) == str:
            assert os.path.exists(tgt_vocab)
            tgt_vocab = Vocab(tgt_vocab)
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab
        self._src_table = self._src_vocab.get_index_table()
        self._tgt_table = self._tgt_vocab.get_index_table()
        self._src_fp = src_fp
        self._tgt_fp = tgt_fp
        self._max_lines = max_lines
        self._gpu_id = gpu_id
        self._gpu_num = gpu_num
        self._batch_size = None
        self._maxlen = None
        self._progress = 0.
        self._rand = np.random.RandomState(3)
        self._src_tf = tf.placeholder(tf.string)
        self._tgt_tf = tf.placeholder(tf.string)
        self._label_tf = tf.placeholder(tf.int64, tuple())
        self._queue = tf.FIFOQueue(10000, [tf.string, tf.string, tf.int64],
                                            shapes=[[], [], []])
        self._size_op = self._queue.size()
        self._enqueue_op = self._queue.enqueue([self._src_tf, self._tgt_tf, self._label_tf])
        self._loaded = False
        self._session = None
        self._epoch_ended = True
    
    def progress(self):
        return self._progress
    
    def epoch_ended(self):
        assert self._batch_size is not None
        return self._epoch_ended and self._session.run(self._size_op) < self._batch_size
    
    def load_data(self):
        self._loaded = True
        self._src_lines = list(map(str.strip, open(self._src_fp)))
        self._tgt_lines = list(map(str.strip, open(self._tgt_fp)))
        if self._max_lines is not None:
            self._src_lines = self._src_lines[:self._max_lines]
            self._tgt_lines = self._src_lines[:self._max_lines]
        self._all_lines = list(zip(self._src_lines, self._tgt_lines))
        self._rand.shuffle(self._all_lines)
        
    def initialize(self, sess):
        self._session = sess
        sess.run(self._src_table.init)
        sess.run(self._tgt_table.init)
        
    def data_iterator(self):
        data_i = 0
        line_count = float(len(self._all_lines))
        rand = np.random.RandomState(3)
        for src, tgt in self._all_lines:
            if data_i % self._gpu_num == self._gpu_id:
                if src.count(" ") >= 50 or tgt.count(" ") >= 50:
                    continue
                if not src or not tgt:
                    continue
                yield (" ".join(src.split()[:10]), " ".join(tgt.split()[:10]), 1)
                for _ in xrange(5):
                    sample_i = rand.choice(len(self._tgt_lines))
                    # if self._tgt_lines[sample_i].count(" ") >= 50:
                    #     continue
                    yield (" ".join(src.split()[:10]), " ".join(self._tgt_lines[sample_i].split()[:10]), 0)
            data_i += 1
            self._progress = data_i / line_count
    
    def thread(self, session):
        self._epoch_ended = False
        for src, tgt, label in self.data_iterator():
            session.run(self._enqueue_op, feed_dict={
                self._src_tf: src, self._tgt_tf: tgt, self._label_tf: label
            })
        self._epoch_ended = True

    def start_epoch(self, n_threads=1):
        import threading
        assert self._session is not None
        if not self._loaded:
            self.load_data()
        for n in range(n_threads):
            t = threading.Thread(target=self.thread, args=(self._session,))
            t.daemon = True  # thread will close when parent quits
            t.start()

    def get_iterator(self, batch_size=32, maxlen=50):
        self._maxlen = maxlen
        self._batch_size = batch_size
        src, tgt, labels = self._queue.dequeue_many(batch_size)
        # Lookup
        src = self._src_table.lookup(tf.string_split(src))
        tgt = self._tgt_table.lookup(tf.string_split(tgt))
        # Padding
        src = tf.sparse_tensor_to_dense(src, default_value=self._NULL_ID)
        tgt = tf.sparse_tensor_to_dense(tgt, default_value=self._NULL_ID)
        return src, tgt, labels

if __name__ == '__main__':
    sys.path.append(".")
    from explab import *
    ss = tf.InteractiveSession()
    
    vocab_path = "{}/iwslt14.tokenized.de-en/iwslt14.en.bpe20k.vocab".format(MAINLINE_ROOT)
    vocab = Vocab(vocab_path)
    index_table = vocab.get_index_table()
    
    text_path = "{}/iwslt14.tokenized.de-en/valid.en.bpe20k".format(MAINLINE_ROOT)
    
    d = NCEDataset(text_path, text_path, vocab_path, vocab_path)
    d.start_epoch()
    src, tgt, labels = d.get_iterator()
    d.initialize(ss)
    ss.run(tf.global_variables_initializer())
    import pdb;pdb.set_trace()
