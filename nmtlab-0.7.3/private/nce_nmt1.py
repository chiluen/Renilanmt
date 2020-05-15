#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os

sys.path.append(".")
from explab import *

from nmtlab import Vocab, MTDataset
from nce_dataset import NCEDataset
from nnlab import nn
import nnlab.nn.functional as F
import tensorflow as tf
import time
from tensorflow.python.client import device_lib
from tensorflow.contrib.nccl import reduce_sum, all_sum
ss = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

vocab_path = "{}/iwslt14.tokenized.de-en/iwslt14.en.bpe20k.vocab".format(MAINLINE_ROOT)
vocab = Vocab(vocab_path)

text_path = "{}/iwslt14.tokenized.de-en/train.en.bpe20k".format(MAINLINE_ROOT)

with tf.device("/cpu"):
    d = NCEDataset(text_path, text_path, vocab_path, vocab_path)

# Run
devices = device_lib.list_local_devices()
gpu_devices = [dev.name for dev in devices if dev.device_type == "GPU"]

loss_list = []
acc_list = []

multi_grads = []
multi_vars = []
for i, device in enumerate(gpu_devices):
    with tf.device(device):
        with tf.name_scope('%s_%d' % ("gpu", i)) as scope:
            var_scope = "vars_{}".format(i)
            with tf.variable_scope(var_scope):
                # Define
                src_emb_layer = nn.Embedding(vocab.size(), 256)
                tgt_emb_layer = nn.Embedding(vocab.size(), 256)
                with tf.variable_scope("enc"):
                    fw_cell = tf.nn.rnn_cell.LSTMCell(256)
                    bw_cell = tf.nn.rnn_cell.LSTMCell(256)
                with tf.variable_scope("dec"):
                    tgt_fw_cell = tf.nn.rnn_cell.LSTMCell(256)
                    tgt_bw_cell = tf.nn.rnn_cell.LSTMCell(256)
                dense1 = nn.Dense(256 * 2, 256)
                dense2 = nn.Dense(256 * 2, 256)
                # Run >>>
                src, tgt, label = d.get_iterator(batch_size=32)
                src_len = (tf.reduce_sum(tf.to_int32(src > 0), axis=1))
                tgt_len = (tf.reduce_sum(tf.to_int32(tgt > 0), axis=1))
                inp_emb = src_emb_layer.get(F.tensor_from_tf(src)).tf
                tgt_emb = tgt_emb_layer.get(F.tensor_from_tf(tgt)).tf
    
                (fw_states, bw_states), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inp_emb, sequence_length=src_len, swap_memory=True, dtype=tf.float32, scope="enc")
            
                (tgt_fw_states, tgt_bw_states), _ = tf.nn.bidirectional_dynamic_rnn(
                    tgt_fw_cell, tgt_bw_cell, tgt_emb,
                    sequence_length=tgt_len, swap_memory=True,
                    dtype=tf.float32)
    
                last_enc_fw_state = fw_states[:, -1]
                first_enc_bw_state = bw_states[:, 0]
                last_dec_fw_state = tgt_fw_states[:, -1]
                first_dec_bw_state = tgt_bw_states[:, 0]
                
                vec1 = dense1.forward(F.tensor_from_tf(tf.concat([last_enc_fw_state, first_enc_bw_state], axis=1)))
                vec2 = dense2.forward(F.tensor_from_tf(tf.concat([last_dec_fw_state, first_dec_bw_state], axis=1)))
                logit = tf.nn.sigmoid((vec1 * vec2).sum(axis=1))
                labels = tf.to_float(label)
                loss = - tf.log(tf.reduce_mean((logit * labels) + ((1 - labels) * (1 - logit))))
                acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int64(logit > 0.5), label)))
    
                loss_list.append(loss)
                acc_list.append(acc)
                
                tvars = tf.trainable_variables(scope=var_scope)
                grads = tf.gradients(loss, tvars)
                multi_vars.append(tvars)
                multi_grads.append(grads)
                

# Training
# tvars = tf.trainable_variables()
dev_n = float(len(gpu_devices))
multi_grads = zip(*[all_sum(gs) for gs in zip(*multi_grads)])
multi_grads = [[g / dev_n for g in gs] for gs in multi_grads]
train_op_list = []
for i, device in enumerate(gpu_devices):
    with tf.device(device):
        with tf.name_scope('%s_%d' % ("gpu", i)) as scope:
            var_scope = "vars_{}".format(i)
            with tf.variable_scope(var_scope):
                grads = multi_grads[i]
                tvars = multi_vars[i]
                optimizer = tf.train.AdamOptimizer(0.0001)
                train_op = optimizer.apply_gradients(zip(grads, tvars), name="train_op")
                train_op_list.append(train_op)

mean_loss = tf.reduce_mean(loss_list)
mean_acc = tf.reduce_mean(acc_list)

ss.run(tf.global_variables_initializer())
d.start_epoch(n_threads=1)
d.initialize(ss)

best_loss = 100000
best_counter = 0
done = False

curr_lr = 0.0001

saver = tf.train.Saver()
for epoch in range(10):
    
    count = 0
    start_time = time.time()
    tot_loss = 0.0
    tot_acc = 0.0
    while True:
        count += 1
        
        loss1, acc1, qsz1, _, _, _, _ = ss.run([
            mean_loss, mean_acc, d._size_op
        ] + train_op_list)
        
        # acc = 0
        tot_loss += loss1
        tot_acc += acc1
        sys.stdout.write('# loss={:.3f}, acc={:.2f}, qsz={}, bps={:.1f} prog={:.1f}%\r'.format(
            loss1, acc1, qsz1, count / (time.time() - start_time), d.progress() * 100
        ))
        sys.stdout.flush()
        
        if count % 1000 == 0:
            time_elapsed = time.time() - start_time
            avg_loss = tot_loss / count
            bps = count / (time.time() - start_time)
            print('{}: loss={:.3f}, acc={:.2f}, bps={:.1f}'.format(epoch, avg_loss, tot_acc / count, bps))
            
            # if avg_loss < best_loss * 0.99:
            #     best_loss = avg_loss
            #     saver.save(nn.runtime.get_session(), args.model_path.replace(".npz", ".ckpt"))
            #     print("saving model .. ")
            #     # if best_counter >= 100:
            #     #     best_counter = 0
            #     #     curr_lr /= 2
            #     #     if curr_lr < 1.e-5:
            #     #         print('learning rate too small - stopping now')
            #     #         done = True
            #     # sess.run(tf.assign(learning_rate, curr_lr))
            #
    # Print every epoch
    time_elapsed = time.time() - start_time
    bps = count / time_elapsed
    # wps = len(words) * bps
    print('%6d: loss = %6.6f, bps = %.1f, lr = %0.6f' % (
        epoch, tot_loss,
        bps,
        curr_lr))
    start_time = time.time()
