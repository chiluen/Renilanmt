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
import horovod.tensorflow as hvd

# Initialize Horovod
hvd.init()


vocab_path_de = "{}/iwslt14.tokenized.de-en/iwslt14.de.bpe20k.vocab".format(MAINLINE_ROOT)
vocab_de = Vocab(vocab_path_de)
vocab_path_en = "{}/iwslt14.tokenized.de-en/iwslt14.en.bpe20k.vocab".format(MAINLINE_ROOT)
vocab_en = Vocab(vocab_path_en)

text_path_de = "{}/iwslt14.tokenized.de-en/train.de.bpe20k".format(MAINLINE_ROOT)
text_path_en = "{}/iwslt14.tokenized.de-en/train.en.bpe20k".format(MAINLINE_ROOT)
valid_text_path_de = "{}/iwslt14.tokenized.de-en/valid.de.bpe20k".format(MAINLINE_ROOT)
valid_text_path_en = "{}/iwslt14.tokenized.de-en/valid.en.bpe20k".format(MAINLINE_ROOT)

with tf.device("/cpu"):
    print("hvd local_rank={} size={}".format(hvd.local_rank(), hvd.size()))
    d = NCEDataset(text_path_de, text_path_en, vocab_path_de, vocab_path_en, gpu_id=hvd.local_rank(), gpu_num=hvd.size())
    valid_d = NCEDataset(valid_text_path_de, valid_text_path_en, vocab_path_de, vocab_path_en, max_lines=32 * 50)

# Define
src_emb_layer = nn.Embedding(vocab_de.size(), 256)
tgt_emb_layer = nn.Embedding(vocab_en.size(), 256)
with tf.variable_scope("enc"):
    fw_cell = tf.nn.rnn_cell.LSTMCell(256)
    bw_cell = tf.nn.rnn_cell.LSTMCell(256)
with tf.variable_scope("dec"):
    tgt_fw_cell = tf.nn.rnn_cell.LSTMCell(256)
    tgt_bw_cell = tf.nn.rnn_cell.LSTMCell(256)
dense1 = nn.Dense(256 * 2, 256)
dense2 = nn.Dense(256 * 2, 256)
dense3 = nn.Dense(256 * 2, 256)
dense4 = nn.Dense(256, 1)

# Run >>>
def build_graph(src, tgt, label):
    src_len = (tf.reduce_sum(tf.to_int32(src > 0), axis=1))
    tgt_len = (tf.reduce_sum(tf.to_int32(tgt > 0), axis=1))
    inp_emb = src_emb_layer.get(F.tensor_from_tf(src)).tf
    tgt_emb = tgt_emb_layer.get(F.tensor_from_tf(tgt)).tf
    
    (fw_states, bw_states), _ = tf.nn.bidirectional_dynamic_rnn(
        fw_cell, bw_cell, inp_emb,
        sequence_length=src_len, swap_memory=True, dtype=tf.float32, scope="enc")
    
    (tgt_fw_states, tgt_bw_states), _ = tf.nn.bidirectional_dynamic_rnn(
        tgt_fw_cell, tgt_bw_cell, tgt_emb,
        sequence_length=tgt_len, swap_memory=True,
        dtype=tf.float32)
    
    last_enc_fw_state = fw_states[:, -1]
    first_enc_bw_state = bw_states[:, 0]
    last_dec_fw_state = tgt_fw_states[:, -1]
    first_dec_bw_state = tgt_bw_states[:, 0]
    
    vec1 = dense1.forward(F.tensor_from_tf(tf.concat([last_enc_fw_state, first_enc_bw_state], axis=1))).tf
    vec2 = dense2.forward(F.tensor_from_tf(tf.concat([last_dec_fw_state, first_dec_bw_state], axis=1))).tf
    vec3 = dense3.forward(F.tensor_from_tf(tf.concat([vec1, vec2], axis=1))).tf
    pre_logit = dense4.forward(vec3).tf
    logit = tf.reshape(pre_logit, [tf.shape(pre_logit)[0]])
    logit = tf.nn.sigmoid(logit)
    labels = tf.to_float(label)
    loss = - tf.log(tf.reduce_mean((logit * labels) + ((1 - labels) * (1 - logit))))
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int64(logit > 0.5), label)))
    return loss, acc

loss, acc = build_graph(*d.get_iterator(batch_size=32))
valid_loss, valid_acc = build_graph(*valid_d.get_iterator(batch_size=32))
opt = tf.train.AdamOptimizer(0.0001)
opt = hvd.DistributedOptimizer(opt)
train_op = opt.minimize(loss)

# Pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())
config.allow_soft_placement = True
my_rank = hvd.local_rank()

ss = tf.Session(config=config)

ss.run(tf.global_variables_initializer())
d.initialize(ss)
valid_d.initialize(ss)

ss.run(hvd.broadcast_global_variables(0))

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
    d.start_epoch()
    while not d.epoch_ended():
        count += 1
        
        loss1, acc1, qsz1, _= ss.run([
            loss, acc, d._size_op, train_op
        ])
        
        # acc = 0
        tot_loss += loss1
        tot_acc += acc1
        sys.stdout.write('[{}] loss={:.3f}, acc={:.2f}, qsz={}, bps={:.1f} prog={:.1f}%            \r'.format(
            my_rank,
            loss1, acc1, qsz1, count / (time.time() - start_time), d.progress() * 100
        ))
        sys.stdout.flush()
        
        if count % 500 == 0 and my_rank == 0:
            valid_loss1 = 0.
            valid_acc1 = 0.
            valid_cnt1 = 0
            valid_d.start_epoch()
            while not valid_d.epoch_ended():
                loss1, acc1 = ss.run([
                    valid_loss, valid_acc
                ])
                valid_loss1 += loss1
                valid_acc1 += acc1
                valid_cnt1 += 1
            mean_loss = valid_loss1 / valid_cnt1
            mean_acc = valid_acc1 / valid_cnt1
            print('[epoch:{}] checkpoint loss={:.3f}, acc={:.2f}             '.format(epoch, mean_loss, mean_acc))
            
            # if avg_loss < best_loss * 0.99:
            #     best_loss = avg_loss
            #     saver.save(nn.runtime.get_session(), args.model_path.replace(".npz", ".ckpt"))
            #     print("saving model .. ")
                # if best_counter >= 100:
                #     best_counter = 0
                #     curr_lr /= 2
                #     if curr_lr < 1.e-5:
                #         print('learning rate too small - stopping now')
                #         done = True
                # sess.run(tf.assign(learning_rate, curr_lr))
    
    print("# End of epoch {}".format(epoch))
