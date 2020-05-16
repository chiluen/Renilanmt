#!/usr/bin/env python
# -*- coding:utf-8- -*-
#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pdb
import math
import os, sys
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append("./nmtlab")
from lanmt.lib_lanmt_model import LANMTModel
from nmtlab.models.transformer import TransformerEmbedding
from nmtlab.utils import MapDict, LazyTensorMap, TensorMap
from nmtlab.utils import OPTS
from lib_renilanmt_module import DisLSTM

class RENILANMTModel(LANMTModel):
    def __init__(self, dis_hidden_size=1024, dis_decoder_layers=6, dis_embed_size=1024,  **kwargs):
        self.dis_hidden_size = dis_hidden_size
        self.dis_decoder_layers = dis_decoder_layers
        self.dis_embed_size = dis_embed_size
        super(RENILANMTModel, self).__init__(**kwargs)
        self.dis = Discriminator(self._tgt_vocab_size, self.dis_embed_size, self.dis_hidden_size, self.dis_decoder_layers, batch_first=True, bidirectional=False)
        self.gen = LANMTModel_M(self.dis.get_fake_loss, **kwargs)
        self.dis.get_translate(self.gen.translate)
    def prepare(self):
        pass

    def forward(self,x,y, sampling=False, return_code=False):
        score_map = {}
        dis_val_map = self.dis(x,y)
        #---dis loss to be dis_loss---#
        dis_val_map['dis_loss'] = dis_val_map.pop('loss')
        gen_val_map = self.gen(x,y,sampling,return_code)
        score_map.update(dis_val_map)
        score_map.update(gen_val_map)
        return score_map


class LANMTModel_M(LANMTModel):
    def __init__(self, get_dis_loss, **kwargs):
        super(LANMTModel_M, self).__init__(**kwargs)
        self.get_dis_loss = get_dis_loss

    def compute_shard_loss(self, decoder_outputs, tgt_seq, tgt_mask, denominator=None, ignore_first_token=True,
                           backward=True):
        assert isinstance(decoder_outputs, TensorMap)
        is_grad_enabled = torch.is_grad_enabled()
        B = tgt_seq.shape[0]
        score_map = defaultdict(list)
        if denominator is None:
            if ignore_first_token:
                denom = tgt_mask[:, 1:].sum()
            else:
                denom = tgt_mask.sum()
        else:
            denom = denominator
        # Compute loss for each shard
        # The computation is performed on detached decoder states
        # Backpropagate the gradients to the deocder states
        OPTS.disable_backward_hooks = True
        for i in range(0, B, self._shard_size):
            j = i + self._shard_size
            decoder_outputs.select_batch(i, j, detach=True)
            logits = self.expand(decoder_outputs)
            reward = self.get_dis_loss(logits.argmax(-1)).detach()
            loss = self.compute_loss(logits, tgt_seq[i:j], tgt_mask[i:j], denominator=denom,
                                     ignore_first_token=ignore_first_token)
            word_acc = self.compute_word_accuracy(logits, tgt_seq[i:j], tgt_mask[i:j], denominator=denom,
                                                  ignore_first_token=ignore_first_token)
            score_map["loss"].append(loss)
            loss = loss + (-reward*1)
            score_map["word_acc"].append(word_acc)
            if i >= B - self._shard_size:
                # Enable the backward hooks to gather the gradients
                OPTS.disable_backward_hooks = False
            if is_grad_enabled:
                loss.backward()
        OPTS.disable_backward_hooks = False
        # Monitor scores
        monitors = {}
        for k in score_map:
            val = sum(score_map[k])
            self.monitor(k, val)
            monitors[k] = val
        # Backpropagate the gradients to all the parameters
        if is_grad_enabled:
            detached_items = list(decoder_outputs.get_detached_items().values())
            state_tensors = [x[1] for x in detached_items]
            grads = [x[0].grad for x in detached_items]
            if backward:
                torch.autograd.backward(state_tensors, grads)
        else:
            state_tensors, grads = None, None
        return monitors, state_tensors, grads

class Discriminator(nn.Module):
    def __init__(self, tgt_vocab_size, input_size, hidden_size, n_layers, batch_first=True, bidirectional=False):
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        super().__init__()
        #self.embed_layer = TransformerEmbedding(self.tgt_vocab_size, self.hidden_size)
        self.embed_layer = nn.Embedding(self.tgt_vocab_size, self.hidden_size)
        self.dis = DisLSTM(self.input_size, self.hidden_size, self.n_layers, self.batch_first, self.bidirectional)
        self.translate = None
        self.criterion = nn.BCELoss()
    def prepare(self):
        pass
    def forward(self, x, y):
        score_map = {'loss': 0}
        #---real----#
        output = self.dis(self.embed_layer(y)).view(-1)
        loss = self.compute_wgan_loss(output,1)
        score_map.update(loss)
        #---fake----#
        assert self.translate is not None
        fake, _, _ = self.translate(x)
        fake = fake.detach()
        output = self.dis(self.embed_layer(fake)).view(-1)
        loss = self.compute_wgan_loss(output,0)
        score_map.update(loss)
        #---WGAN---#
        score_map['loss'] = score_map['fake_loss'] - score_map['real_loss']
        #---DCGAN---#
        #score_map['loss'] = score_map['fake_loss'] + score_map['real_loss']
        is_grad_enabled = torch.is_grad_enabled()
        if is_grad_enabled:
            score_map['loss'].backward()
        return score_map

    def get_fake_loss(self,x):
        output = self.dis(self.embed_layer(x)).view(-1)
        output = output.mean()
        return output

    def compute_loss(self, x, label):
        if label > 0 :
            loss_name = "real_loss"
        else:
            loss_name = "fake_loss"
        #B = [*x.shape]
        B = x.shape
        x_label = torch.full(B , label)
        if torch.cuda.is_available():
            x_label = x_label.cuda()
            x = x.cuda()
        loss = self.criterion(x, x_label)
        scores = {
            loss_name: loss
        }
        return scores

    def compute_wgan_loss(self, x, label):
        loss={}
        x=x.mean()
        if label == 1:
            loss={'real_loss':x}
        else:
            loss={'fake_loss':x}
        return loss
    def get_translate(self,gen_translate):
        self.translate = gen_translate
        return True
    def to_float(self, x):
        #if self._fp16:
        #    return x.half()
        #else:
        return x.float()
