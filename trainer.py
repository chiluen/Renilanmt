#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from nmtlab.trainers.base import TrainerKit
from six.moves import xrange
import sys
import time
import pdb
MAX_EPOCH = 10000


from nmtlab.utils import OPTS



class RENILANMTTrainer(TrainerKit):
    def __init__(self, model, dataset, optimizer, dis_optimizer, dis_scheduler=None, 
             scheduler=None, multigpu=False, using_horovod=True):
        self.dis_trainer = TrainerKit1( model.dis, dataset, dis_optimizer, dis_scheduler, multigpu, using_horovod)
        self.gen_trainer = TrainerKit1( model.gen, dataset, optimizer, scheduler, multigpu, using_horovod)

        super().__init__(model,dataset,optimizer,scheduler,multigpu,using_horovod)
    def run(self):
        """Run the training from begining to end.
        """
        self.valid(force=True)
        self._model.train(True)
        for epoch in xrange(MAX_EPOCH):
            self.begin_epoch(epoch)
            for step, batch in enumerate(self._dataset.train_set()):
                self.begin_step(step)
                self.train(batch)
                self.valid()
            self.end_epoch()
            # Check if finished
            if self.is_finished():
                self.save()
                break

    def train(self, batch):
        dis_clip_ck = True 
        self.dis_trainer.train(batch, dis_clip_ck)
        if not OPTS.dis_train_only :
            if self._global_step % OPTS.dis_1GXD == 0 :
                dis_clip_ck = False
                self.gen_trainer.train(batch, dis_clip_ck)
        vars = self.extract_vars(batch)
        val_map = self._model(*vars)
        self.print_progress(val_map)
        self.record_train_scores(val_map)
        self._global_step += 1
        return val_map         

    def print_progress(self, val_map):
        progress = int(float(self._current_step) / self._n_train_batch * 100)
        speed = float(self._current_step * self._batch_size) / (time.time() - self._begin_time) * self._n_devices
        unit = "token" if self._dataset.batch_type() == "token" else "batch"
        sys.stdout.write("[epoch {}|{}%] loss={:.2f} | dis_loss={:.7f} |  fake_loss={:.7f} | real_loss={:.7f} | {:.1f} {}/s   \r".format(
            self._current_epoch + 1, progress, val_map["loss"], val_map["dis_loss"], val_map["fake_loss"], val_map["real_loss"], speed, unit))
        sys.stdout.flush()

class TrainerKit1(TrainerKit):
    def train(self, batch, dis_clip_ck):
        """Run one forward and backward step with given batch.
        """
        self._optimizer.zero_grad()
        vars = self.extract_vars(batch)
        val_map = self._model(*vars)
        if self._multigpu and not self._horovod:
            for k, v in val_map.items():
                val_map[k] = v.mean()
        if not OPTS.shard:
            val_map["loss"].backward()
        if self._clip_norm > 0:
            self.clip_norm(dis_clip_ck)
            #if self._multigpu and self._horovod:
            #     self._optimizer.synchronize()
            #torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_norm)

        if self._global_step == 100 or self._global_step == 500 or self._global_step % 1000 == 0:
            self.plot_grad_flow(self._model.dis.named_parameters())
        self._optimizer.step()
        #self.print_progress(val_map)
        self.record_train_scores(val_map)
        self._global_step += 1
        return val_map         
    
    def clip_norm(self,dis_clip_ck=False):
        if dis_clip_ck :
            for p in self._model.dis.parameters():
                p.data.clamp_(-self._clip_norm, self._clip_norm)
        else:
            if self._multigpu and self._horovod:
                 self._optimizer.synchronize()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_norm)
    
    def plot_grad_flow(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
    
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        plt.clf()
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=-0.5, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        my_file = 'gradient' + str(self._global_step)+'.png'
        plt.savefig('./log/'+ my_file)





