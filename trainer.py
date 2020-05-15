#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

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

        self.dis_trainer = TrainerKit1(model.dis, dataset, dis_optimizer,
                                      scheduler=dis_scheduler, multigpu=multigpu, using_horovod=using_horovod)
        self.gen_trainer = TrainerKit1(model.gen, dataset, optimizer,
                                      scheduler=scheduler, multigpu=multigpu, using_horovod=using_horovod)

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
        self.dis_trainer.train(batch)
        #if self._global_step % OPTS.dis_1GXD == 0 :
        #    self.gen_trainer.train(batch)
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
        sys.stdout.write("[epoch {}|{}%] loss={:.2f} | dis_loss={:.5f} |  fake_loss={:.5f} | real_loss={:.5f} | {:.1f} {}/s   \r".format(
            self._current_epoch + 1, progress, val_map["loss"], val_map["dis_loss"], val_map["fake_loss"], val_map["real_loss"], speed, unit))
        sys.stdout.flush()

class TrainerKit1(TrainerKit):
    def train(self, batch):
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
            # if self._multigpu and self._horovod:
            #     self._optimizer.synchronize()
            #torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_norm)
            for p in self._model.parameters():
                p.data.clamp_(-self._clip_norm, self._clip_norm)

        self._optimizer.step()
        #self.print_progress(val_map)
        self.record_train_scores(val_map)
        self._global_step += 1
        return val_map         








