#!/usr/bin/env python 
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import pdb



class DisLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers,batch_first=True,bidirectional=False, **kwargs):
        super(DisLSTM, self).__init__(**kwargs)
        self.input = input_size
        self.hidden_dim = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, bidirectional = False)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size/4))
        self.fc2 = nn.Linear(int(hidden_size/4), 1)
        self.bn1 = nn.BatchNorm1d(int(hidden_size/4))
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, (hidden,cn) = self.rnn(x, hidden)
        out = out[:,-1,:]
        out = self.fc1(out)
        out = self.relu(out)
        #out = self.bn1(out)
        out = self.fc2(out)
        #out = torch.sigmoid(out) wgan
        return out

    def init_hidden(self, batch_size):
        hidden0 = torch.randn(self.n_layers, batch_size, self.hidden_dim)
        c0 = torch.randn(self.n_layers, batch_size, self.hidden_dim)
        if torch.cuda.is_available():
            hidden0 = hidden0.cuda()
            c0 = c0.cuda()
        return (hidden0,c0)


















