import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMBaseline(nn.Module):

    def __init__(self, num_joints, num_actions, num_layers=3, layer_size=100, droput=0.2):
        assert(num_layers >= 3)
        super(LSTMBaseline, self).__init__()

        self.num_joints = num_joints
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.layer_size = layer_size

        self.lstms = [nn.LSTM(self.num_joints * 3, self.layer_size)]
        self.fusions = [nn.Linear(self.layer_size, self.layer_size)]
        for i in range(num_layers - 2):
            self.lstms.append(nn.LSTM(self.layer_size, self.layer_size))
            self.fusions.append(nn.Linear(self.layer_size, self.layer_size))
        self.lstms.append(nn.LSTM(self.layer_size, self.layer_size, dropout=droput))
        self.fusions.append(nn.Linear(self.layer_size, self.layer_size))

        self.funnel_in = nn.Linear(self.layer_size, self.num_actions)

        self.init_hidden_states()

    def init_hidden_states(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.hiddens = [0] * 3
        for i in range(self.num_layers):
            self.hiddens[i] = (torch.randn(1, 1, self.layer_size),
                       torch.randn(1, 1, self.layer_size))

    def forward(self, joints_sequence):
        out = joints_sequence
        for i in range(self.num_layers):
            out, self.hiddens[i] = self.lstms[i](out, self.hiddens[i])
            out = self.fusions[i](out)
        out = self.funnel_in(out)
        # because size of batch is 1
        out = out.view(out.shape[0], -1)
        out = F.log_softmax(out, dim=1)
        return out