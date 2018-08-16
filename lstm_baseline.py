import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMBaseline(nn.Module):

    def __init__(self, num_joints, num_actions):
        super(LSTMBaseline, self).__init__()

        self.num_joints = num_joints
        num_dims = self.num_joints * 3
        self.hidden_dim = num_dims


        self.lstm1 = nn.LSTM(num_dims, num_dims)
        self.fusion1 = nn.Linear(num_dims, num_dims)

        self.lstm2 = nn.LSTM(num_dims, num_dims)
        self.fusion2 = nn.Linear(num_dims, num_dims)

        self.lstm3 = nn.LSTM(num_dims, num_dims)
        self.fusion3 = nn.Linear(num_dims, num_dims)

        self.funnel_in = nn.Linear(num_dims, num_actions)
        self.softmax_final = nn.LogSoftmax(dim=-1)

        self.init_hidden_states()

    def init_hidden_states(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.hidden1 = (torch.zeros(1, 1, self.hidden_dim),
                   torch.zeros(1, 1, self.hidden_dim))
        self.hidden2 = (torch.zeros(1, 1, self.hidden_dim),
                   torch.zeros(1, 1, self.hidden_dim))
        self.hidden3 = (torch.zeros(1, 1, self.hidden_dim),
                   torch.zeros(1, 1, self.hidden_dim))

    def forward(self, joints_sequence):
        out, self.hidden1 = self.lstm1(joints_sequence, self.hidden1)
        out = self.fusion1(out)
        out, self.hidden2 = self.lstm2(out, self.hidden2)
        out = self.fusion2(out)
        out, self.hidden3 = self.lstm3(out, self.hidden3)
        out = self.fusion3(out)
        out = self.funnel_in(out)
        # because size of batch is 1
        out = out.view(out.shape[0], -1)
        out = F.log_softmax(out)
        return out