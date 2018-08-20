import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def cudafy(object, use_cuda):
    if use_cuda:
        return object.cuda()
    else:
        return object

class LSTMBaseline(nn.Module):

    def __init__(self, num_joints, num_actions, use_cuda=False):
        super(LSTMBaseline, self).__init__()

        self.num_joints = num_joints
        self.num_actions = num_actions
        self.use_cuda = use_cuda

        num_dims = self.num_joints * 3
        num_internal_dims = 100
        self.hidden_dim = num_internal_dims

        self.lstm1 = cudafy(nn.LSTM(num_dims, num_internal_dims), self.use_cuda)
        self.fusion1 = cudafy(nn.Linear(num_internal_dims, num_internal_dims), self.use_cuda)

        self.lstm2 = cudafy(nn.LSTM(num_internal_dims, num_internal_dims), self.use_cuda)
        self.fusion2 = cudafy(nn.Linear(num_internal_dims, num_internal_dims), self.use_cuda)

        self.lstm3 = cudafy(nn.LSTM(num_internal_dims, num_internal_dims, dropout=0.2), self.use_cuda)
        self.fusion3 = cudafy(nn.Linear(num_internal_dims, num_internal_dims), self.use_cuda)

        self.funnel_in = cudafy(nn.Linear(num_internal_dims, num_actions), self.use_cuda)

        self.init_hidden_states()

    def get_random_tensor(self, dims):
        ret = torch.randn(dims)
        if self.use_cuda:
            ret = ret.cuda()
        return ret

    def init_hidden_states(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.hidden1 = (self.get_random_tensor((1, 1, self.hidden_dim)),
                        self.get_random_tensor((1, 1, self.hidden_dim)))
        self.hidden2 = (self.get_random_tensor((1, 1, self.hidden_dim)),
                        self.get_random_tensor((1, 1, self.hidden_dim)))
        self.hidden3 = (self.get_random_tensor((1, 1, self.hidden_dim)),
                        self.get_random_tensor((1, 1, self.hidden_dim)))

    def forward(self, joints_sequence):
        out, hidden1 = self.lstm1(joints_sequence, self.hidden1)
        out = self.fusion1(out)
        out, self.hidden2 = self.lstm2(out, self.hidden2)
        out = self.fusion2(out)
        out, self.hidden3 = self.lstm3(out, self.hidden3)
        out = self.fusion3(out)
        out = self.funnel_in(out)
        # because size of batch is 1
        out = out.view(out.shape[0], -1)
        out = F.log_softmax(out, dim=1)
        return out