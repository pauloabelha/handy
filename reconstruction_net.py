import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from network_blocks import *





class ReconstructNet(nn.Module):

    num_input_channels = -1

    def __init__(self, params_dict):
        super(ReconstructNet, self).__init__()

        self.num_input_channels = params_dict['num_input_channels']

        self.conv_sequence =\
            NetBlocksSequenceConvBatchRelu(3, kernel_sizes=[4, 4, 4, 4],
                                           strides=[1, 1, 1, 2],
                                           filters=[64, 32, 16, 8],
                                           in_channels=self.num_input_channels)
        self.flatten = NetBlocksFlatten()

    def forward(self, x):
        x = self.conv_sequence(x)
        x = self.flatten(x)
        return x
        