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

        self.conv1 = NetBlockConvBatchRelu(kernel_size=4, stride=1, filters=64,
                                           in_channels=self.num_input_channels)

    def forward(self, x):
        x = self.conv1(x)

        return x
        