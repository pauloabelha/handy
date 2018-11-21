import torch.nn.functional as F
from network_blocks import *
import numpy as np
from torch.nn import init

class ReconstructNet(nn.Module):

    losses = {}
    num_input_channels = -1

    def __init__(self, params_dict):
        super(ReconstructNet, self).__init__()

        self.num_input_channels = params_dict['num_input_channels']

        self.num_layers_encoding = 5
        self.kernel_size = 3
        self.stride = 1
        self.out_channels = [64, 32, 16, 8, 4]
        self.paddings = [8, 0, 0, 0, 0]
        self.kernel_sizes = [self.kernel_size] * self.num_layers_encoding
        self.strides = [self.stride] * self.num_layers_encoding

        self.conv_sequence =\
            NetBlocksSequenceConvBatchRelu(num_blocks=self.num_layers_encoding,
                                           in_channels=self.num_input_channels,
                                           out_channels=self.out_channels,
                                           kernel_sizes=self.kernel_sizes,
                                           strides=self.strides,
                                           paddings=self.paddings)

        self.flatten = NetBlocksFlatten()
        kernel_size_1 = self.out_channels[-1]
        self.kernel_sizes = self.kernel_sizes[::-1]
        self.strides = self.strides[::-1]
        self.out_channels = self.out_channels[::-1][1:] + [self.num_input_channels]
        self.deconv_sequence = \
            NetBlocksSequenceDeconvBatchRelu(num_blocks=self.num_layers_encoding,
                                             in_channels=kernel_size_1,
                                             out_channels=self.out_channels,
                                             kernel_sizes=self.kernel_sizes,
                                             strides=self.strides)

        self.init_weights()

    def forward(self, x):
        x_initial_shape = x.shape
        x = self.conv_sequence(x)
        x_latent_shape = x.shape
        x = self.flatten(x)
        x = x.view(x_latent_shape)
        x = self.deconv_sequence(x)
        x = x[:, :, 0:x_initial_shape[2], 0:x_initial_shape[3]]
        return x

    def loss(self, output, label):
        self.losses['mse'] = F.mse_loss(output, label)
        return self.losses['mse']

    def init_weights(self):
        for module in self.conv_sequence.modules():
            if isinstance(module, nn.modules.conv.Conv2d):
                nn.init.xavier_uniform_(module.weight)
