import torch.nn.functional as F
from network_blocks import *
from torch.autograd import Variable

class ReconstructNet(nn.Module):

    losses = {}
    num_input_channels = -1

    def __init__(self, params_dict):
        super(ReconstructNet, self).__init__()

        self.num_input_channels = params_dict['num_input_channels']

        self.num_layers_encoding = 5
        self.kernel_size = 3
        self.stride = 1
        self.out_channel_first = 32
        self.out_channels = [0] * self.num_layers_encoding
        for i in range(self.num_layers_encoding):
            self.out_channels[i] = int(self.out_channel_first / 2**i)
        self.kernel_sizes = [self.kernel_size] * self.num_layers_encoding
        self.strides = [self.stride] * self.num_layers_encoding


        self.conv_sequence =\
            NetBlocksSequenceConvBatchRelu(num_blocks=self.num_layers_encoding,
                                           kernel_sizes=self.kernel_sizes,
                                           strides=self.strides,
                                           out_channels=self.out_channels,
                                           in_channels=self.num_input_channels)
        self.flatten = NetBlocksFlatten()
        self.kernel_sizes.append(self.kernel_size)
        self.strides.append(self.stride)
        self.out_channels = self.out_channels[::-1] + [self.num_input_channels]
        self.deconv_sequence = \
            NetBlocksSequenceDeconvBatchRelu(num_blocks=self.num_layers_encoding + 1,
                                             kernel_sizes=self.kernel_sizes,
                                             strides=self.strides,
                                             out_channels=self.out_channels,
                                             in_channels=self.out_channels[0])

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