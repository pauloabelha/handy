import torch.nn.functional as F
from network_blocks import *
from torch.autograd import Variable

class ReconstructNet(nn.Module):

    losses = {}
    num_input_channels = -1

    def __init__(self, params_dict):
        super(ReconstructNet, self).__init__()

        self.num_input_channels = params_dict['num_input_channels']

        self.conv_sequence =\
            NetBlocksSequenceConvBatchRelu(4, kernel_sizes=[4, 4, 4, 4],
                                           strides=[1, 1, 1, 1],
                                           out_channels=[32, 16, 8, 4],
                                           in_channels=self.num_input_channels)
        self.flatten = NetBlocksFlatten()
        self.deconv_sequence = \
            NetBlocksSequenceDeconvBatchRelu(5, kernel_sizes=[4, 4, 4, 4, 4],
                                           strides=[1, 1, 1, 1, 1],
                                           out_channels=[4, 8, 16, 32, 3],
                                           in_channels=4,
                                             paddings=[0, 0, 11, 0, 0])

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