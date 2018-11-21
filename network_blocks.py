import torch.nn as nn
import torch

def NetBlockConvBatchRelu(in_channels, out_channels, kernel_size, stride,
                          padding=0):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

def NetBlocksSequenceConvBatchRelu(num_blocks, in_channels, out_channels,
                                   kernel_sizes, strides, paddings=0):
    if paddings == 0:
        paddings = [0] * num_blocks
    conv_sequence = []
    conv_sequence.append(NetBlockConvBatchRelu(in_channels, out_channels[0],
                                               kernel_sizes[0], strides[0],
                                               padding=paddings[0]))
    for i in range(num_blocks - 1):
        conv_sequence.append(NetBlockConvBatchRelu(out_channels[i], out_channels[i+1],
                                                   kernel_sizes[i+1], strides[i+1],
                                                   paddings[i+1]))
    conv_sequence = nn.Sequential(*conv_sequence)
    return conv_sequence

def NetBlocksDeconvBatchRelu(in_channels, out_channels, kernel_size, stride,
                          padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU()
    )

def NetBlocksSequenceDeconvBatchRelu(num_blocks, in_channels, out_channels,
                                   kernel_sizes, strides, paddings=0):
    if paddings == 0:
        paddings = [0] * num_blocks
    deconv_sequence = []
    deconv_sequence.append(NetBlocksDeconvBatchRelu(in_channels, out_channels[0],
                                               kernel_sizes[0], strides[0],
                                               padding=paddings[0]))
    for i in range(num_blocks-1):
        deconv_sequence.append(NetBlockConvBatchRelu(out_channels[i], out_channels[i+1],
                                                   kernel_sizes[i+1], strides[i+1],
                                                   paddings[i+1]))
    deconv_sequence = nn.Sequential(*deconv_sequence)
    return deconv_sequence

class NetBlocksFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class NetBlockSoftmaxLogProbability2D(torch.nn.Module):
    def __init__(self):
        super(NetBlockSoftmaxLogProbability2D, self).__init__()

    def forward(self, x):
        orig_shape = x.data.shape
        seq_x = []
        for channel_ix in range(orig_shape[1]):
            softmax_ = F.softmax(x[:, channel_ix, :, :].contiguous()
                                 .view((orig_shape[0], orig_shape[2] * orig_shape[3])), dim=1)\
                .view((orig_shape[0], orig_shape[2], orig_shape[3]))
            seq_x.append(softmax_.log())
        x = torch.stack(seq_x, dim=1)
        return x
