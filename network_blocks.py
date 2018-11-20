import torch.nn as nn
import torch

def NetBlockConvBatchRelu(kernel_size, stride, filters, in_channels, padding=0):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=filters),
            nn.ReLU()
        )

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
