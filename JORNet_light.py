import hand_detection_net
from hand_detection_net import HALNet
import torch.nn as nn
from util import cudafy
import numpy as np

class JORNet_light(HALNet):
    innerprod1_size = 256 * 16 * 16
    crop_res = (128, 128)
    #innerprod1_size = 65536

    def map_out_to_loss(self, innerprod1_size):
        return cudafy(nn.Linear(in_features=innerprod1_size, out_features=200), self.use_cuda)

    def map_out_conv(self, in_channels):
        return cudafy(hand_detection_net.HALNetConvBlock(
            kernel_size=3, stride=1, filters=21, in_channels=in_channels, padding=1),
            self.use_cuda)

    def __init__(self, params_dict):
        super(JORNet_light, self).__init__(params_dict)

        if params_dict['hand_only']:
            self.num_joints = 60  # hand
        else:
            self.num_joints = 66  # hand and object
        self.out_poses1 = cudafy(
            nn.Linear(in_features=160000, out_features=1000), self.use_cuda)
        self.out_poses2 = cudafy(
            nn.Linear(in_features=1000, out_features=self.num_joints), self.use_cuda)

    def forward(self, x):
        _, _, _, conv4fout = self.forward_common_net(x)
        innerprod1_size = conv4fout.shape[1] * conv4fout.shape[2] * conv4fout.shape[3]
        out_poses = conv4fout.view(-1, innerprod1_size)
        out_poses = self.out_poses1(out_poses)
        out_poses = self.out_poses2(out_poses)
        return out_poses