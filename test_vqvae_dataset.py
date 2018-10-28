import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import argparse
import fpa_dataset
from lstm_baseline import LSTMBaseline
from util import myprint
from hand_detection_net import HALNet
import losses as my_losses
import trainer

parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
parser.add_argument('-r', dest='dataset_root_folder', required=True, help='Root folder for dataset')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
parser.add_argument('--split-filename', default='', help='Dataset split filename')

args = parser.parse_args()

transform_color = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.5, 0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5, 0.5))])

transform_depth = transforms.Compose([transforms.ToTensor()])

train_loader = fpa_dataset.DataLoaderReconstruction(root_folder=args.dataset_root_folder,
                                              type='train',
                                              input_type="rgbd",
                                              transform_color=transform_color,
                                              transform_depth=transform_depth,
                                              batch_size=args.batch_size,
                                              split_filename=args.split_filename,)

print('Length of dataset: {}'.format(len(train_loader.dataset)))