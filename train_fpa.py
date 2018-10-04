import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import fpa_dataset
from lstm_baseline import LSTMBaseline
from util import myprint


parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
parser.add_argument('-r', dest='dataset_root_folder', required=True, help='Root folder for dataset')
parser.add_argument('-split-filename', default='', help='Dataset split filename')
parser.add_argument('-e', dest='num_epochs', type=int, required=True,
                    help='Total number of epochs to train')
parser.add_argument('-f', dest='checkpoint_filepath', default='lstm_baseline.pth.tar',
                    help='Checkpoint file path')
parser.add_argument('--use-cuda', dest='use_cuda', action='store_true', default=False,
                    help='Whether to use cuda for training')
parser.add_argument('-o', dest='log_filepath', default='log_lstm_baseline.txt',
                    help='Output file for logging')
parser.add_argument('-l', dest='epoch_log', type=int, default=10,
                    help='Total number of epochs to train')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
parser.add_argument('--gt_folder', dest='gt_folder', default='Hand_pose_annotation_v1',
                    help='Folder with Subject groundtruth')
parser.add_argument('--num_joints', type=int, dest='num_joints', default=21, help='Number of joints')

args = parser.parse_args()


train_loader = fpa_dataset.DataLoaderTracking(root_folder=args.dataset_root_folder,
                                      type='train', transform=None,
                                      batch_size=args.batch_size,
                                      split_filename=args.split_filename)
'''
lstm_baseline = LSTMBaseline(num_joints=21,
                             num_actions=num_actions,
                             use_cuda=args.use_cuda)
lstm_baseline.train()
if args.use_cuda:
    lstm_baseline.cuda()

optimizer = optim.Adadelta(lstm_baseline.parameters(), rho=0.9,
                           weight_decay=0.005,
                           lr=0.05)
'''

losses = []
for i in range(len(train_loader.dataset)):
    losses.append([])

for epoch_idx in range(args.num_epochs - 1):
    epoch = epoch_idx + 1
    myprint('---------------- Epoch {}/{} ----------------'.format(epoch, args.num_epochs), args.log_filepath)
    for batch_idx, train_tuple in enumerate(train_loader):
        print(batch_idx)
        # zero out torch gradients
        #optimizer.zero_grad()

        # clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        #lstm_baseline.init_hidden_states()

