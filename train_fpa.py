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
parser.add_argument('--split-filename', default='', help='Dataset split filename')
parser.add_argument('-e', dest='num_epochs', type=int, required=True,
                    help='Total number of epochs to train')
parser.add_argument('--log-interval', dest='log_interval', type=int, default=100,
                    help='Intervalwith which to log')
parser.add_argument('-f', dest='checkpoint_filepath', default='lstm_baseline.pth.tar',
                    help='Checkpoint file path')
parser.add_argument('--use-cuda', dest='use_cuda', action='store_true', default=False,
                    help='Whether to use cuda for training')
parser.add_argument('-o', dest='log_filepath', default='log_lstm_baseline.txt',
                    help='Output file for logging')
parser.add_argument('-l', dest='epoch_log', type=int, default=10,
                    help='Total number of epochs to train')
parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
                    help='Momentum for AdaDelta')
parser.add_argument('--wieght-decay', dest='weight_decay', type=float, default=0.005,
                    help='Weight decay for AdaDelta')
parser.add_argument('--lr', dest='lr', type=float, default=0.05,
                    help='Learning rate for AdaDelta')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
parser.add_argument('--gt_folder', dest='gt_folder', default='Hand_pose_annotation_v1',
                    help='Folder with Subject groundtruth')
parser.add_argument('--num_joints', type=int, dest='num_joints', default=21, help='Number of joints')

args = parser.parse_args()

transform_color = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.5, 0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5, 0.5))])

transform_depth = transforms.Compose([transforms.ToTensor()])

train_loader = fpa_dataset.DataLoaderTracking(root_folder=args.dataset_root_folder,
                                      type='train', transform_color=transform_color,
                                              transform_depth=transform_depth,
                                      batch_size=args.batch_size,
                                      split_filename=args.split_filename,)

print('Length of dataset: {}'.format(len(train_loader.dataset)))

model_params_dict = {
    'joint_ixs': range(2)
}

model = HALNet(model_params_dict)
if args.use_cuda:
    model = model.cuda()
model.train()

optimizer = optim.Adadelta(model.parameters(),
                           rho=args.momentum,
                           weight_decay=args.weight_decay,
                           lr=args.lr)
loss_func = my_losses.cross_entropy_loss_p_logq
losses = []
for i in range(len(train_loader.dataset)):
    losses.append([])

train_vars = {
    'iter_size': 1,
    'total_loss': 0,
    'verbose': True,
    'checkpoint_filenamebase': 'checkpoint',
    'checkpoint_filename': 'checkpoint.pth.tar',
    'tot_iter': len(train_loader),
    'num_batches': len(train_loader),
    'curr_iter': 0,
    'batch_size': args.batch_size,
    'log_interval': args.log_interval,
    'best_loss': 1e10,
    'losses': [],
    'output_filepath': 'log.txt',
    'tot_epoch': args.num_epochs,
}

for epoch_idx in range(args.num_epochs - 1):
    epoch = epoch_idx + 1
    train_vars['curr_epoch_iter'] = epoch
    continue_batch_end_ix = -1
    for batch_idx, (data, label_heatmaps) in enumerate(train_loader):
        if batch_idx < continue_batch_end_ix:
            print('Continuing... {}/{}'.format(batch_idx, continue_batch_end_ix))
            continue
        train_vars['batch_idx'] = batch_idx
        train_vars['curr_iter'] = batch_idx + 1
        if args.use_cuda:
            data = data.cuda()
            label_heatmaps = label_heatmaps.cuda()
        # zero out torch gradients
        optimizer.zero_grad()

        # clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        output = model(data)

        loss = my_losses.calculate_loss_HALNet(loss_func,
                                               output, label_heatmaps, model.joint_ixs, model.WEIGHT_LOSS_INTERMED1,
                                               model.WEIGHT_LOSS_INTERMED2, model.WEIGHT_LOSS_INTERMED3,
                                               model.WEIGHT_LOSS_MAIN, train_vars['iter_size'])
        loss.backward()
        train_vars['total_loss'] = loss.item()
        train_vars['losses'].append(train_vars['total_loss'])
        if train_vars['total_loss'] < train_vars['best_loss']:
            train_vars['best_loss'] = train_vars['total_loss']

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            trainer.print_log_info(model, optimizer, epoch,  train_vars)
