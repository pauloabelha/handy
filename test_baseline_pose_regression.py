import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import argparse
import fpa_dataset
from lstm_baseline import LSTMBaseline
from util import myprint
from JORNet_light import JORNet_light
import losses as my_losses
import trainer
import numpy as np
import visualize as vis


parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
parser.add_argument('-r', dest='dataset_root_folder', required=True, help='Root folder for dataset')
parser.add_argument('-c', dest='checkpoint_filename', required=True, help='Checkpoint filename')
parser.add_argument('--split-filename', default='', help='Dataset split filename')
parser.add_argument('--use-cuda', dest='use_cuda', action='store_true', default=False,
                    help='Whether to use cuda')

args = parser.parse_args()
args.batch_size = 1
args.log_interval = 500

transform_color = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.5, 0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5, 0.5))])

transform_depth = transforms.Compose([transforms.ToTensor()])

test_loader = fpa_dataset.DataLoaderPoseRegression(root_folder=args.dataset_root_folder,
                                              type='train',
                                              input_type="depth",
                                              transform_color=transform_color,
                                              transform_depth=transform_depth,
                                              batch_size=args.batch_size,
                                              split_filename=args.split_filename,)

print('Length of dataset: {}'.format(len(test_loader.dataset)))

model_params_dict = {
    'joint_ixs': range(2)
}

model, _, _, _ = trainer.load_checkpoint(args.checkpoint_filename, JORNet_light, use_cuda=True)
if args.use_cuda:
    model.cuda()

loss_func = my_losses.cross_entropy_loss_p_logq
losses = []
for i in range(len(test_loader.dataset)):
    losses.append([])

train_vars = {
    'iter_size': 1,
    'total_loss': 0,
    'verbose': True,
    'checkpoint_filenamebase': 'checkpoint_test_pose',
    'checkpoint_filename': 'checkpoint_test_pose.pth.tar',
    'tot_iter': len(test_loader),
    'num_batches': len(test_loader),
    'curr_iter': 0,
    'batch_size': args.batch_size,
    'log_interval': args.log_interval,
    'best_loss': 1e10,
    'losses': [],
    'output_filepath': 'log.txt',
    'tot_epoch': 1,
}

epoch = 1
train_vars['curr_epoch_iter'] = epoch
continue_batch_end_ix = -1
for batch_idx, (depth_img_torch, hand_obj_pose, hand_root) in enumerate(test_loader):
    if batch_idx < continue_batch_end_ix:
        print('Continuing... {}/{}'.format(batch_idx, continue_batch_end_ix))
        continue
    train_vars['batch_idx'] = batch_idx
    train_vars['curr_iter'] = batch_idx + 1
    if args.use_cuda:
        depth_img_torch = depth_img_torch.cuda()
        hand_obj_pose = hand_obj_pose.cuda()

    output = model(depth_img_torch)

    hand_pose_abs = np.zeros((21, 3))
    output_numpy = output.detach().cpu().numpy()
    hand_pose_rel = output_numpy[0, 0:60].reshape((20, 3))
    hand_pose_abs[0, :] = hand_root.detach().cpu().numpy()
    hand_pose_abs[1:, :] = hand_pose_abs[0, :] + hand_pose_rel

    #vis.plot_3D_joints(hand_pose_abs)
    #vis.show()

    loss = my_losses.euclidean_loss(output, hand_obj_pose)
    train_vars['total_loss'] = loss.item()
    train_vars['losses'].append(train_vars['total_loss'])
    if train_vars['total_loss'] < train_vars['best_loss']:
        train_vars['best_loss'] = train_vars['total_loss']

    if batch_idx > 0 and batch_idx % args.log_interval == 0:
        trainer.print_log_info(model, None, epoch,  train_vars)
trainer.print_log_info(model, None, epoch, train_vars)
