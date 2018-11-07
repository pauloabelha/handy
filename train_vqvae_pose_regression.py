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

# train pose regression using images resoncstruced from a vqvae

parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
parser.add_argument('-r', dest='dataset_root_folder', required=True, help='Root folder for dataset')
parser.add_argument('--split-filename', default='', help='Dataset split filename')
parser.add_argument('-e', dest='num_epochs', type=int, required=True,
                    help='Total number of epochs to train')
parser.add_argument('-l', dest='log_interval', type=int, default=1000,
                    help='Intervalwith which to log')
parser.add_argument('-f', dest='checkpoint_filepath', default='lstm_baseline.pth.tar',
                    help='Checkpoint file path')
parser.add_argument('--use-cuda', dest='use_cuda', action='store_true', default=False,
                    help='Whether to use cuda for training')
parser.add_argument('-o', dest='log_filepath', default='log_lstm_baseline.txt',
                    help='Output file for logging')
parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
                    help='Momentum for AdaDelta')
parser.add_argument('--wieght-decay', dest='weight_decay', type=float, default=0.005,
                    help='Weight decay for AdaDelta')
parser.add_argument('--lr', dest='lr', type=float, default=0.05,
                    help='Learning rate for AdaDelta')
parser.add_argument('--fpa-subj-split', type=bool, default=False, help='Whether to use the FPA paper cross-subject split')
parser.add_argument('--fpa-obj-split', type=bool, default=False, help='Whether to use the FPA paper cross-object split')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
parser.add_argument('--gt_folder', dest='gt_folder', default='Hand_pose_annotation_v1',
                    help='Folder with Subject groundtruth')
parser.add_argument('--num_joints', type=int, dest='num_joints', default=21, help='Number of joints')
parser.add_argument('-c', dest='checkpoint_filename', required=True, help='Checkpoint filename')

args = parser.parse_args()
args.fpa_subj_split = True

transform_color = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.5, 0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5, 0.5))])

transform_depth = transforms.Compose([transforms.ToTensor()])



if args.checkpoint_filename == '':
    print('Creating model...')
    model_params_dict = {
        'joint_ixs': range(2),
        'hand_only': args.fpa_subj_split,
        'num_channels': 2
    }

    model = JORNet_light(model_params_dict)
    if args.use_cuda:
        model = model.cuda()
    train_vars = {
        'iter_size': 1,
        'total_loss': 0,
        'verbose': True,
        'checkpoint_filenamebase': 'checkpoint_pose_subj_vqvae',
        'checkpoint_filename': 'checkpoint_pose_subj_vqvae.pth.tar',
        'curr_iter': 0,
        'batch_size': args.batch_size,
        'log_interval': args.log_interval,
        'best_loss': 1e10,
        'losses': [],
        'output_filepath': 'log.txt',
        'tot_epoch': args.num_epochs,
        'split_filename': args.split_filename,
        'fpa_subj_split': args.fpa_subj_split,
        'fpa_obj_split': args.fpa_obj_split,
        'dataset_root_folder': args.dataset_root_folder
    }
else:
    print('Loading model from checkpoint: {}'.format(args.checkpoint_filename))
    model, _, train_vars, _ = trainer.load_checkpoint(args.checkpoint_filename, JORNet_light,
                                             use_cuda=True,
                                             fpa_subj=args.fpa_subj_split,
                                             num_channels=2)
    if args.use_cuda:
        model.cuda()
    train_vars['split_filename'] = args.split_filename
    train_vars['fpa_subj_split'] = args.fpa_subj_split
    train_vars['fpa_obj_split'] = args.fpa_obj_split
    train_vars['dataset_root_folder'] = args.dataset_root_folder

model.train()

train_loader = fpa_dataset.DataLoaderPoseRegressionFromVQVAE(root_folder=train_vars['dataset_root_folder'],
                                              type='train',
                                              input_type="rgbd",
                                              transform_color=transform_color,
                                              transform_depth=transform_depth,
                                              batch_size=train_vars['batch_size'],
                                              split_filename=train_vars['split_filename'],
                                              fpa_subj_split=train_vars['fpa_subj_split'],
                                              fpa_obj_split=train_vars['fpa_obj_split'])

train_vars['tot_iter']: len(train_loader)
train_vars['num_batches']: len(train_loader)

print('Length of dataset: {}'.format(len(train_loader.dataset)))

print('Creating optimizer...')
optimizer = optim.Adadelta(model.parameters(),
                           rho=args.momentum,
                           weight_decay=args.weight_decay,
                           lr=args.lr)
loss_func = my_losses.cross_entropy_loss_p_logq
losses = []
for i in range(len(train_loader.dataset)):
    losses.append([])

continue_epoch = train_vars['epoch']
continue_batch_idx = train_vars['batch_idx']
continue_to_batch = True

for epoch_idx in range(args.num_epochs - 1):
    epoch = epoch_idx + 1
    train_vars['epoch'] = epoch
    if epoch < train_vars['continue_epoch']:
        print('Continuing epoch: ({}/{})/{}'.format(epoch, train_vars['continue_epoch'], args.num_epochs))
        continue
    train_vars['curr_epoch_iter'] = epoch
    for batch_idx, (depth_img_torch, hand_obj_pose) in enumerate(train_loader):
        if continue_to_batch:
            if batch_idx <= train_vars['continue_batch_idx']:
                if batch_idx % 10 == 0:
                    print('Continuing batch: ({}/{})/{}'.format(batch_idx, train_vars['continue_batch_idx'], len(train_loader)))
                    continue
            else:
                print('Arrived at continue batch. Log iteration: {}'.format(args.log_interval))
                continue_to_batch = False

        if epoch == continue_epoch and batch_idx < args.log_interval:
            if batch_idx % (int(args.log_interval / 10)) == 0:
                print('Pre-log batch iterations. Logging after them, at every {} batch iterations: {}/{}'.
                      format(args.log_interval, batch_idx, args.log_interval))
        train_vars['batch_idx'] = batch_idx
        train_vars['curr_iter'] = batch_idx + 1
        if args.use_cuda:
            depth_img_torch = depth_img_torch.cuda()
            hand_obj_pose = hand_obj_pose.cuda()
        # zero out torch gradients
        optimizer.zero_grad()

        # clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        output = model(depth_img_torch)

        loss = my_losses.euclidean_loss(output, hand_obj_pose)
        loss.backward()
        train_vars['total_loss'] = loss.item()
        train_vars['losses'].append(train_vars['total_loss'])
        if train_vars['total_loss'] < train_vars['best_loss']:
            train_vars['best_loss'] = train_vars['total_loss']

        optimizer.step()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            trainer.print_log_info(model, optimizer, epoch,  train_vars)
    batch_idx = 0
    trainer.print_log_info(model, optimizer, epoch, train_vars)
