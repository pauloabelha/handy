import numpy as np
import torch.optim as optim
from torchvision import transforms
import argparse
import fpa_dataset
from JORNet_light import JORNet_light
import losses as my_losses
import trainer
import visualize as vis
import camera as cam

def get_avg_3D_error(out_numpy, gt_numpy):
    assert len(out_numpy.shape) == len(gt_numpy.shape) and \
           out_numpy.shape[0] == gt_numpy.shape[0] and \
           out_numpy.shape[1] == gt_numpy.shape[1]
    avg_3D_error_sub = np.abs(out_numpy - gt_numpy)
    avg_3D_error = np.zeros((avg_3D_error_sub.shape[0]))
    for j in range(out_numpy.shape[1]):
        avg_3D_error += np.power(avg_3D_error_sub[:, j], 2)
    return np.sum(np.sqrt(avg_3D_error)) / avg_3D_error.shape[0]


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

test_loader = fpa_dataset.DataLoaderPoseRegressionFromVQVAE(root_folder=train_vars['dataset_root_folder'],
                                              type='test',
                                              input_type="rgbd",
                                              transform_color=transform_color,
                                              transform_depth=transform_depth,
                                              batch_size=1,
                                              split_filename=train_vars['split_filename'],
                                              fpa_subj_split=train_vars['fpa_subj_split'],
                                              fpa_obj_split=train_vars['fpa_obj_split'])

train_vars['tot_iter']: len(test_loader)
train_vars['num_batches']: len(test_loader)

print('Length of dataset: {}'.format(len(test_loader.dataset)))

continue_epoch = 0#train_vars['epoch']
continue_batch_idx = -1#train_vars['batch_idx']
continue_to_batch = True

train_vars['hand_joints_3d_error'] = []
args.log_interval = 100

print('Ready to test!')
for batch_idx, (depth_img_torch, hand_obj_pose, hand_root) in enumerate(test_loader):
    if batch_idx > 0 and batch_idx % args.log_interval == 0:
        print('Testing... Logging every {} batch iterations: {}/{}'.
              format(args.log_interval, batch_idx, len(test_loader.dataset)))
        print('    Average hand joint 3D error: {}'.format(np.mean(np.array(train_vars['hand_joints_3d_error']))))
        print('    Stddev hand joint 3D error:  {}'.format(np.std(np.array(train_vars['hand_joints_3d_error']))))
        print('-------------------------------------------------------------------------')

    train_vars['batch_idx'] = batch_idx
    train_vars['curr_iter'] = batch_idx + 1
    if args.use_cuda:
        depth_img_torch = depth_img_torch.cuda()
        hand_obj_pose = hand_obj_pose.cuda()
     # clear out the hidden state of the LSTM,
    # detaching it from its history on the last instance.
    output = model(depth_img_torch)

    output_numpy = output.detach().cpu().numpy()
    hand_obj_pose_numpy = hand_obj_pose.detach().cpu().numpy()
    hand_root_numpy = hand_root.detach().cpu().numpy()

    if not args.fpa_subj_split:
        out_obj_pose_abs = output_numpy[0, 60:].reshape((6,))
        out_obj_pose_abs[0:3] = hand_root_numpy + out_obj_pose_abs[0:3]

        gt_obj_pose_abs = hand_obj_pose_numpy[0, 60:].reshape((6,))
        gt_obj_pose_abs[0:3] = hand_root_numpy + gt_obj_pose_abs[0:3]

        obj_transl_avg_3D_error = get_avg_3D_error(out_obj_pose_abs[0:3].reshape((1, 3)),
                                                   gt_obj_pose_abs[0:3].reshape((1, 3)))

        obj_angle_avg_error = get_avg_3D_error(out_obj_pose_abs[3:].reshape((1, 3)),
                                               gt_obj_pose_abs[3:].reshape((1, 3)))

    out_hand_pose_abs = np.zeros((21, 3))
    hand_pose_rel = output_numpy[0, 0:60].reshape((20, 3))
    out_hand_pose_abs[0, :] = hand_root_numpy
    out_hand_pose_abs[1:, :] = out_hand_pose_abs[0, :] + hand_pose_rel
    out_hand_pose_uv = cam.joints_depth2color(out_hand_pose_abs, cam.fpa_depth_intrinsics)[:, 0:2]

    gt_hand_pose_abs = np.zeros((21, 3))
    gt_hand_pose_abs[0, :] = hand_root_numpy
    gt_hand_pose_rel = hand_obj_pose_numpy
    gt_hand_pose_rel = gt_hand_pose_rel[0, 0:60].reshape((20, 3))
    gt_hand_pose_abs[1:, :] = gt_hand_pose_abs[0, :] + gt_hand_pose_rel
    gt_hand_pose_uv = cam.joints_depth2color(gt_hand_pose_abs, cam.fpa_depth_intrinsics)[:, 0:2]

    # fig = vis.plot_joints(gt_hand_pose_uv)
    # vis.plot_joints(out_hand_pose_uv, fig=fig)
    # vis.show()

    # fig, ax = vis.plot_3D_joints(gt_hand_pose_abs, color='C0')
    # vis.plot_3D_joints(out_hand_pose_abs, fig=fig, ax=ax)
    # vis.show()

    # get hand joint 3D error (mm)
    hand_joints_3d_error = get_avg_3D_error(out_hand_pose_abs, gt_hand_pose_abs)

    loss = my_losses.euclidean_loss(output, hand_obj_pose)
    train_vars['hand_joints_3d_error'].append(hand_joints_3d_error)

    train_vars['total_loss'] = loss.item()
    train_vars['losses'].append(train_vars['total_loss'])
    if train_vars['total_loss'] < train_vars['best_loss']:
        train_vars['best_loss'] = train_vars['total_loss']

    # if batch_idx > 0 and batch_idx % args.log_interval == 0:
    #    trainer.print_log_info(model, None, epoch,  train_vars)

# trainer.print_log_info(model, None, epoch, train_vars)

hand_error_per_slot = np.zeros((81,))
hand_3d_error_array = np.array(train_vars['hand_joints_3d_error'])
num_errors = hand_3d_error_array.shape[0]
for i in range(hand_error_per_slot.shape[0]):
    hand_error_per_slot[i] = np.sum(hand_3d_error_array < i) / num_errors
for i in range(hand_error_per_slot.shape[0]):
    print('{}'.format(hand_error_per_slot[i]))
