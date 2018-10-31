from torchvision import transforms
import argparse
import fpa_dataset
from JORNet_light import JORNet_light
import losses as my_losses
import trainer
import numpy as np
import visualize as vis
import camera as cam


parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
parser.add_argument('-r', dest='dataset_root_folder', required=True, help='Root folder for dataset')
parser.add_argument('-c', dest='checkpoint_filename', required=True, help='Checkpoint filename')
parser.add_argument('--split-filename', default='', help='Dataset split filename')
parser.add_argument('--use-cuda', dest='use_cuda', action='store_true', default=False,
                    help='Whether to use cuda')

args = parser.parse_args()
args.batch_size = 1
args.log_interval = 500

args.fpa_subj_split = True

transform_color = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.5, 0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5, 0.5))])

transform_depth = transforms.Compose([transforms.ToTensor()])

test_loader = fpa_dataset.DataLoaderPoseRegression(root_folder=args.dataset_root_folder,
                                              type='test',
                                              input_type="depth",
                                              transform_color=transform_color,
                                              transform_depth=transform_depth,
                                              batch_size=args.batch_size,
                                              split_filename=args.split_filename,
                                              fpa_subj_split=args.fpa_subj_split)

print('Length of dataset: {}'.format(len(test_loader.dataset)))

model_params_dict = {
    'joint_ixs': range(2)
}

print('Loading model from checkpoint: {}'.format(args.checkpoint_filename))
model, _, _, _ = trainer.load_checkpoint(args.checkpoint_filename, JORNet_light, use_cuda=True, fpa_subj=args.fpa_subj_split)
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

def get_avg_3D_error(out_numpy, gt_numpy):
    assert len(out_numpy.shape) == len(gt_numpy.shape) and \
           out_numpy.shape[0] == gt_numpy.shape[0] and \
           out_numpy.shape[1] == gt_numpy.shape[1]
    avg_3D_error_sub = np.abs(out_numpy - gt_numpy)
    avg_3D_error = np.zeros((avg_3D_error_sub.shape[0]))
    for j in range(out_numpy.shape[1]):
        avg_3D_error += np.power(avg_3D_error_sub[:, j], 2)
    return np.sum(np.sqrt(avg_3D_error)) / avg_3D_error.shape[0]

print('Ready to train')
epoch = 1
train_vars['curr_epoch_iter'] = epoch
continue_batch_end_ix = -1
train_vars['hand_joints_3d_error'] = []

log_interval = int(args.log_interval/2)

for batch_idx, (depth_img_torch, hand_obj_pose, hand_root) in enumerate(test_loader):
    if batch_idx < continue_batch_end_ix:
        print('Continuing... {}/{}'.format(batch_idx, continue_batch_end_ix))
        continue
    if batch_idx % log_interval == 0:
        print('Training... Logging every {} batch iterations: {}/{}'.
              format(log_interval, batch_idx, len(test_loader.dataset)))
        if batch_idx > 10:
            print('Average hand joint 3D error: {}'.format(np.mean(np.array(train_vars['hand_joints_3d_error']))))
            print('Stddev hand joint 3D error: {}'.format(np.std(np.array(train_vars['hand_joints_3d_error']))))

    hand_obj_pose_numpy = hand_obj_pose.detach().cpu().numpy()
    hand_root_numpy = hand_root.detach().cpu().numpy()

    train_vars['batch_idx'] = batch_idx
    train_vars['curr_iter'] = batch_idx + 1
    if args.use_cuda:
        depth_img_torch = depth_img_torch.cuda()
        hand_obj_pose = hand_obj_pose.cuda()

    output = model(depth_img_torch)
    output_numpy = output.detach().cpu().numpy()

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

    #fig = vis.plot_joints(gt_hand_pose_uv)
    #vis.plot_joints(out_hand_pose_uv, fig=fig)
    #vis.show()

    #fig, ax = vis.plot_3D_joints(gt_hand_pose_abs, color='C0')
    #vis.plot_3D_joints(out_hand_pose_abs, fig=fig, ax=ax)
    #vis.show()

    # get hand joint 3D error (mm)
    hand_joints_3d_error = get_avg_3D_error(out_hand_pose_abs, gt_hand_pose_abs)

    loss = my_losses.euclidean_loss(output, hand_obj_pose)
    train_vars['hand_joints_3d_error'].append(hand_joints_3d_error)



    train_vars['total_loss'] = loss.item()
    train_vars['losses'].append(train_vars['total_loss'])
    if train_vars['total_loss'] < train_vars['best_loss']:
        train_vars['best_loss'] = train_vars['total_loss']

    #if batch_idx > 0 and batch_idx % args.log_interval == 0:
    #    trainer.print_log_info(model, None, epoch,  train_vars)

#trainer.print_log_info(model, None, epoch, train_vars)

hand_error_per_slot = np.zeros((81, ))
hand_3d_error_array = np.array(train_vars['hand_joints_3d_error'])
num_errors = hand_3d_error_array.shape[0]
for i in range(hand_error_per_slot.shape[0]):
    hand_error_per_slot[i] = np.sum(hand_3d_error_array < i) / num_errors
for i in range(hand_error_per_slot.shape[0]):
    print('{}'.format(hand_error_per_slot[i]))