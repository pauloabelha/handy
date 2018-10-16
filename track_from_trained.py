import fpa_io
import visualize
import camera as cam
import torch
import argparse
from hand_detection_net import HALNet
import trainer
import numpy as np

parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
parser.add_argument('-r', dest='dataset_root_folder', required=True, help='Root folder for dataset')
parser.add_argument('-c', dest='checkpoint_filename', required=True, help='Checkpoint filename')
parser.add_argument('--split-filename', default='', help='Dataset split filename')
parser.add_argument('--use-cuda', dest='use_cuda', action='store_true', default=False,
                    help='Whether to use cuda')
args = parser.parse_args()
args.batch_size = 1


dataset_root_folder = 'C:/Users/Administrator/Documents/Datasets/fpa_benchmark/'
gt_folder = 'Hand_pose_annotation_v1'
data_folder = 'video_files'
subject = 'Subject_1'
action = 'put_tea_bag'
sequence = '4'
gt_filepath = '/'.join([dataset_root_folder, gt_folder, subject, action, sequence, 'skeleton.txt'])
curr_data_folder = '/'.join([dataset_root_folder, data_folder, subject, action, sequence])

model, _, _, _ = trainer.load_checkpoint(args.checkpoint_filename, HALNet, use_cuda=True)
if args.use_cuda:
    model.cuda()

gt_skeletons = fpa_io.read_action_joints_sequence(gt_filepath)

fig = visualize.create_fig()
for i in range(99):
    if i < 10:
        frame_num = '000' + str(i)
    else:
        frame_num = '00' + str(i)

    color_filepath = '/'.join([curr_data_folder, 'color', 'color_' + frame_num + '.jpeg'])
    depth_filepath = '/'.join([curr_data_folder, 'depth', 'depth_' + frame_num + '.png'])
    depth_img = fpa_io.read_depth_img(depth_filepath)
    depth_img = depth_img.reshape((1, 1, depth_img.shape[0], depth_img.shape[1]))
    depth_img = torch.from_numpy(depth_img).float()
    if args.use_cuda:
        depth_img = depth_img.cuda()

    output = model(depth_img)
    output_bbox = np.zeros((2, 2))
    out_heatmaps = output[3].detach().cpu().numpy()[0, :, :, :]
    output_bbox[0, :] = np.unravel_index(np.argmax(out_heatmaps[0]), (640, 480))
    output_bbox[1, :] = np.unravel_index(np.argmax(out_heatmaps[1]), (640, 480))

    visualize.plot_image(depth_img.cpu().numpy()[0, 0, :, :], fig=fig)
    visualize.plot_bound_box(output_bbox, fig=fig, color='red')

    visualize.pause(0.001)
    visualize.clear_plot()

    #joints_uv = cam.joints_depth2color(joints, cam.fpa_depth_intrinsics)
    #visualize.plot_joints_from_colorspace(joints_colorspace=joints_uv, data=depth_img,
    #                                           fig=fig, title='/'.join([subject, action, sequence]))
    #visualize.pause(0.001)
    #visualize.clear_plot()

visualize.show()


