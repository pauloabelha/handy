import fpa_io
import visualize
import camera as cam


dataset_root_folder = '/home/paulo/fpa_benchmark/'
gt_folder = 'Hand_pose_annotation_v1'
data_folder = 'video_files'
subject = 'Subject_1'
action = 'open_letter'
sequence = '4'
gt_filepath = '/'.join([dataset_root_folder, gt_folder, subject, action, sequence, 'skeleton.txt'])
curr_data_folder = '/'.join([dataset_root_folder, data_folder, subject, action, sequence])


gt_skeletons = fpa_io.read_action_joints_sequence(gt_filepath)

fig = visualize.create_fig()
for i in range(99):
    if i < 10:
        frame_num = '000' + str(i)
    else:
        frame_num = '00' + str(i)
    joints = gt_skeletons[i].joints
    color_filepath = '/'.join([curr_data_folder, 'color', 'color_' + frame_num + '.jpeg'])
    depth_filepath = '/'.join([curr_data_folder, 'depth', 'depth_' + frame_num + '.png'])
    depth_img = fpa_io.read_depth_img(depth_filepath)
    joints_uv = cam.joints_depth2color(joints, cam.fpa_depth_intrinsics)
    visualize.plot_joints_from_colorspace(joints_colorspace=joints_uv, data=depth_img,
                                          fig=fig, title='/'.join([subject, action, sequence]))
    visualize.pause(0.1)
    visualize.clear_plot()

visualize.show()


