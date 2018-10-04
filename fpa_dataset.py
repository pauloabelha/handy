import torch
from torch.utils.data.dataset import Dataset
import fpa_io
import visualize as vis
import camera as cam
import numpy as np
import io_image


class FPADataset(Dataset):
    gt_folder = 'Hand_pose_annotation_v1'
    video_files_folder = 'video_files'
    color_folder = 'color'
    depth_folder = 'depth'
    dataset_tuples = None

    def __init__(self, root_folder, type, transform=None, img_res=None, split_filename=''):
        self.root_folder = root_folder
        self.transform = transform
        self.img_res = img_res
        self.split_filename = split_filename
        self.type = type

        if self.split_filename == '':
            fpa_io.create_split_file(self.root_folder,
                                     self.gt_folder,
                                     num_train_seq=2,
                                     actions=None)
        else:
            self.dataset_tuples = fpa_io.load_split_file(
                self.root_folder, self.split_filename)

    def __getitem__(self, idx):
        data_tuple = self.dataset_tuples[self.type][idx]
        return data_tuple

    def __len__(self):
        return len(self.dataset_tuples[self.type])

class FPADatasetTracking(Dataset):

    orig_img_res = (640, 480)
    new_img_res = (640, 480)
    video_folder = 'video_files/'
    pose_folder = 'Hand_pose_annotation_v1/'
    color_folder = 'color/'
    depth_folder = 'depth/'
    color_fileext = 'jpeg'
    depth_fileext = 'png'
    dataset_split = None
    
    def __init__(self, root_folder, type,  transform_color=None,
                 transform_depth=None, img_res=None, split_filename=''):
        self.root_folder = root_folder
        self.transform_color = transform_color
        self.transform_depth = transform_depth
        self.split_filename = split_filename
        self.type = type
        if not img_res is None:
            self.new_img_res = img_res
        self.img_res_transf = np.zeros((2,))
        self.img_res_transf[0] = self.new_img_res[0] / self.orig_img_res[0]
        self.img_res_transf[1] = self.new_img_res[1] / self.orig_img_res[1]
        
        if self.split_filename == '':
            fpa_io.create_split_file_tracking(self.root_folder,
                                     self.video_folder,
                                     perc_train=0.7, perc_valid=0.15)
        else:
            self.dataset_split = fpa_io.load_split_file(
                self.root_folder, self.split_filename)

    def __getitem__(self, idx):
        idx_split = self.dataset_split[self.type][idx]
        file_num = idx_split[1]
        depth_filepath = self.root_folder + self.video_folder + idx_split[0] +\
                         self.depth_folder + 'depth_' +\
                         file_num + '.' + self.depth_fileext
        depth_image = fpa_io.read_depth_img(depth_filepath)
        data_image = depth_image.reshape((depth_image.shape[0],
                                           depth_image.shape[1],
                                           1))
        data_image = self.transform_depth(data_image).float()
        joints_filepath = self.root_folder + self.pose_folder +\
                        idx_split[0] + 'skeleton.txt'
        joints = fpa_io.read_action_joints_sequence(joints_filepath)[int(file_num)]

        joints_uv = cam.joints_depth2color(joints.reshape((21, 3)), cam.fpa_depth_intrinsics)
        print(joints_uv)

        crop_depth, crop_coord = io_image.crop_hand_depth(joints_uv, depth_image)
        print(crop_coord)
        crop_depth = crop_depth.astype(float)

        joints_uv[:, 0] -= crop_coord[0]
        joints_uv[:, 1] -= crop_coord[1]
        vis.plot_joints_from_colorspace(joints_colorspace=joints_uv,
                                        data=crop_depth,
                                        title=depth_filepath)
        # vis.plot_image(color_image)
        vis.show()

        return data_image, joints

    def __len__(self):
        return len(self.dataset_split[self.type])



def DataLoaderTracking(root_folder, type, transform_color=None, transform_depth=None, batch_size=4,
               img_res=None, split_filename=''):
    dataset = FPADatasetTracking(root_folder,
                         type,
                         transform_color=transform_color,
                                 transform_depth=transform_depth,
                         img_res=img_res,
                         split_filename=split_filename)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False)