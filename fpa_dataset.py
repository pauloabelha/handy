import torch
from torch.utils.data.dataset import Dataset
import fpa_io
import visualize as vis
import camera as cam
import numpy as np
import io_image
import converter as conv


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

class FPADataset(Dataset):
    crop_res = (200, 200)
    orig_img_res = (640, 480)
    new_img_res = (640, 480)
    video_folder = 'Video_files/'
    pose_folder = 'Hand_pose_annotation_v1/'
    color_folder = 'color/'
    depth_folder = 'depth/'
    color_fileext = 'jpeg'
    depth_fileext = 'png'
    dataset_split = None
    for_autoencoding = False
    input_type = ""

    def __init__(self, root_folder, type, input_type, transform_color=None,
                 transform_depth=None, img_res=None, crop_res=None, split_filename='',
                 for_autoencoding=False):
        self.root_folder = root_folder
        self.transform_color = transform_color
        self.transform_depth = transform_depth
        self.split_filename = split_filename
        self.type = type
        self.for_autoencoding = for_autoencoding
        self.input_type = input_type


    def get_subpath_and_file_num(self, idx):
        idx_split = self.dataset_split[self.type][idx]
        subpath = idx_split[0]
        file_num = idx_split[1]
        return subpath, file_num

    def read_depth_img(self, subpath, file_num):
        depth_filepath = self.root_folder + self.video_folder + subpath + \
                         self.depth_folder + 'depth_' + \
                         file_num + '.' + self.depth_fileext
        return fpa_io.read_depth_img(depth_filepath)

    def conv_depth_img_with_torch_transform(self, depth_img_numpy, transform):
        depth_img_torch = depth_img_numpy.reshape((depth_img_numpy.shape[0],
                                              depth_img_numpy.shape[1], 1)).astype(float)
        depth_img_torch = transform(depth_img_torch).float()
        return depth_img_torch

    def get_depth_img_with_torch_transform(self, idx, transform):
        subpath, file_num = self.get_subpath_and_file_num(idx)
        depth_img_numpy = self.read_depth_img(subpath, file_num)
        depth_img_torch = self.conv_depth_img_with_torch_transform(depth_img_numpy, transform)
        return depth_img_torch

    def get_hand_joint(self, subpath, file_num):
        joints_filepath = self.root_folder + self.pose_folder + \
                          subpath + 'skeleton.txt'
        return fpa_io.read_action_joints_sequence(joints_filepath)[int(file_num)]

    def __getitem__(self, idx):
        return None

    def __len__(self):
        return len(self.dataset_split[self.type])

class FPADatasetTracking(FPADataset):

    def __init__(self, root_folder, type, input_type,  transform_color=None,
                 transform_depth=None, img_res=None, crop_res=None, split_filename='',
                 for_autoencoding=False):
        super(FPADatasetTracking, self).__init__(root_folder, type,
                                                 input_type,
                                                 transform_color=transform_color,
                                                 transform_depth=transform_depth,
                                                 img_res=img_res,
                                                 crop_res=crop_res,
                                                 split_filename=split_filename,
                                                 for_autoencoding=for_autoencoding)
        if not crop_res is None:
            self.crop_res = crop_res

        if self.split_filename == '':
            fpa_io.create_split_file_tracking(self.root_folder,
                                              self.video_folder,
                                              perc_train=0.7, perc_valid=0.15)
        else:
            self.dataset_split = fpa_io.load_split_file(
                self.root_folder, self.split_filename)

    def __getitem__(self, idx):
        subpath, file_num = self.get_subpath_and_file_num(idx)

        depth_filepath = self.root_folder + self.video_folder + subpath + \
                         self.depth_folder + 'depth_' + \
                         file_num + '.' + self.depth_fileext
        depth_image = fpa_io.read_depth_img(depth_filepath)

        joints_filepath = self.root_folder + self.pose_folder + \
                          subpath + 'skeleton.txt'
        joints = fpa_io.read_action_joints_sequence(joints_filepath)[int(file_num)]

        joints_uv = cam.joints_depth2color(joints.reshape((21, 3)), cam.fpa_depth_intrinsics)[:, 0:2]
        joints_uv[:, 0] = np.clip(joints_uv[:, 0], a_min=0, a_max=depth_image.shape[0] - 1)
        joints_uv[:, 1] = np.clip(joints_uv[:, 1], a_min=0, a_max=depth_image.shape[1] - 1)

        data_image = depth_image.reshape((depth_image.shape[0], depth_image.shape[1], 1)).astype(float)
        data_image = self.transform_depth(data_image).float()
        if self.for_autoencoding:
            return data_image, data_image
        _, crop_coords = io_image.crop_hand_depth(joints_uv, depth_image)
        crop_coords_numpy = np.zeros((2, 2))
        crop_coords_numpy[0, 0] = crop_coords[0]
        crop_coords_numpy[0, 1] = crop_coords[1]
        crop_coords_numpy[1, 0] = crop_coords[2]
        crop_coords_numpy[1, 1] = crop_coords[3]

        corner_heatmap1 = conv.color_space_label_to_heatmap(crop_coords_numpy[0, :],
                                                          heatmap_res=self.orig_img_res,
                                                          orig_img_res=self.orig_img_res)
        #print(np.unravel_index(np.argmax(corner_heatmap1), corner_heatmap1.shape))
        corner_heatmap2 = conv.color_space_label_to_heatmap(crop_coords_numpy[1, :],
                                                     heatmap_res=self.orig_img_res,
                                                     orig_img_res=self.orig_img_res)

        corner_heatmaps = np.stack((corner_heatmap1, corner_heatmap2))
        corner_heatmaps = torch.from_numpy(corner_heatmaps).float()
        return data_image, corner_heatmaps


class FPADatasetPoseRegression(FPADataset):

    def __init__(self, root_folder, type, input_type, transform_color=None,
                 transform_depth=None, img_res=None, crop_res=None, split_filename='',
                 for_autoencoding=False):
        super(FPADatasetPoseRegression, self).__init__(root_folder, type,
                                                 input_type,
                                                 transform_color=transform_color,
                                                 transform_depth=transform_depth,
                                                 img_res=img_res,
                                                 split_filename=split_filename,
                                                 for_autoencoding=for_autoencoding)
        if self.split_filename == '':
            fpa_io.create_split_obj_pose(self.root_folder,
                                              self.video_folder,
                                              perc_train=0.7, perc_valid=0.15)
        else:
            self.dataset_split = fpa_io.load_split_file(
                self.root_folder, self.split_filename)

    def __getitem__(self, idx):
        subpath, file_num = self.get_subpath_and_file_num(idx)
        depth_img_torch = self.get_depth_img_with_torch_transform(idx, self.transform_depth)
        hand_joints = self.get_hand_joint(subpath, file_num)
        a = 0

def DataLoaderTracking(root_folder, type, input_type, transform_color=None, transform_depth=None, batch_size=4,
               img_res=None, split_filename='', for_autoencoding=False):
    dataset = FPADatasetTracking(root_folder,
                         type, input_type,
                         transform_color=transform_color,
                                 transform_depth=transform_depth,
                         img_res=img_res,
                         split_filename=split_filename,
                                 for_autoencoding=for_autoencoding)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False)

def DataLoaderPoseRegression(root_folder, type, input_type, transform_color=None, transform_depth=None, batch_size=4,
               img_res=None, split_filename='', for_autoencoding=False):
    dataset = FPADatasetPoseRegression(root_folder,
                         type, input_type,
                         transform_color=transform_color,
                                 transform_depth=transform_depth,
                         img_res=img_res,
                         split_filename=split_filename,
                                 for_autoencoding=for_autoencoding)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False)


''' OLD, BUT WORKING

crop_depth = crop_depth.astype(float)

        joints_uv[:, 0] -= crop_coords[0]
        joints_uv[:, 1] -= crop_coords[1]
        #vis.plot_joints_from_colorspace(joints_colorspace=joints_uv,
        #                                data=crop_depth,
        #                                title=depth_filepath)
        # vis.plot_image(color_image)
        # vis.show()

        crop_depth_res = crop_depth.shape
        crop_depth = io_image.change_res_image(crop_depth, self.crop_res)
        a = (crop_depth.shape[0] / crop_depth_res[0])
        joints_uv_crop = np.copy(joints_uv)
        joints_uv_crop[:, 0] = joints_uv_crop[:, 0] * a
        joints_uv_crop[:, 1] *= (crop_depth.shape[1] / crop_depth_res[1])
        joints_uv_crop = joints_uv_crop.astype(int)
        #print(joints_uv_crop)
        #vis.plot_joints_from_colorspace(joints_colorspace=joints_uv_crop,
        #                                data=crop_depth,
        #                                title=depth_filepath)
        # vis.plot_image(color_image)
        #vis.show()


'''