import torch
from torch.utils.data.dataset import Dataset
import fpa_io
import visualize as vis
import camera as cam
import numpy as np
import io_image
import converter as conv
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable


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
    pixel_bound = 100
    crop_res = (200, 200)
    orig_img_res = (640, 480)
    new_img_res = (640, 480)
    video_folder = 'Video_files/'
    hand_pose_folder = 'Hand_pose_annotation_v1/'
    obj_pose_folder = 'Object_6D_pose_annotation_v1/'
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
        self.split_filename = split_filename


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
        return depth_img_torch, depth_img_numpy

    def get_hand_joints(self, idx):
        subpath, file_num = self.get_subpath_and_file_num(idx)
        joints_filepath = self.root_folder + self.hand_pose_folder + \
                          subpath + 'skeleton.txt'
        return fpa_io.read_action_joints_sequence(joints_filepath)[int(file_num)]

    def get_obj_pose(self, idx):
        subpath, file_num = self.get_subpath_and_file_num(idx)
        obj_pose_filepath = self.root_folder + self.obj_pose_folder + \
                            subpath + 'object_pose.txt'
        return fpa_io.read_obj_poses(obj_pose_filepath)[int(file_num)]

    def get_cropped_depth_img(self, depth_img, hand_joints):
        joints_uv = cam.joints_depth2color(hand_joints.reshape((21, 3)),
                                           cam.fpa_depth_intrinsics)[:, 0:2]
        joints_uv[:, 0] = np.clip(joints_uv[:, 0], a_min=0,
                                  a_max=depth_img.shape[0] - 1)
        joints_uv[:, 1] = np.clip(joints_uv[:, 1], a_min=0,
                                  a_max=depth_img.shape[1] - 1)

        data_image = depth_img.reshape((depth_img.shape[0],
                                        depth_img.shape[1], 1)).astype(float)
        #data_image = self.transform_depth(data_image).float()
        #if self.for_autoencoding:
        #    return data_image, data_image

        cropped_img, crop_coords = io_image.crop_hand_depth(joints_uv,
                                                            depth_img,
                                                            pixel_bound=self.pixel_bound)
        return cropped_img, crop_coords

    def conv_hand_joints_to_rel(self, hand_joints):
        hand_joints = hand_joints.reshape((21, 3))
        hand_root = hand_joints[0, :]
        hand_joints = hand_joints[1:, :] - hand_root
        return hand_root, hand_joints.reshape((60,))

    def __getitem__(self, idx):
        return None

    def __len__(self):
        if (self.dataset_split):
            return len(self.dataset_split[self.type])
        else:
            return 0

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
            fpa_io.create_split_file(self.root_folder,
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

        joints_filepath = self.root_folder + self.hand_pose_folder + \
                          subpath + 'skeleton.txt'
        hand_joints = fpa_io.read_action_joints_sequence(joints_filepath)[int(file_num)]

        _, crop_coords = self.get_cropped_depth_img(depth_image, hand_joints)
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

    pixel_bound = 100

    default_split_filename = 'fpa_split_obj_pose.p'

    def __init__(self, root_folder, type, input_type, split_filename = '',
                 transform_color=None, transform_depth=None, img_res=None, crop_res=None,
                 for_autoencoding=False,
                 fpa_subj_split=False,
                 fpa_obj_split=False):
        super(FPADatasetPoseRegression, self).__init__(root_folder, type,
                                                 input_type,
                                                 transform_color=transform_color,
                                                 transform_depth=transform_depth,
                                                 img_res=img_res,
                                                 split_filename=split_filename,
                                                 for_autoencoding=for_autoencoding)
        self.fpa_subj_split = fpa_subj_split
        self.fpa_obj_split = fpa_obj_split
        if split_filename == '':
            fpa_io.create_split_file(self.root_folder, self.video_folder,
                                         perc_train=0.7, perc_valid=0.15,
                                         only_with_obj_pose=False,
                                         fpa_subj_split=fpa_subj_split,
                                         fpa_obj_split=fpa_obj_split,
                                         split_filename='fpa_split_subj.p')
            self.split_filename = self.default_split_filename
        self.dataset_split = fpa_io.load_split_file(
                self.root_folder, self.split_filename)

    def __getitem__(self, idx):
        hand_joints = self.get_hand_joints(idx)
        hand_root, hand_joints_rel = self.conv_hand_joints_to_rel(hand_joints)

        if self.fpa_subj_split:
            hand_obj_pose = hand_joints_rel
        else:
            obj_pose_rel = self.get_obj_pose(idx)
            obj_pose_rel[0:3] = obj_pose_rel[0:3] - hand_root
            hand_obj_pose = np.concatenate((hand_joints_rel, obj_pose_rel), 0)

        hand_obj_pose = torch.from_numpy(hand_obj_pose).float()

        subpath, file_num = self.get_subpath_and_file_num(idx)
        depth_img_numpy = self.read_depth_img(subpath, file_num)
        cropped_depth_img, crop_coords = self.get_cropped_depth_img(depth_img_numpy,
                                                                    hand_joints)
        cropped_depth_img = io_image.change_res_image(cropped_depth_img, new_res=(200, 200))
        depth_img_torch = self.conv_depth_img_with_torch_transform(cropped_depth_img, self.transform_depth)

        if self.type == "train":
            return depth_img_torch, hand_obj_pose
        else:
            return depth_img_torch, hand_obj_pose, hand_root

class FPADatasetReconstruction(FPADataset):

    gen_obj_folder = 'gen_objs/'

    normalise_const_max_depth = 2000

    def resize2d(self, img, size):
        return (F.adaptive_avg_pool2d(Variable(img, volatile=True), size)).data

    def __init__(self, root_folder, type, input_type, split_filename = '',
                 transform_color=None, transform_depth=None, img_res=None, crop_res=None,
                 for_autoencoding=False):

        super(FPADatasetReconstruction, self).__init__(root_folder, type,
                                                 input_type,
                                                 transform_color=transform_color,
                                                 transform_depth=transform_depth,
                                                 img_res=img_res,
                                                 split_filename=split_filename,
                                                 for_autoencoding=for_autoencoding)
        self.dataset_split = fpa_io.load_split_file(
            self.root_folder, self.split_filename)
        self.pixel_bound = 100

    def __getitem__(self, idx):
        subpath, file_num = self.get_subpath_and_file_num(idx)
        depth_img = self.read_depth_img(subpath, file_num).astype(float)

        #depth_obj_img_path = self.root_folder + self.gen_obj_folder + subpath + \
        #                str(int(file_num)) + '_depth.jpg'
        #depth_obj_img = fpa_io.read_depth_img(depth_obj_img_path)

        depth_obj_csv_path = self.root_folder + self.gen_obj_folder + subpath + \
                             str(int(file_num)) + '_depth.csv'
        img2_depth_array = np.loadtxt(open(depth_obj_csv_path, "rb"), delimiter=",")
        depth_obj_img = img2_depth_array.T

        joints_filepath = self.root_folder + self.hand_pose_folder + \
                          subpath + 'skeleton.txt'
        hand_joints = fpa_io.read_action_joints_sequence(joints_filepath)[int(file_num)]
        depth_img, _ = self.get_cropped_depth_img(depth_img, hand_joints)
        depth_obj_img, _ = self.get_cropped_depth_img(depth_obj_img, hand_joints)

        depth_img /= self.normalise_const_max_depth
        depth_obj_img /= self.normalise_const_max_depth

        #vis.plot_image(depth_img)
        #vis.show()
        #vis.plot_image(depth_obj_img)
        #vis.show()

        depth_img = depth_img.reshape((1, depth_img.shape[0], depth_img.shape[1]))
        depth_img_torch = torch.from_numpy(depth_img).float()
        depth_img_torch = self.resize2d(depth_img_torch, self.crop_res)

        depth_obj_img = depth_obj_img.reshape((1, depth_obj_img.shape[0], depth_obj_img.shape[1]))
        depth_obj_img_torch = torch.from_numpy(depth_obj_img).float()
        depth_obj_img_torch = self.resize2d(depth_obj_img_torch, self.crop_res)

        return depth_img_torch, depth_obj_img_torch

def DataLoaderReconstruction(root_folder, type, input_type,
                                 transform_color=None, transform_depth=None,
                                 batch_size=4, img_res=None, split_filename='',
                                 for_autoencoding=False):
    dataset = FPADatasetReconstruction(root_folder, type, input_type,
                                 transform_color=transform_color,
                                 transform_depth=transform_depth,
                                 img_res=img_res,
                                 split_filename=split_filename,
                                 for_autoencoding=for_autoencoding)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False)

def DataLoaderTracking(root_folder, type, input_type,
                                 transform_color=None, transform_depth=None,
                                 batch_size=4, img_res=None, split_filename='',
                                 for_autoencoding=False):
    dataset = FPADatasetTracking(root_folder, type, input_type,
                                 transform_color=transform_color,
                                 transform_depth=transform_depth,
                                 img_res=img_res,
                                 split_filename=split_filename,
                                 for_autoencoding=for_autoencoding)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False)

def DataLoaderPoseRegression(root_folder, type, input_type,
                             transform_color=None, transform_depth=None,
                             batch_size=4, img_res=None, split_filename='',
                             for_autoencoding=False,
                             fpa_subj_split=False,
                             fpa_obj_split=False):
    dataset = FPADatasetPoseRegression(root_folder,
                                       type, input_type,
                                       transform_color=transform_color,
                                       transform_depth=transform_depth,
                                       img_res=img_res,
                                       split_filename=split_filename,
                                       for_autoencoding=for_autoencoding,
                                       fpa_subj_split=fpa_subj_split,
                                       fpa_obj_split=fpa_obj_split)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False)

