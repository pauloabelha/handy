import torch
from torch.utils.data.dataset import Dataset
import fpa_io
import visualize as vis


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

    img_res = (640, 480)
    gt_folder = 'video_files'
    video_files_folder = 'video_files'
    color_folder = 'color/'
    depth_folder = 'depth/'
    img_fileext = 'jpeg'
    dataset_tuples = None
    
    def __init__(self, root_folder, type,  transform=None, img_res=None, split_filename=''):
        self.root_folder = root_folder
        self.transform = transform
        self.split_filename = split_filename
        self.type = type
        
        if self.split_filename == '':
            fpa_io.create_split_file_tracking(self.root_folder,
                                     self.gt_folder,
                                     perc_train=0.7, perc_valid=0.15)
        else:
            self.dataset_split = fpa_io.load_split_file(
                self.root_folder, self.split_filename)

    def __getitem__(self, idx):
        idx_split = self.dataset_split[self.type][idx]
        root_filepath = idx_split[0]
        file_num = idx_split[1]
        color_filepath = root_filepath + self.color_folder + 'color_' + file_num + '.' + self.img_fileext
        color_image = fpa_io.read_color_img(color_filepath, img_res=self.img_res)
        vis.plot_image(color_image)
        vis.show()
        return color_image

    def __len__(self):
        return len(self.dataset_split[self.type])



def DataLoaderTracking(root_folder, type, transform=None, batch_size=4,
               img_res=None, split_filename=''):
    dataset = FPADatasetTracking(root_folder,
                         type,
                         transform=transform,
                         img_res=img_res,
                         split_filename=split_filename)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False)