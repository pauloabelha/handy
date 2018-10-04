import torch
from torch.utils.data.dataset import Dataset
import fpa_io

class FPADataset(Dataset):

    gt_folder = 'Hand_pose_annotation_v1'
    dataset_tuples = None
    
    def __init__(self, root_folder, type,  transform=None, img_res=None, split_filename=None):
        self.root_folder = root_folder
        self.transform = transform
        self.img_res = img_res
        self.split_filename = split_filename
        self.type = type
        
        if self.split_filename is None:
            fpa_io.create_split_file(self.root_folder,
                                     self.gt_folder,
                                     num_train_seq=2,
                                     actions=None)
        else:
            dataset_tuples = fpa_io.load_split_file(
                self.root_folder, self.split_filename)

    def __getitem__(self, idx):
        return self.dataset_tuples[self.type][idx]

    def __len__(self):
        return len(self.dataset_tuples[self.type])



def DataLoader(root_folder,  transform=None, batch_size=4, img_res=(64, 64), split_filename=None):
    dataset = FPADataset(root_folder,
                         transform=transform,
                         img_res=img_res,
                         split_filename=split_filename)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False)