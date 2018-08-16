import numpy as np
import io_image
import os
import pickle

num_joints = 21

class GTSkeleton:
    def __init__(self, filepath, action, frame_num, joints):
        self.filepath = filepath
        self.action = action
        self.frame_num = frame_num
        self.joints = joints

def conv_joints_to_canonical(joints_fpa):
    joints = np.zeros((joints_fpa.shape[0], 3))
    joints[0] = joints_fpa[0]
    for i in range(5):
        ix_canonical= (i * 4) + 1
        joints[ix_canonical] = joints_fpa[i+1]
    curr_ix_canonical = 2
    curr_ix_fpa = 6
    for i in range(5):
        joints[curr_ix_canonical:curr_ix_canonical+3] =\
            joints_fpa[curr_ix_fpa:curr_ix_fpa+3]
        curr_ix_canonical += 4
        curr_ix_fpa += 3

    return joints

def read_action_joints_sequence(filepath):
    #action = filepath.split('/')[6]
    with open(filepath) as f:
        lines = f.readlines()
        joints_sequence = np.zeros((len(lines), num_joints * 3))
        idx = 0
        for line in lines:
            line_split = line.split(' ')
            #frame_num = line_split[0]
            joints = np.array([float(x) for x in line_split[1:]]).reshape((num_joints, 3))
            joints = conv_joints_to_canonical(joints)
            joints_sequence[idx, :] = joints.reshape((num_joints * 3,))
            idx += 1
    return joints_sequence

def read_color_img(filepath):
    color_img = io_image.read_RGB_image(filepath)
    return color_img

def read_depth_img(filepath):
    depth_img = io_image.read_RGB_image(filepath)
    return depth_img

def create_split_file(dataset_root_folder, gt_folder, data_folder, num_train_seq,
                      split_filename='fpa_split.p', actions=None):
    if num_train_seq >= 3:
        raise 1
    dataset_tuple_train = []
    dataset_tuple_valid = []
    dataset_tuple_test = []
    action_to_idx = {}
    idx_to_action = {}
    data_path = '/'.join([dataset_root_folder, data_folder])
    subject_dirs = os.listdir(data_path)
    orig_num_train_seq =num_train_seq
    num_actions = -1
    for subject_dir in subject_dirs:
        subject_path = '/'.join([data_path, subject_dir])
        action_dirs = os.listdir(subject_path)
        for action_dir in action_dirs:
            if actions is None or action_dir in actions:
                try:
                    action_to_idx[action_dir]
                except KeyError:
                    action_to_idx[action_dir] = num_actions + 1
                    idx_to_action[num_actions + 1] = action_dir
                    num_actions += 1
                action_path = '/'.join([subject_path, action_dir])
                seq_dirs = os.listdir(action_path)
                num_seqs = len(seq_dirs)
                seq_array = np.array(range(num_seqs)) + 1
                seq_ixs = np.random.choice(seq_array, num_seqs, replace=False)
                if num_seqs == 3:
                    num_train_seq = 1
                else:
                    num_train_seq = orig_num_train_seq
                for i in range(num_train_seq):
                    curr_seq_ix = seq_ixs[i]
                    seq_gt_filepath = '/'.join([dataset_root_folder, gt_folder,
                                            subject_dir, action_dir, str(curr_seq_ix),
                                                'skeleton.txt'])
                    joints_seq = read_action_joints_sequence(seq_gt_filepath)
                    dataset_tuple_train.append((subject_dir, action_dir, curr_seq_ix, joints_seq))
                num_valid_seqs = int(np.floor((num_seqs - num_train_seq) / 2))
                for i in range(num_valid_seqs):
                    curr_seq_ix = seq_ixs[(num_train_seq+i)]
                    seq_gt_filepath = '/'.join([dataset_root_folder, gt_folder,
                                                subject_dir, action_dir, str(curr_seq_ix),
                                                'skeleton.txt'])
                    joints_seq = read_action_joints_sequence(seq_gt_filepath)
                    dataset_tuple_valid.append((subject_dir, action_dir, curr_seq_ix, joints_seq))
                num_test_seqs = num_seqs - (num_train_seq + num_valid_seqs)
                for i in range(num_test_seqs):
                    curr_seq_ix = seq_ixs[(num_train_seq + num_valid_seqs + i)]
                    seq_gt_filepath = '/'.join([dataset_root_folder, gt_folder,
                                                subject_dir, action_dir, str(curr_seq_ix),
                                                'skeleton.txt'])
                    joints_seq = read_action_joints_sequence(seq_gt_filepath)
                    dataset_tuple_test.append((subject_dir, action_dir, curr_seq_ix, joints_seq))
    dataset_tuples = {
        'train': dataset_tuple_train,
        'valid': dataset_tuple_valid,
        'test': dataset_tuple_test,
        'action_to_idx': action_to_idx,
        'idx_to_action': idx_to_action,
        'num_actions': num_actions + 1
    }
    with open('/'.join([dataset_root_folder, split_filename]), 'wb') as handle:
        pickle.dump(dataset_tuples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return dataset_tuples

def load_split_file(dataset_root_folder, split_filename='fpa_split.p'):
    return pickle.load(open('/'.join([dataset_root_folder, split_filename]), "rb"))

def get_action_idx(dataset_tuples, action_name):
    action_idx = 0

    return action_idx

