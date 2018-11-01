import numpy as np
import io_image
import os
import pickle
import util
from eulerangles import euler2mat, mat2euler

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

def read_obj_poses(filepath):
    with open(filepath) as f:
        lines = f.readlines()
        if len(lines) == 0:
            return None
        joints_sequence = np.zeros((len(lines), num_joints * 3))
        i = -1
        obj_poses = np.zeros((len(lines), 6))
        for line in lines:
            i += 1
            line_split = line.split(' ')
            frame_num = line_split[0]
            obj_mtx = np.array([float(x) for x in line_split[1:-1]]).reshape((4, 4)).T
            obj_transl = obj_mtx[0:3, 3]
            obj_rot_mtx = obj_mtx[0:3, 0:3]
            obj_euler_angles = np.array(mat2euler(obj_rot_mtx))
            obj_poses[i, 0:3] = obj_transl
            obj_poses[i, 3:] = obj_euler_angles
        return obj_poses

def read_action_joints_sequence(filepath):
    #action = filepath.split('/')[6]
    with open(filepath) as f:
        lines = f.readlines()
        if len(lines) == 0:
            return None
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

def read_color_img(filepath, img_res=None):
    color_img = io_image.read_RGB_image(filepath, new_res=img_res)
    return color_img

def read_depth_img(filepath):
    depth_img = io_image.read_RGB_image(filepath)
    return depth_img



def create_split_file_old(dataset_root_folder, gt_folder, num_train_seq,
                      split_filename='fpa_split.p', actions=None):
    if num_train_seq >= 3:
        raise 1
    dataset_tuple_train = []
    dataset_tuple_valid = []
    dataset_tuple_test = []
    action_to_idx = {}
    idx_to_action = {}
    data_path = '/'.join([dataset_root_folder, gt_folder])
    subject_dirs = os.listdir(data_path)
    orig_num_train_seq =num_train_seq
    num_actions = -1
    for subject_dir in subject_dirs:
        print('Splitting dataset: {} of {}'.format(subject_dir, len(subject_dirs)))
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
                    if joints_seq is None:
                        print('Found action sequence without ground truth: {}'.
                              format((subject_dir, action_dir, curr_seq_ix)))
                        continue
                    dataset_tuple_train.append((subject_dir, action_dir, curr_seq_ix, joints_seq))
                num_valid_seqs = int(np.floor((num_seqs - num_train_seq) / 2))
                for i in range(num_valid_seqs):
                    curr_seq_ix = seq_ixs[(num_train_seq+i)]
                    seq_gt_filepath = '/'.join([dataset_root_folder, gt_folder,
                                                subject_dir, action_dir, str(curr_seq_ix),
                                                'skeleton.txt'])
                    joints_seq = read_action_joints_sequence(seq_gt_filepath)
                    if joints_seq is None:
                        print('Found action sequence without ground truth: {}'.
                              format((subject_dir, action_dir, curr_seq_ix)))
                        continue
                    dataset_tuple_valid.append((subject_dir, action_dir, curr_seq_ix, joints_seq))
                num_test_seqs = num_seqs - (num_train_seq + num_valid_seqs)
                for i in range(num_test_seqs):
                    curr_seq_ix = seq_ixs[(num_train_seq + num_valid_seqs + i)]
                    seq_gt_filepath = '/'.join([dataset_root_folder, gt_folder,
                                                subject_dir, action_dir, str(curr_seq_ix),
                                                'skeleton.txt'])
                    joints_seq = read_action_joints_sequence(seq_gt_filepath)
                    if joints_seq is None:
                        print('Found action sequence without ground truth: {}'.
                              format((subject_dir, action_dir, curr_seq_ix)))
                        continue
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

def create_split_file(dataset_root_folder, gt_folder, perc_train, perc_valid,
                      split_filename='fpa_split_tracking.p',
                      only_with_obj_pose=False,
                           fpa_subj_split=False,
                      fpa_obj_split=False):
    if fpa_subj_split and fpa_obj_split:
        raise 1
    if fpa_subj_split:
        print('Performing the FPA paper cross subjet split')
    if fpa_obj_split:
        print('Performing the FPA paper cross object split')
    train_subjs_split = ['Subject_1', 'Subject_3', 'Subject_4']
    test_objs_split = ['peanut', 'fork', 'milk', 'tea', 'soap', 'spray', 'flash', 'paper', 'letter',
                        'calculator', 'phone', 'coin', 'card', 'wine']
    objs_with_pose_annotation = ['juice', 'milk', 'salt', 'soap']
    data_path = '/'.join([dataset_root_folder, gt_folder])
    subject_dirs = os.listdir(data_path)
    path_tuples = []
    subject_idx = 0
    num_frames = 0
    num_frames_with_milk = 0
    num_frames_with_salt = 0
    num_frames_with_juice = 0
    num_frames_with_soap = 0
    for subject_dir in subject_dirs:
        subject_idx += 1
        print('Splitting dataset: Subject {} of {} ; {}'.format(subject_idx, len(subject_dirs), subject_dir))
        subject_path = '/'.join([data_path, subject_dir])
        action_dirs = os.listdir(subject_path)
        for action_dir in action_dirs:
            action_path = '/'.join([subject_path, action_dir])
            seq_dirs = os.listdir(action_path)

            curr_action_has_obj_pose_annotation = False
            if only_with_obj_pose:
                for obj_name in objs_with_pose_annotation:
                    if obj_name in action_dir:
                        curr_action_has_obj_pose_annotation = True
                        break
            if only_with_obj_pose and not curr_action_has_obj_pose_annotation:
                continue
            for seq_dir in seq_dirs:
                seq_path = '/'.join([action_path, seq_dir]) + '/'
                color_files = util.list_files_in_dir(seq_path + 'color/')
                depth_files = util.list_files_in_dir(seq_path + 'depth/')

                # TODO : check that skeleton.txt exist in respective folder
                curr_subpath = subject_dir + '/' + action_dir + '/' + seq_dir + '/'

                obj_pose_filepath = dataset_root_folder + 'Object_6D_pose_annotation_v1/' + curr_subpath + 'object_pose.txt'
                if only_with_obj_pose and not os.path.isfile(obj_pose_filepath):
                    print('WARNING. Could not find obj pose annotation ground truth for: {}'.format(obj_pose_filepath))
                    continue

                skeleton_filepath = dataset_root_folder + 'Hand_pose_annotation_v1/' + curr_subpath + 'skeleton.txt'
                if not os.path.isfile(skeleton_filepath):
                    print('WARNING. Could not find skeleton ground truth for: {}'.format(skeleton_filepath))
                    continue

                if not len(color_files) == len(color_files):
                    print('WARNING. Skipping: Number of color and depth files is different: {}'.format(seq_path))
                else:
                    for color_file in color_files:
                        color_num = color_file.split('.')[0].split('_')[1]
                        for depth_file in depth_files:
                            depth_num = depth_file.split('.')[0].split('_')[1]
                            if color_num == depth_num:
                                path_tuples.append((curr_subpath, color_num))
                                break

    print('Performing split per se')
    ixs_randomize = np.random.choice(len(path_tuples), len(path_tuples), replace=False)
    path_tuples = np.array(path_tuples)
    path_tuples_randomised = path_tuples[ixs_randomize]
    path_tuples_train = []
    path_tuples_valid = []
    path_tuples_test = []
    num_train = 0
    num_valid = 0
    num_test = 0
    if fpa_subj_split:
        split_filename = 'fpa_subj_split.p'
        for path_tuple in path_tuples:
            subject = path_tuple[0].split('/')[0]
            if subject in train_subjs_split:
                path_tuples_train.append(path_tuple)
                num_train += 1
            else:
                path_tuples_test.append(path_tuple)
                num_test += 1
        print('Num train: {}'.format(num_train))
        print('Num test: {}'.format(num_test))
        ixs_randomize_train = np.random.choice(num_train, num_train, replace=False)
        path_tuples_train = np.array(path_tuples_train)
        path_tuples_train = path_tuples_train[ixs_randomize_train]
        path_tuples_test = np.array(path_tuples_test)
        ixs_randomize_test = np.random.choice(num_test, num_test, replace=False)
        path_tuples_test = path_tuples_test[ixs_randomize_test]
        perc_train = num_train / (num_train + num_test)
        perc_valid = 0
        perc_test = num_test / (num_train + num_test)
    elif fpa_obj_split:
        split_filename = 'fpa_obj_split.p'
        for path_tuple in path_tuples:
            action = path_tuple[0].split('/')[1]
            found_obj = False
            for test_obj_split in test_objs_split:
                if test_obj_split in action:
                    path_tuples_test.append(path_tuple)
                    num_test += 1
                    found_obj = True
                    break
            if not found_obj:
                path_tuples_train.append(path_tuple)
                num_train += 1
        print('Num train: {}'.format(num_train))
        print('Num test: {}'.format(num_test))
        ixs_randomize_train = np.random.choice(num_train, num_train, replace=False)
        path_tuples_train = np.array(path_tuples_train)
        path_tuples_train = path_tuples_train[ixs_randomize_train]
        path_tuples_test = np.array(path_tuples_test)
        ixs_randomize_test = np.random.choice(num_test, num_test, replace=False)
        path_tuples_test = path_tuples_test[ixs_randomize_test]
        perc_train = num_train / (num_train + num_test)
        perc_valid = 0
        perc_test = num_test / (num_train + num_test)
    else:
        num_train = int(np.floor(len(path_tuples) * perc_train))
        num_valid = int(np.floor(len(path_tuples) * perc_valid))
        path_tuples_train = path_tuples_randomised[0: num_train]
        path_tuples_valid = path_tuples_randomised[num_train: num_train + num_valid]
        path_tuples_test = path_tuples_randomised[num_train + num_valid:]
        perc_test = 1 - (perc_train + perc_valid)

    file_ixs = np.array(range(len(ixs_randomize)))
    file_ixs_randomized = file_ixs[ixs_randomize]
    file_ixs_train = file_ixs_randomized[0: num_train]
    file_ixs_valid = file_ixs_randomized[num_train: num_train + num_valid]
    file_ixs_test = file_ixs_randomized[num_train + num_valid:]

    dataset_tuples = {
        'fpa_subj_split': fpa_subj_split,
        'fpa_obj_split': fpa_obj_split,
        'path_tuples': path_tuples,
        'perc_train': perc_train,
        'perc_valid': perc_valid,
        'perc_test': perc_test,
        'all_seq': path_tuples,
        'ixs_random': ixs_randomize,
        'all_random': path_tuples_randomised,
        'train': path_tuples_train,
        'valid': path_tuples_valid,
        'test': path_tuples_test,
        'train_ixs': file_ixs_train,
        'valid_ixs': file_ixs_valid,
        'text_ixs': file_ixs_test
    }
    with open('/'.join([dataset_root_folder, split_filename]), 'wb') as handle:
        pickle.dump(dataset_tuples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Created split file: {}'.format([dataset_root_folder, split_filename]))
    return dataset_tuples



def load_split_file(dataset_root_folder, split_filename='fpa_split.p'):
    return pickle.load(open('/'.join([dataset_root_folder, split_filename]), "rb"))

def get_action_idx(dataset_tuples, action_name):
    action_idx = 0

    return action_idx

