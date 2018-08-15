import numpy as np
import io_image

num_joints = 21

class GTSkeleton:
    def __init__(self, filepath, frame_num, joints):
        self.filepath = filepath
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

def read_skeletons(filepath):
    with open(filepath) as f:
        lines = f.readlines()
        gt_skeletons = []
        for line in lines:
            line_split = line.split(' ')
            frame_num = line_split[0]
            joints = np.array([float(x) for x in line_split[1:]]).reshape((num_joints, 3))
            joints = conv_joints_to_canonical(joints)
            gt_skeletons.append(GTSkeleton(filepath, frame_num, joints))
    return gt_skeletons

def read_color_img(filepath):
    color_img = io_image.read_RGB_image(filepath)
    return color_img

def read_depth_img(filepath):
    depth_img = io_image.read_RGB_image(filepath)
    return depth_img