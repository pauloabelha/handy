import numpy as np
from numpy import genfromtxt
import fpa_io
import visualize as vis

dataset_root = "C:/Users/Administrator/Documents/Datasets/fpa_benchmark/"
hand_gt_folder = "Hand_pose_annotation_v1/"
subject = 'Subject_1'
action = 'close_juice_bottle'
seq = '1/'
frame = 21

unreal_bone_lengths_filepath = dataset_root + 'bonelengths/' + 'UnrealMaleRightHands.txt'
unreal_bone_lengths = genfromtxt(unreal_bone_lengths_filepath, delimiter=',')
unreal_bone_lengths = unreal_bone_lengths[:, 1]

subject_bone_lengths_filepath = dataset_root + 'bonelengths/' + subject + '.txt'
subject_bone_lengths = genfromtxt(subject_bone_lengths_filepath, delimiter=',')
subject_bone_lengths = subject_bone_lengths[:, 1]

bone_prop = (unreal_bone_lengths / subject_bone_lengths).reshape((20,1))
bone_prop[bone_prop < 0] = 1.0

hand_gt_filepath = dataset_root + hand_gt_folder + subject + '/' + action + '/' + seq + 'skeleton.txt'

hand_joints = fpa_io.read_action_joints_sequence(hand_gt_filepath)[int(31)]
hand_joints = hand_joints.reshape((21, 3))
hand_joints -= hand_joints[0, :]
hand_joints_unreal = np.copy(hand_joints)

i = 0
for finger_idx in range(5):
    finger_start_joint_idx = (finger_idx * 4) + 1
    for j in range(3):
        parent_joint_idx = finger_start_joint_idx + j
        parent_joint_before = np.copy(hand_joints_unreal[parent_joint_idx, :])
        curr_bone_prop = bone_prop[parent_joint_idx - 1]
        hand_joints_unreal[parent_joint_idx, :] *= curr_bone_prop
        parent_joint_transl = hand_joints_unreal[parent_joint_idx, :] - parent_joint_before
        print(str(parent_joint_transl) + " " + str(curr_bone_prop) + " " + str(1/curr_bone_prop))
        for k in range(3 - j):
            joint2_idx = parent_joint_idx + k + 1
            hand_joints_unreal[joint2_idx, :] += parent_joint_transl
            print(str(parent_joint_idx) + " " + str(joint2_idx))
            a = 0

fig, ax = vis.plot_3D_joints(hand_joints)
vis.plot_3D_joints(hand_joints_unreal, fig=fig, ax=ax)
vis.show()
