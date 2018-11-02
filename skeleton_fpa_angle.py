from skeleton import *
import visualize as vis

fingers_angles = get_fingers_angles_canonical()

dataset_root_folder = 'C:/Users/Administrator/Documents/Datasets/fpa_benchmark/'

subpath = 'Subject_1/charge_cell_phone/1/'
frame = 0

fpa_skeleton = read_fpa_skeleton(dataset_root_folder, subpath, frame)
fpa_skeleton = fpa_skeleton - fpa_skeleton[0, :]

bone_lengths = get_fpa_bone_lengths(fpa_skeleton)


print(bone_lengths)

fpa_skeleton_angles = get_fpa_skeleton_angles(fpa_skeleton)
print(fpa_skeleton_angles[0:3])
print(fpa_skeleton_angles[3:].reshape((5, 4)))

vis.plot_3D_joints(fpa_skeleton)
vis.show()

bone_lengths2 = list([[]] * 5)
for i in range(5):
    bone_lengths2[i] = bone_lengths[i*4:i*4 + 4]
hand_matrix = Theta_to_hand_matrix(fpa_skeleton_angles, bone_lengths2, fingers_angles)
plot_hand_matrix(hand_matrix)