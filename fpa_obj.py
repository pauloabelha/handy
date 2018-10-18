import argparse
import os
import trimesh
from matplotlib import pyplot as plt
from PIL import Image
import fpa_cam
import numpy as np
import visualize as vis

# Loading utilities
def load_objects(obj_root, obj_names):

    all_models = {}
    for obj_name in obj_names:
        obj_path = os.path.join(obj_root, '{}_model'.format(obj_name),
                                '{}_model.ply'.format(obj_name))
        mesh = trimesh.load(obj_path)
        all_models[obj_name] = {
            'verts': np.array(mesh.vertices),
            'faces': np.array(mesh.faces)
        }
    return all_models

def get_obj_transform(subject, action, seq, frame, obj_root):
    seq_path = os.path.join(obj_root, subject, action,
                            str(seq), 'object_pose.txt')
    with open(seq_path, 'r') as seq_f:
        raw_lines = seq_f.readlines()
    raw_line = raw_lines[int(frame)]
    line = raw_line.strip().split(' ')
    trans_matrix = np.array(line[1:]).astype(np.float32)
    trans_matrix = trans_matrix.reshape(4, 4).transpose()
    print('Loading obj transform from {}'.format(seq_path))
    return trans_matrix

dataset_root = "C:/Users/Administrator/Documents/Datasets/fpa_benchmark/"
subject = 'Subject_1'
action = 'pour_juice_bottle'
seq = '3'
frame = 0

obj_gt_root = dataset_root + "Object_6D_pose_annotation_v1/"
obj_models_root = dataset_root + "Object_models"
obj_names = ['juice_bottle', 'liquid_soap', 'milk', 'salt']
obj_name = 'juice_bottle'

# Load objects mesh
objs_info = load_objects(obj_models_root, obj_names)


# Plot everything
fig = plt.figure()

for i in range(99):
    if i < 10:
        frame_num = '000' + str(i)
    else:
        frame_num = '00' + str(i)

    # Load object transform
    obj_transf = get_obj_transform(subject, action, seq, frame_num, obj_gt_root)
    # Get object vertices
    obj_verts = objs_info[obj_name]['verts'] * 1000
    # Apply transform to object
    hom_verts = np.concatenate(
        [obj_verts, np.ones([obj_verts.shape[0], 1])], axis=1)
    verts_trans = obj_transf.dot(hom_verts.T).T
    # Apply camera extrinsic to objec
    verts_camcoords = fpa_cam.cam_color_extr.dot(
        verts_trans.transpose()).transpose()[:, :3]
    # Project and object skeleton using camera intrinsics
    verts_hom2d = np.array(fpa_cam.cam_color_intr).dot(
        verts_camcoords.transpose()).transpose()
    verts_proj = (verts_hom2d / verts_hom2d[:, 2:])[:, :2]

    # Load image and display
    img_path = os.path.join(dataset_root, 'Video_files', subject,
                            action, seq, 'color',
                            'color_{:04d}.jpeg'.format(int(frame_num)))
    print('Loading image from {}'.format(img_path))
    img = Image.open(img_path)
    #img_numpy = np.array(img).T
    #obj_pixels = np.array(verts_proj).astype(int)
    #img_numpy[:, obj_pixels[:, 0], obj_pixels[:, 1]] = 0
    #vis.plot_image(img_numpy)
    #vis.show()

    ax = plt.gca()
    ax.imshow(img, alpha=0.5)
    ax.scatter(verts_proj[:, 0], verts_proj[:, 1], c='r')
    #ax.scatter(verts_camcoords[:, 0], verts_camcoords[:, 1], s=1)

    plt.pause(0.001)
    plt.clf()

plt.show()
