import argparse
import importlib
import ast
import torch.optim as optim
import time
import datetime
import torch
from torch.autograd import Variable
import numpy as np
import io_image

parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
parser.add_argument('--dataset-dict', required=True,
                    help='String representation of a dataset''s dictionary parameter"')
parser.add_argument('--log-root-folder', default='',
                    help='Root folder for logging results')
parser.add_argument('--log-img-prefix', default='log_img_',
                    help='Prefix for logging images')
parser.add_argument('-e', dest='num_epochs', type=int, required=True,
                    help='Total number of epochs to train')
parser.add_argument('--max-log-images', type=int, default=4,
                    help='Max number of images to log')
parser.add_argument('--log-interval', dest='log_interval', type=int, default=10,
                    help='Intervalwith which to log')
parser.add_argument('-c', dest='checkpoint_filepath', default='lstm_baseline.pth.tar',
                    help='Checkpoint file path')
parser.add_argument('--use-cuda', dest='use_cuda', action='store_true', default=False,
                    help='Whether to use cuda for training')
parser.add_argument('-o', dest='log_filepath', default='log_net.txt',
                    help='Output file for logging')
parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
                    help='Momentum for AdaDelta')
parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=0.005,
                    help='Weight decay for AdaDelta')
parser.add_argument('--lr', dest='lr', type=float, default=0.05,
                    help='Learning rate for AdaDelta')
parser.add_argument('--net', required=True,
                    help='Network model module and class name.'
                         'Example: "reconstruction_net.ReconstructNet"')
parser.add_argument('--net-dict', required=True,
                    help='String representation of a network''s dictionary parameter"')
parser.add_argument('--data-loader', required=True,
                    help='Data loader module and function name.'
                         'Example: "fpa_dataset.FPARGBDReconstruction"')
args = parser.parse_args()

# Return three lists of images for data, label and output (for reconstruction)
def get_imgs_to_save(data, labels, output, max_log_images):
    data_imgs = []
    labels_imgs = []
    output_imgs = []
    num_log_images = min(max_log_images, data.shape[0])
    for i in range(num_log_images):
        data_imgs.append(train_loader.dataset.inv_transform_RGB_img(
            data[0].detach().cpu()))
        labels_imgs.append(train_loader.dataset.inv_transform_RGB_img(
            labels[0].detach().cpu()))
        output_imgs.append(train_loader.dataset.inv_transform_RGB_img(
            output[0].detach().cpu()))
    return data_imgs, labels_imgs, output_imgs

# Save images to file
def save_imgs_to_file(list_imgs, img_filepath):
    grid_res_x = 0
    grid_res_y = 0
    num_rows = len(list_imgs)
    num_cols = len(list_imgs[0])
    for i in range(num_rows):
        grid_res_x += list_imgs[i][0].shape[0]
    for j in range(num_cols):
        grid_res_y += list_imgs[0][j].shape[1]
    grid_num_channels = list_imgs[0][0].shape[2]
    grid = np.zeros((grid_res_x, grid_res_y, grid_num_channels))
    for i in range(num_rows):
        for j in range(num_cols):
            x_curr_res = list_imgs[i][j].shape[0]
            y_curr_res = list_imgs[i][j].shape[1]
            x_start = i * x_curr_res
            x_end = x_start + x_curr_res
            y_start = j * y_curr_res
            y_end = y_start + y_curr_res
            grid[x_start:x_end, y_start:y_end, :] = list_imgs[i][j]
    io_image.save_image(grid, img_filepath)


# Start log file
def log_print(msg, log_filepath):
    msg = str(msg) + "\n"
    with open(log_filepath, "a") as myfile:
        myfile.write(msg)
    print(msg)
open(args.log_filepath, 'w').close()

# Print header
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
log_print("Generated from train_ney.py in Handy repository"
          " (https://github.com/pauloabelha/handy.git)", args.log_filepath)
log_print("Timestamp: " + str(timestamp), args.log_filepath)
log_print("GPU: " + str(torch.cuda.get_device_name(0)), args.log_filepath)

# Print passed arguments
log_print("Arguments: " + str(args), args.log_filepath)

# Network initialization block
# import network class from its module and class name
net_module_str, net_class_str = args.net.split('.')
net_module = importlib.import_module(net_module_str)
NetworkClass = getattr(net_module, net_class_str)
# get network params dict from its string representation
net_params_dict = ast.literal_eval(args.net_dict)
# inititalize network
net_model = NetworkClass(net_params_dict)
if args.use_cuda:
    net_model.cuda()
net_model.train()
log_print("Network params dict: " + str(net_params_dict), args.log_filepath)
log_print("Network loaded: ", args.log_filepath)
log_print(net_model, args.log_filepath)

# Data loader block
# import data loader function from its module and function name
data_loader_module_str, data_loader_function_str = args.data_loader.split('.')
data_loader_module = importlib.import_module(data_loader_module_str)
DataLoader = getattr(data_loader_module, data_loader_function_str)
# get network params dict from its string representation
dataset_params_dict = ast.literal_eval(args.dataset_dict)
dataset_params_dict['type'] = 'train'
log_print("Dataset params dict: " + str(dataset_params_dict), args.log_filepath)
train_loader = DataLoader(dataset_params_dict)
if train_loader is None:
    log_print("Could not load train loader function (train_loader is None)", args.log_filepath)
    exit(1)
log_print("Data loader loaded: " + str(DataLoader), args.log_filepath)
log_print("Dataset length: " + str(len(train_loader)), args.log_filepath)

# Optimizer initialization block
optimizer = optim.Adadelta(net_model.parameters(),
                           rho=args.momentum,
                           weight_decay=args.weight_decay,
                           lr=args.lr)
log_print("Optimizer loaded: " + str(optimizer), args.log_filepath)

# Training
train_vars = {
    'epoch_idx': 0,
    'batch_idx': 0,
    'losses': [],
}
log_print("Training started", args.log_filepath)
for epoch_idx in range(args.num_epochs):
    for batch_idx, (data, labels) in enumerate(train_loader):
        if args.use_cuda:
            data = data.cuda()
            labels = labels.cuda()
        data = Variable(data)
        labels = Variable(labels)
        # Clean optimizer
        optimizer.zero_grad()
        # Forward pass
        output = net_model(data)
        # Calculate loss
        loss = net_model.loss(output, labels)
        train_vars['losses'].append(loss.item())
        # Backprop
        loss.backward()
        optimizer.step()
        # Log image results for the first batch (at every epoch,
        # so we can compare progress)
        if batch_idx == 0:
            data_imgs, labels_imgs, output_imgs = \
                get_imgs_to_save(data, labels, output, args.max_log_images)
            img_filepath = args.log_root_folder + args.log_img_prefix +\
                           str(epoch_idx) + '_0.png'
            save_imgs_to_file([data_imgs, labels_imgs, output_imgs], img_filepath)
        # Log current results
        if batch_idx % args.log_interval == 0:
            # Log message
            log_msg = "Training: Epoch {}, Batch {}, Loss {}, Average (last 10) loss: {}".format(
                epoch_idx, batch_idx, train_vars['losses'][-1],
            np.mean(train_vars['losses'][-10:]))
            log_print(log_msg, args.log_filepath)
            # Log reconstructed images
            data_imgs, labels_imgs, output_imgs = \
                get_imgs_to_save(data, labels, output, args.max_log_images)
            img_filepath = args.log_root_folder + args.log_img_prefix + 'curr.png'
            save_imgs_to_file([data_imgs, labels_imgs, output_imgs], img_filepath)




