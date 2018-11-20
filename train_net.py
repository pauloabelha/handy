import argparse
import importlib
import ast
import torch.optim as optim

parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
parser.add_argument('-r', dest='dataset_root_folder', required=True, help='Root folder for dataset')
parser.add_argument('--split-filename', default='', help='Dataset split filename')
parser.add_argument('-e', dest='num_epochs', type=int, required=True,
                    help='Total number of epochs to train')
parser.add_argument('--log-interval', dest='log_interval', type=int, default=100,
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
parser.add_argument('--batch-size', type=int, default=4, help='Batch size')

parser.add_argument('--net', required=True,
                    help='Network model module and class name.'
                         'Example: "reconstruction_net.ReconstructNet"')
parser.add_argument('--net-dict', required=True,
                    help='String representation of a network''s dictionary parameter"')
parser.add_argument('--data-loader', required=True,
                    help='Data loader module and function name.'
                         'Example: "fpa_dataset.FPARGBDReconstruction"')
args = parser.parse_args()

# Start log file
def log_print(msg, log_filepath):
    msg = str(msg) + "\n"
    with open(log_filepath, "a") as myfile:
        myfile.write(msg)
    print(msg)
open(args.log_filepath, 'w').close()

# Print passed arguments
log_print("Arguments: " + str(args), args.log_filepath)

# Network initialization block
# import network class from its module and class name
net_module_str, net_class_str = args.net.split('.')
net_module = importlib.import_module(net_module_str)
NetworkClass = getattr(net_module, net_class_str)
# get network params dict form its string representation
net_params_dict = ast.literal_eval(args.net_dict)
# inititalize network
net_model = NetworkClass(net_params_dict)
log_print("Network params dict:", args.log_filepath)
log_print(net_params_dict, args.log_filepath)
log_print("Network:", args.log_filepath)
log_print(net_model, args.log_filepath)
log_print("Network loaded succesfully", args.log_filepath)

# Data loader block
# import data loader function from its module and function name
data_loader_module_str, data_loader_function_str = args.data_loader.split('.')
data_loader_module = importlib.import_module(data_loader_module_str)
DataLoader = getattr(data_loader_module, data_loader_function_str)
train_loader = DataLoader(root_folder=args.dataset_root_folder,
                          batch_size=args.batch_size)
if train_loader is None:
    log_print("Could not load train loader function (train_loader is None)", args.log_filepath)
    exit(1)
log_print("Data loader: " + str(DataLoader), args.log_filepath)
log_print("Dataset root folder: " + args.dataset_root_folder, args.log_filepath)
log_print("Dataset batch size: " + str(args.batch_size), args.log_filepath)
log_print("Dataset length: " + str(len(train_loader)), args.log_filepath)

# Optimizer initialization block
optimizer = optim.Adadelta(net_model.parameters(),
                           rho=args.momentum,
                           weight_decay=args.weight_decay,
                           lr=args.lr)
log_print("Optimizer loaded: " + str(optimizer), args.log_filepath)

# Training
log_print("Training started", args.log_filepath)
for epoch_idx in range(args.num_epochs):
    for batch_idx, (data, labels) in enumerate(train_loader):
        log_print("Batch: " + str(batch_idx + 1), args.log_filepath)
