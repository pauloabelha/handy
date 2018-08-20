import torch
import torch.nn as nn
import torch.optim as optim
import fpa_io
from lstm_baseline import LSTMBaseline
import numpy as np
import argparse

train = True
load_model = True

parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
parser.add_argument('-r', dest='dataset_root_folder', default='', required=True, help='Root folder for dataset')
parser.add_argument('-e', dest='num_epochs', type=int, required=True,
                    help='Total number of epochs to train')
parser.add_argument('-f', dest='checkpoint_filepath', default='lstm_baseline.pth.tar',
                    help='Checkpoint file path')
parser.add_argument('--use_cuda', dest='use_cuda', action='store_true', default=False,
                    help='Whether to use cuda for training')
parser.add_argument('-o', dest='log_filepath', default='log_lstm_baseline.txt',
                    help='Output file for logging')
parser.add_argument('-l', dest='epoch_log', type=int, default=10,
                    help='Total number of epochs to train')
parser.add_argument('--gt_folder', dest='gt_folder', default='Hand_pose_annotation_v1',
                    help='Folder with Subject groundtruth')
parser.add_argument('--num_joints', type=int, dest='num_joints', default=21, help='Number of joints')

args = parser.parse_args()

def myprint(msg, filepath=None):
    print(msg)
    if not filepath is None:
        with open(filepath, 'a') as f:
            f.write(msg + '\n')

def validate_model(model, dataset_tuples, use_cuda, log_filepath=None):
    accuracies = {}
    for i in range(len(dataset_tuples['valid'])):
        # get input for the model
        subj_name, action_name, seq_ix, joints_seq = dataset_tuples['valid'][i]
        joints_seq = torch.from_numpy(joints_seq).float()
        joints_seq = joints_seq.view(joints_seq.shape[0], 1, -1)

        if use_cuda:
            joints_seq = joints_seq.cuda()

        # feed forward
        model_output = model(joints_seq)
        if use_cuda:
            model_output = model_output.cpu()

        # get action outputs
        action_idx = dataset_tuples['action_to_idx'][action_name]
        model_output_numpy = model_output.data.numpy()
        action_idxs = np.argmax(model_output_numpy, axis=1)
        accuracy = np.sum(action_idxs == action_idx) / action_idxs.shape[0]
        accuracies[(subj_name, action_name, seq_ix)] = accuracy

    accuracies_all = []
    for key in accuracies.keys():
        myprint(str(key) + ' : ' + str(accuracies[key]), log_filepath)
        accuracies_all.append(accuracies[key])

    myprint('Overall mean: ' + str(np.mean(accuracies_all)), log_filepath)
    myprint('Overall stddev: ' + str(np.std(accuracies_all)), log_filepath)
    return accuracies

def save_checkpoint(state, filename='checkpoint.pth.tar', log_filepath=None):
    myprint("\tSaving a checkpoint...", log_filepath)
    torch.save(state, filename)


loss_function = nn.NLLLoss()



actions=['charge_cell_phone', 'clean_glasses',
         'close_juice_bottle', 'close_liquid_soap',
         'close_milk', 'close_peanut_butter',
         'drink_mug', 'flip_pages',
         'flip_sponge', 'give_card']

fpa_io.create_split_file(args.dataset_root_folder, args.gt_folder, '',
                         num_train_seq=2,
                         actions=None)

dataset_tuples = fpa_io.load_split_file(args.dataset_root_folder)

lstm_baseline = LSTMBaseline(num_joints=21,
                             num_actions=dataset_tuples['num_actions'],
                             use_cuda=args.use_cuda)

if args.use_cuda:
    lstm_baseline = lstm_baseline.cuda()

optimizer = optim.Adadelta(lstm_baseline.parameters(), rho=0.9,
                           weight_decay=0.005,
                           lr=0.05)

myprint('Log filepath: ' + str(args.log_filepath), args.log_filepath)
myprint('Checkpoint filepath: ' + str(args.checkpoint_filepath), args.log_filepath)
myprint('Using CUDA: ' + str(args.use_cuda), args.log_filepath)
myprint('Number of epochs: ' + str(args.num_epochs), args.log_filepath)
myprint('Number of training triples: ' + str(len(dataset_tuples['train'])), args.log_filepath)

losses = []
for i in range(len(dataset_tuples['train'])):
    losses.append([])

if train:
    lstm_baseline.train()
    load_model = False
    for epoch_idx in range(args.num_epochs - 1):
        epoch = epoch_idx + 1
        myprint('---------------- Epoch {}/{} ----------------'.format(epoch, args.num_epochs), args.log_filepath)
        for i in range(len(dataset_tuples['train'])):
            # zero out torch gradients
            optimizer.zero_grad()

            # clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            lstm_baseline.init_hidden_states()

            # get input for the model
            subj_name, action_name, seq_ix, joints_seq = dataset_tuples['train'][i]
            joints_seq = torch.from_numpy(joints_seq).float()
            joints_seq = joints_seq.view(joints_seq.shape[0], 1, -1)
            if args.use_cuda:
                joints_seq = joints_seq.cuda()

            # feed forward
            model_output = lstm_baseline(joints_seq)

            # get loss
            action_idx = dataset_tuples['action_to_idx'][action_name]
            target = torch.tensor([action_idx] * model_output.shape[0], dtype=torch.long)
            if args.use_cuda:
                target = target.cuda()
            loss = loss_function(model_output, target)
            losses[i].append(loss.item())
            loss.backward()

            # optimise
            optimizer.step()

        should_log_epoch = epoch % args.epoch_log == 0
        if epoch > 0 and should_log_epoch:
            validate_model(lstm_baseline, dataset_tuples, args.use_cuda, args.log_filepath)
            checkpoint_dict = {
                'model_state_dict': lstm_baseline.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'actions': actions,
                'args': args
            }
            save_checkpoint(checkpoint_dict, filename=args.checkpoint_filepath,
                            log_filepath=args.log_filepath)

if load_model:
    a = 0

validate_model(lstm_baseline, dataset_tuples, log_filepath=args.log_filepath, use_cuda=args.use_cuda)
