import torch
import torch.nn as nn
import torch.optim as optim
import fpa_io
from lstm_baseline import LSTMBaseline
import numpy as np


def validate_model(model, dataset_tuples):
    accuracies = {}
    for i in range(len(dataset_tuples['valid'])):
        # get input for the model
        subj_name, action_name, seq_ix, joints_seq = dataset_tuples['valid'][i]
        joints_seq = torch.from_numpy(joints_seq).float()
        joints_seq = joints_seq.view(joints_seq.shape[0], 1, -1)

        # feed forward
        model_output = model(joints_seq)

        # get action outputs
        action_idx = dataset_tuples['action_to_idx'][action_name]
        model_output_numpy = model_output.data.numpy()
        action_idxs = np.argmax(model_output_numpy, axis=1)
        accuracy = np.sum(action_idxs == action_idx) / action_idxs.shape[0]
        accuracies[(subj_name, action_name, seq_ix)] = accuracy

    accuracies_all = []
    for key in accuracies.keys():
        print('{} : {}'.format(key, accuracies[key]))
        accuracies_all.append(accuracies[key])

    print('Overall mean: {}'.format(np.mean(accuracies_all)))
    print('Overall stddev: {}'.format(np.std(accuracies_all)))
    return accuracies

train = True
load_model = True
num_epochs = 1000
epoch_log = 10

loss_function = nn.NLLLoss()

num_joints = 21
dataset_root_folder = '/home/paulo/fpa_benchmark'
gt_folder = 'Hand_pose_annotation_v1'
data_folder = 'video_files'

actions=['charge_cell_phone', 'clean_glasses',
         'close_juice_bottle', 'close_liquid_soap',
         'close_milk', 'close_peanut_butter',
         'drink_mug', 'flip_pages',
         'flip_sponge', 'give_card']

fpa_io.create_split_file(dataset_root_folder, gt_folder, data_folder,
                             num_train_seq=2,
                             actions=None)

dataset_tuples = fpa_io.load_split_file(dataset_root_folder)

lstm_baseline = LSTMBaseline(num_joints=21, num_actions=dataset_tuples['num_actions'])
optimizer = optim.Adadelta(lstm_baseline.parameters(),
                               rho=0.9,
                               weight_decay=0.005,
                               lr=0.05)

print('Number of actions: {}'.format(dataset_tuples['num_actions']))

losses = []
for i in range(len(dataset_tuples['train'])):
    losses.append([])

if train:
    lstm_baseline.train()
    load_model = False
    for epoch_idx in range(num_epochs - 1):
        epoch = epoch_idx + 1
        print('---------------- Epoch {}/{} ----------------'.format(epoch, num_epochs))
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

            # feed forward
            model_output = lstm_baseline(joints_seq)

            # get loss
            action_idx = dataset_tuples['action_to_idx'][action_name]
            target = torch.tensor([action_idx] * model_output.shape[0], dtype=torch.long)
            loss = loss_function(model_output, target)
            losses[i].append(loss.item())
            loss.backward()

            # optimise
            optimizer.step()

        should_log_epoch = epoch % epoch_log == 0
        if epoch > 0 and should_log_epoch:
            validate_model(lstm_baseline, dataset_tuples)
            #for i in range(len(dataset_tuples['train'])):
            #    subj_name, action_name, seq_ix, joints_seq = dataset_tuples['train'][i]
            #    action_idx = dataset_tuples['action_to_idx'][action_name]
            #    print('{}\t{}\t\t\t{}\tLoss {}'.format(subj_name, action_name, seq_ix, np.mean(losses[i][-epoch_log:])))


if load_model:
    a = 0

validate_model(lstm_baseline, dataset_tuples)
