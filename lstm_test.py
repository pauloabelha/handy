import torch
import torch.nn as nn
import torch.optim as optim
import fpa_io
from lstm_baseline import LSTMBaseline
import numpy as np


num_epochs = 100

loss_function = nn.NLLLoss()

num_joints = 21
dataset_root_folder = '/home/paulo/fpa_benchmark'
gt_folder = 'Hand_pose_annotation_v1'
data_folder = 'video_files'

fpa_io.create_split_file(dataset_root_folder, gt_folder, data_folder, num_train_seq=2)
dataset_tuples = fpa_io.load_split_file(dataset_root_folder)

lstm_baseline = LSTMBaseline(num_joints=21, num_actions=dataset_tuples['num_actions'])
optimizer = optim.SGD(lstm_baseline.parameters(), lr=0.1)

print('Number of actions: {}'.format(dataset_tuples['num_actions']))

losses = []
for i in range(dataset_tuples['num_actions']):
    losses.append([])

for epoch in range(num_epochs):
    print('---------------- Epoch {} ----------------'.format(epoch))
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
        losses[action_idx].append(loss.item())
        loss.backward()
    if epoch > 2:
        for i in range(len(dataset_tuples['train'])):
            subj_name, action_name, seq_ix, joints_seq = dataset_tuples['train'][i]
            action_idx = dataset_tuples['action_to_idx'][action_name]
            print('{}\t{}\t\t\t{}\tLoss {}'.format(subj_name, action_name, seq_ix, np.mean(losses[action_idx][-3:])))

        # optimise
        optimizer.step()
