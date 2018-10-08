import converter
from debugger import print_verbose
import resnet
import numpy as np
import torch
import argparse
import os
from random import randint
import datetime

def load_checkpoint(filename, model_class, use_cuda=False):
    torch_file = torch.load(filename, map_location=lambda storage, loc: storage)
    '''
    if use_cuda:
        try:
            torch_file = torch.load(filename)
        except:
            torch_file = torch.load(filename, map_location=lambda storage, loc: storage)
            use_cuda = False
    else:
        torch_file = torch.load(filename, map_location=lambda storage, loc: storage)
    '''
    model_state_dict = torch_file['model_state_dict']
    train_vars = torch_file['train_vars']
    params_dict = {}
    params_dict['joint_ixs'] = range(2)
    params_dict['use_cuda'] = use_cuda
    params_dict['cross_entropy'] = True
    if not use_cuda:
        params_dict['use_cuda'] = False
    model = model_class(params_dict)
    model.load_state_dict(model_state_dict)
    if use_cuda:
        model = model.cuda()
    optimizer_state_dict = torch_file['optimizer_state_dict']
    optimizer = torch.optim.Adadelta(model.parameters())
    optimizer.load_state_dict(optimizer_state_dict)
    del optimizer_state_dict, model_state_dict
    return model, optimizer, train_vars, train_vars

def save_final_checkpoint(train_vars, model, optimizer):
    msg = ''
    msg += print_verbose("\nReached final number of iterations: " + str(train_vars['num_iter']), train_vars['verbose'])
    msg += print_verbose("\tSaving final model checkpoint...", train_vars['verbose'])
    if not train_vars['output_filepath'] == '':
        with open(train_vars['output_filepath'], 'a') as f:
            f.write(msg + '\n')
    final_model_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_vars': train_vars,
    }
    save_checkpoint(final_model_dict,
                            filename=train_vars['checkpoint_filenamebase'] +
                                     'final' + str(train_vars['num_iter']) + '.pth.tar')
    train_vars['done_training'] = True
    return train_vars

# initialize control variables
def initialize_train_vars(args):
    train_vars = {}
    train_vars['done_training'] = False
    train_vars['start_epoch'] = 0
    train_vars['losses'] = []
    train_vars['start_iter_mod'] = 1
    train_vars['start_iter'] = 1
    train_vars['num_iter'] = args.num_iter
    train_vars['num_epochs'] = args.num_epochs
    train_vars['best_model_dict'] = 0
    train_vars['log_interval'] = args.log_interval
    train_vars['log_interval_valid'] = args.log_interval_valid
    train_vars['batch_size'] = args.batch_size
    train_vars['max_mem_batch'] = args.max_mem_batch
    train_vars['losses_main'] = []
    train_vars['losses_joints'] = []
    train_vars['best_loss_joints'] = 1e10
    train_vars['total_joints_loss'] = 0
    train_vars['losses_heatmaps'] = []
    train_vars['best_loss_heatmaps'] = 1e10
    train_vars['total_heatmaps_loss'] = 0
    train_vars['pixel_losses'] = []
    train_vars['pixel_losses_sample'] = []
    train_vars['best_loss'] = 1e10
    train_vars['best_pixel_loss'] = 1e10
    train_vars['best_pixel_loss_sample'] = 1e10
    train_vars['best_model_dict'] = {}
    train_vars['heatmap_ixs'] = args.heatmap_ixs
    train_vars['use_cuda'] = args.use_cuda
    train_vars['cross_entropy'] = False
    train_vars['root_folder'] = os.path.dirname(os.path.abspath(__file__)) + '/'
    train_vars['checkpoint_filenamebase'] = 'trained_net_log_'
    train_vars['iter_size'] = int(args.batch_size / args.max_mem_batch)
    train_vars['n_iter_per_epoch'] = 0
    train_vars['done_training'] = False
    train_vars['tot_toc'] = 0
    train_vars['output_filepath'] = args.output_filepath
    train_vars['verbose'] = args.verbose
    return train_vars

def load_resnet_weights_into_HALNet(halnet, verbose, n_tabs=1):
    print_verbose("Loading RESNet50...", verbose, n_tabs)
    resnet50 = resnet.resnet50(pretrained=True)
    print_verbose("Done loading RESNet50", verbose, n_tabs)
    # initialize HALNet with RESNet50
    print_verbose("Initializaing network with RESNet50...", verbose, n_tabs)
    # initialize level 1
    # initialize conv1
    resnet_weight = resnet50.conv1.weight.data.cpu()
    float_tensor = np.random.normal(np.mean(resnet_weight.numpy()),
                                    np.std(resnet_weight.numpy()),
                                    (resnet_weight.shape[0],
                                     1, resnet_weight.shape[2],
                                     resnet_weight.shape[2]))
    resnet_weight_numpy = resnet_weight.numpy()
    resnet_weight = np.concatenate((resnet_weight_numpy, float_tensor), axis=1)
    resnet_weight = torch.FloatTensor(resnet_weight)
    halnet.conv1[0]._parameters['weight'].data.copy_(resnet_weight)
    # initialize level 2
    # initialize res2a
    resnet_weight = resnet50.layer1[0].conv1.weight.data
    halnet.res2a.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[0].conv2.weight.data
    halnet.res2a.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[0].conv3.weight.data
    halnet.res2a.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[0].downsample[0].weight.data
    halnet.res2a.left_res[0]._parameters['weight'].data.copy_(resnet_weight)
    # initialize res2b
    resnet_weight = resnet50.layer1[1].conv1.weight.data
    halnet.res2b.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[1].conv2.weight.data
    halnet.res2b.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[1].conv3.weight.data
    halnet.res2b.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
    # initialize res2c
    resnet_weight = resnet50.layer1[2].conv1.weight.data
    halnet.res2c.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[2].conv2.weight.data
    halnet.res2c.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[2].conv3.weight.data
    halnet.res2c.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
    # initialize res3a
    resnet_weight = resnet50.layer2[0].conv1.weight.data
    halnet.res3a.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[0].conv2.weight.data
    halnet.res3a.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[0].conv3.weight.data
    halnet.res3a.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[0].downsample[0].weight.data
    halnet.res3a.left_res[0]._parameters['weight'].data.copy_(resnet_weight)
    # initialize res3b
    resnet_weight = resnet50.layer2[1].conv1.weight.data
    halnet.res3b.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[1].conv2.weight.data
    halnet.res3b.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[1].conv3.weight.data
    halnet.res3b.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
    # initialize res3c
    resnet_weight = resnet50.layer2[2].conv1.weight.data
    halnet.res3c.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[2].conv2.weight.data
    halnet.res3c.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[2].conv3.weight.data
    halnet.res3c.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
    print_verbose("Done initializaing network with RESNet50", verbose, n_tabs)
    print_verbose("Deleting resnet from memory", verbose, n_tabs)
    del resnet50
    return halnet

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("\tSaving a checkpoint...")
    torch.save(state, filename)

def pixel_stdev(norm_heatmap):
    mean_norm_heatmap = np.mean(norm_heatmap)
    stdev_norm_heatmap = np.std(norm_heatmap)
    lower_bound = mean_norm_heatmap - stdev_norm_heatmap
    upper_bound = mean_norm_heatmap + stdev_norm_heatmap
    pixel_count_lower = np.where(norm_heatmap >= lower_bound)
    pixel_count_upper = np.where(norm_heatmap <= upper_bound)
    pixel_count_mask = pixel_count_lower and pixel_count_upper
    return np.sqrt(norm_heatmap[pixel_count_mask].size)

def print_target_info(target):
    if len(target.shape) == 4:
        target = target[0, :, :, :]
    target = converter.convert_torch_dataoutput_to_canonical(target.data.numpy()[0])
    norm_target = converter.normalize_output(target)
    # get joint inference from max of heatmap
    max_heatmap = np.unravel_index(np.argmax(norm_target, axis=None), norm_target.shape)
    print("Heamap max: " + str(max_heatmap))
    # data_image = visualize.add_squares_for_joint_in_color_space(data_image, max_heatmap, color=[0, 50, 0])
    # sample from heatmap
    heatmap_sample_flat_ix = np.random.choice(range(len(norm_target.flatten())), 1, p=norm_target.flatten())
    heatmap_sample_uv = np.unravel_index(heatmap_sample_flat_ix, norm_target.shape)
    heatmap_mean = np.mean(norm_target)
    heatmap_stdev = np.std(norm_target)
    print("Heatmap mean: " + str(heatmap_mean))
    print("Heatmap stdev: " + str(heatmap_stdev))
    print("Heatmap pixel standard deviation: " + str(pixel_stdev(norm_target)))
    heatmap_sample_uv = (int(heatmap_sample_uv[0]), int(heatmap_sample_uv[1]))
    print("Heatmap sample: " + str(heatmap_sample_uv))

def print_header_info(model, dataset_loader, train_vars):
    msg = ''
    msg += print_verbose("-----------------------------------------------------------", train_vars['verbose']) + "\n"
    msg += print_verbose("Output filenamebase: " + train_vars['output_filepath'], train_vars['verbose']) + "\n"
    msg += print_verbose("Model info", train_vars['verbose']) + "\n"
    try:
        heatmap_ixs = model.heatmap_ixs
        msg += print_verbose("Joints indexes: " + str(heatmap_ixs), train_vars['verbose']) + "\n"
        msg += print_verbose("Number of joints: " + str(len(heatmap_ixs)), train_vars['verbose']) + "\n"
    except:
        msg += print_verbose("Joints indexes: " + str(model.num_heatmaps), train_vars['verbose']) + "\n"
    msg += print_verbose("-----------------------------------------------------------", train_vars['verbose']) + "\n"
    msg += print_verbose("Max memory batch size: " + str(train_vars['max_mem_batch']), train_vars['verbose']) + "\n"
    msg += print_verbose("Length of dataset (in max mem batch size): " + str(len(dataset_loader)),
                         train_vars['verbose']) + "\n"
    msg += print_verbose("Training batch size: " + str(train_vars['batch_size']), train_vars['verbose']) + "\n"
    msg += print_verbose("Starting epoch: " + str(train_vars['start_epoch']), train_vars['verbose']) + "\n"
    msg += print_verbose("Starting epoch iteration: " + str(train_vars['start_iter_mod']),
                         train_vars['verbose']) + "\n"
    msg += print_verbose("Starting overall iteration: " + str(train_vars['start_iter']),
                         train_vars['verbose']) + "\n"
    msg += print_verbose("-----------------------------------------------------------", train_vars['verbose']) + "\n"
    msg += print_verbose("Number of iterations per epoch: " + str(train_vars['n_iter_per_epoch']),
                         train_vars['verbose']) + "\n"
    msg += print_verbose("Number of iterations to train: " + str(train_vars['num_iter']),
                         train_vars['verbose']) + "\n"
    msg += print_verbose("Approximate number of epochs to train: " +
                         str(round(train_vars['num_iter'] / train_vars['n_iter_per_epoch'], 1)),
                         train_vars['verbose']) + "\n"
    msg += print_verbose("-----------------------------------------------------------", train_vars['verbose']) + "\n"

    if not train_vars['output_filepath'] == '':
        with open(train_vars['output_filepath'], 'w+') as f:
            f.write(msg + '\n')

def print_log_info(model, optimizer, epoch, train_vars, save_best=True, save_a_checkpoint=True):
    vars = train_vars
    total_loss = train_vars['total_loss']
    model_class_name = type(model).__name__
    verbose = train_vars['verbose']
    print_verbose("", verbose)
    print_verbose("-------------------------------------------------------------------------------------------", verbose)
    if save_a_checkpoint:
        print_verbose("Saving checkpoints:", verbose)
        print_verbose("-------------------------------------------------------------------------------------------",  verbose)
        checkpoint_model_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_vars': train_vars,
        }
        save_checkpoint(checkpoint_model_dict, filename=vars['checkpoint_filenamebase'] + '.pth.tar')

    msg = ''
    msg += print_verbose("-------------------------------------------------------------------------------------------",
                         verbose) + "\n"
    now = datetime.datetime.now()
    msg += print_verbose('Time: ' + now.strftime("%Y-%m-%d %H:%M"), verbose) + "\n"

    msg += print_verbose("-------------------------------------------------------------------------------------------",
                         verbose) + "\n"

    msg += print_verbose("-------------------------------------------------------------------------------------------",
                         verbose) + "\n"
    msg += print_verbose('Training (Epoch #' + str(epoch) + ' ' + str(train_vars['curr_epoch_iter']) + '/' + \
                         str(train_vars['tot_epoch']) + ')' + ', (Batch ' + str(train_vars['batch_idx'] + 1) + \
                         '(' + str(train_vars['iter_size']) + ')' + '/' + \
                         str(train_vars['num_batches']) + ')' + ', (Iter #' + str(train_vars['curr_iter']) + \
                         '(' + str(train_vars['batch_size']) + ')' + \
                         ' - log every ' + str(train_vars['log_interval']) + ' iter): ', verbose) + '\n'
    msg += print_verbose("-------------------------------------------------------------------------------------------",
                         verbose) + "\n"
    msg += print_verbose("Current loss: " + str(total_loss), verbose) + "\n"
    msg += print_verbose("Best loss: " + str(vars['best_loss']), verbose) + "\n"
    msg += print_verbose("Mean total loss: " + str(np.mean(vars['losses'])), verbose) + "\n"
    msg += print_verbose("Mean loss for last " + str(train_vars['log_interval']) +
                         " iterations (average total loss): " + str(
        np.mean(vars['losses'][-train_vars['log_interval']:])), verbose) + "\n"

    msg += print_verbose("-------------------------------------------------------------------------------------------",
                         verbose) + "\n"


    if not train_vars['output_filepath'] == '':
        with open(train_vars['output_filepath'], 'a') as f:
            f.write(msg + '\n')

    return 1


def get_vars(model_class):
    RANDOM_ID = randint(1000000000, 2000000000)

    model, optimizer, train_vars, train_vars = parse_args(model_class=model_class, random_id=RANDOM_ID)
    if not train_vars['output_filepath'] == '':
        output_split_name = train_vars['output_filepath'].split('.')
        train_vars['output_filepath'] = output_split_name[0] + '_' + str(model_class.__name__) + '_' +\
                                          str(RANDOM_ID) + '.' + output_split_name[1]
    if model_class.__name__ == 'JORNet':
        train_vars['crop_hand'] = True
    else:
        train_vars['crop_hand'] = False
    return model, optimizer, train_vars


def run_until_curr_iter(batch_idx, train_vars):
    if train_vars['curr_epoch_iter'] < train_vars['start_iter_mod']:
        msg = ''
        if batch_idx % train_vars['iter_size'] == 0:
            msg += print_verbose("\rGoing through iterations to arrive at last one saved... " +
                                 str(int(train_vars['curr_epoch_iter'] * 100.0 / train_vars[
                                     'start_iter_mod'])) + "% of " +
                                 str(train_vars['start_iter_mod']) + " iterations (" +
                                 str(train_vars['curr_epoch_iter']) + "/" + str(train_vars['start_iter_mod']) + ")",
                                 train_vars['verbose'], n_tabs=0, erase_line=True)
            train_vars['curr_epoch_iter'] += 1
            train_vars['curr_iter'] += 1
            train_vars['curr_epoch_iter'] += 1
        if not train_vars['output_filepath'] == '':
            with open(train_vars['output_filepath'], 'a') as f:
                f.write(msg + '\n')
        return False, train_vars
    return True, train_vars