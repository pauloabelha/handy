from torchvision import transforms
import argparse
import fpa_dataset
from hand_detection_net import HALNet
import losses as my_losses
import trainer
import visualize as vis
import numpy as np

parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
parser.add_argument('-r', dest='dataset_root_folder', required=True, help='Root folder for dataset')
parser.add_argument('-c', dest='checkpoint_filename', required=True, help='Checkpoint filename')
parser.add_argument('--split-filename', default='', help='Dataset split filename')
parser.add_argument('--use-cuda', dest='use_cuda', action='store_true', default=False,
                    help='Whether to use cuda')

args = parser.parse_args()
args.batch_size = 1

model, _, _, _ = trainer.load_checkpoint(args.checkpoint_filename, HALNet, use_cuda=True)
if args.use_cuda:
    model.cuda()

transform_color = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.5, 0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5, 0.5))])

transform_depth = transforms.Compose([transforms.ToTensor()])

train_loader = fpa_dataset.DataLoaderTracking(root_folder=args.dataset_root_folder,
                                      type='test', transform_color=transform_color,
                                              transform_depth=transform_depth,
                                      batch_size=args.batch_size,
                                      split_filename=args.split_filename,)

print('Length of dataset: {}'.format(len(train_loader.dataset)))

train_vars = {
        'iter_size': 1,
        'total_loss': 0,
        'verbose': True,
        'checkpoint_filenamebase': 'checkpoint',
        'checkpoint_filename': 'checkpoint.pth.tar',
        'tot_iter': len(train_loader),
        'num_batches': len(train_loader),
        'curr_iter': 0,
        'batch_size': args.batch_size,
        'best_loss': 1e10,
        'losses': [],
        'output_filepath': 'log.txt',
}

loss_func = my_losses.cross_entropy_loss_p_logq
losses = []
for i in range(len(train_loader.dataset)):
    losses.append([])

for batch_idx, (data, label_heatmaps) in enumerate(train_loader):
    train_vars['batch_idx'] = batch_idx
    train_vars['curr_iter'] = batch_idx + 1
    if args.use_cuda:
        data = data.cuda()
        label_heatmaps = label_heatmaps.cuda()

    # clear out the hidden state of the LSTM,
    # detaching it from its history on the last instance.
    output = model(data)

    loss = my_losses.calculate_loss_HALNet(loss_func,
                                           output, label_heatmaps, model.joint_ixs, model.WEIGHT_LOSS_INTERMED1,
                                           model.WEIGHT_LOSS_INTERMED2, model.WEIGHT_LOSS_INTERMED3,
                                           model.WEIGHT_LOSS_MAIN, train_vars['iter_size'])

    heatmaps = output[3].detach().cpu().numpy()[0, :, :, :]
    pixel_idxs = np.zeros((2, 2))
    pixel_idxs[0, :] = np.unravel_index(np.argmax(heatmaps[0]), (640, 480))
    print(pixel_idxs[0, :])
    pixel_idxs[1, :] = np.unravel_index(np.argmax(heatmaps[1]), (640, 480))
    print(pixel_idxs[1, :])
    fig = vis.plot_image(data.cpu().numpy()[0, 0, :, :])
    vis.plot_bound_box(pixel_idxs, fig=fig)
    vis.show()
    del heatmaps

    train_vars['total_loss'] = loss.item()
    print(loss.item())
    train_vars['losses'].append(train_vars['total_loss'])
    if train_vars['total_loss'] < train_vars['best_loss']:
        train_vars['best_loss'] = train_vars['total_loss']

