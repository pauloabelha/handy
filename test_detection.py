from torchvision import transforms
import argparse
import fpa_dataset
from hand_detection_net import HALNet
import losses as my_losses
import trainer
import visualize as vis
import numpy as np
from scipy.spatial.distance import cdist
import io_image

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
                                      type='test', input_type='depth', transform_color=transform_color,
                                              transform_depth=transform_depth,
                                      batch_size=args.batch_size,
                                      split_filename=args.split_filename,)

len_dataset = len(train_loader.dataset)
print('Length of dataset: {}'.format(len_dataset))

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

losses_pixel = []

def calculate_pixel_loss(out_heatmaps, label_heatmaps):
    output_bbox = np.zeros((2, 2))
    output_bbox[0, :] = np.unravel_index(np.argmax(out_heatmaps[0]), (640, 480))
    output_bbox[1, :] = np.unravel_index(np.argmax(out_heatmaps[1]), (640, 480))
    output_bbox = output_bbox.astype(int)

    label_heatmaps_numpy = label_heatmaps.detach().cpu().numpy()[0, :, :, :]
    label_bbox = np.zeros((2, 2))
    label_bbox[0, :] = np.unravel_index(np.argmax(label_heatmaps_numpy[0]), (640, 480))
    label_bbox[1, :] = np.unravel_index(np.argmax(label_heatmaps_numpy[1]), (640, 480))

    loss_pixel = cdist(label_bbox, output_bbox, 'euclidean')
    loss_pixel = (loss_pixel[0, 0] + loss_pixel[1, 1]) / 2
    return loss_pixel, output_bbox, label_bbox

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

    loss_pixel, label_bbox, out_bbox = calculate_pixel_loss(
        output[3].detach().cpu().numpy()[0, :, :, :], label_heatmaps)
    losses_pixel.append(loss_pixel)

    if batch_idx == 4:
        depth_img = data.cpu().numpy()[0, 0, :, :]
        depth_img_crop = depth_img[label_bbox[0, 0]:label_bbox[1, 0], label_bbox[0, 1]:label_bbox[1, 1]]
        depth_img_crop = io_image.change_res_image(depth_img_crop, ((200, 200)))
        vis.plot_image(depth_img_crop, title=train_loader.dataset.get_img_title(batch_idx))
        vis.show()

        fig = vis.plot_image(depth_img, title=train_loader.dataset.get_img_title(batch_idx))
        #vis.plot_bound_box(label_bbox, fig=fig, color='blue')
        #vis.plot_bound_box(out_bbox, fig=fig, color='red')
        vis.show()
        a = 0

    train_vars['total_loss'] = loss.item()

    print('Batch {} / {}'.format(batch_idx, len_dataset))
    print('Loss (pixel): {}'.format(loss_pixel))
    print('\tMean loss (pixel): {}'.format(np.mean(losses_pixel)))
    print('\tStddev loss (pixel): {}'.format(np.std(losses_pixel)))


    #if batch_idx % 1000 == 0:
    #    losses_pixel_np = np.array(losses_pixel)
    #    a = np.sum((losses_pixel_np < 5.0)) / len(losses_pixel)
    #    b = np.sum((losses_pixel_np < 10.0)) / len(losses_pixel)
    #    vis.plot_histogram(losses_pixel, n_bins=100)
    #    vis.show()
    #    a = 0



