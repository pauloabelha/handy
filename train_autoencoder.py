import sys
sys.path.append('../VQ-VAE')
from auto_encoder2 import VQ_CVAE
import argparse
from torch import optim
from torchvision import transforms
import fpa_dataset

parser = argparse.ArgumentParser(description='Train an autoencoder for hand depth image reconstruction')
parser.add_argument('-r', dest='dataset_root_folder', required=True, help='Root folder for dataset')
parser.add_argument('--split-filename', default='', help='Dataset split filename')
parser.add_argument('-e', dest='num_epochs', type=int, required=True,
                    help='Total number of epochs to train')
parser.add_argument('--use-cuda', dest='use_cuda', action='store_true', default=False,
                    help='Whether to use cuda for training')
parser.add_argument('-l', dest='epoch_log', type=int, default=10,
                    help='Total number of epochs to train')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size')

args = parser.parse_args()
args.use_cuda = True

transform_depth = transforms.Compose([transforms.ToTensor()])

lr = 2e-4
d = 128
k = 256
num_channels_in = 1
num_channels_out = 1

model = VQ_CVAE(d=d, k=k, num_channels_in=num_channels_in, num_channels_out=num_channels_out)
if args.use_cuda:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.5)

train_loader = fpa_dataset.DataLoaderTracking(root_folder=args.dataset_root_folder,
                                      type='train', transform_color=None,
                                      transform_depth=transform_depth,
                                      batch_size=args.batch_size,
                                      split_filename=args.split_filename,)

for epoch_idx in range(args.num_epochs - 1):
    epoch = epoch_idx + 1
    continue_batch_end_ix = -1
    for batch_idx, (data, _) in enumerate(train_loader):
        if batch_idx < continue_batch_end_ix:
            print('Continuing... {}/{}'.format(batch_idx, continue_batch_end_ix))
            continue
        optimizer.zero_grad()
        if args.use_cuda:
            data = data.cuda()
        outputs = model(data)
        loss = model.loss_function(data, *outputs)
        loss.backward()
        optimizer.step()
        a = 0