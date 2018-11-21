from os import listdir
from os.path import isfile, join
import time

# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/2
class UnNormalize(object):
    def __init__(self, mean, std, img=False):
        self.mean = mean
        self.std = std
        self.img = img

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
            if self.img:
                t.mul_(255)
        return tensor

def cudafy(object, use_cuda):
    if use_cuda:
        return object.cuda()
    else:
        return object

def display_est_time_loop(tot_toc, curr_ix, tot_iter, prefix=''):
    if curr_ix == tot_iter:
        neat_time = time.strftime('%H:%M:%S', time.gmtime(tot_toc))
        print("\r" + prefix + 'Total elapsed time (HH:MM:SS): ' + neat_time, end='')
        print('')
    else:
        avg_toc = tot_toc / curr_ix
        estimated_time_hours = (avg_toc * (tot_iter - curr_ix))
        neat_time = time.strftime('%H:%M:%S', time.gmtime(estimated_time_hours))
        perc = str(round(curr_ix*100/tot_iter))
        print('\r' + prefix + 'Estimated time (HH:MM:SS): ' + neat_time + ' ' + perc + '%', end='')
    return tot_toc

def _print_layer_output_shape(layer_name, output_shape):
    print("Layer " + layer_name + " output shape: " + str(output_shape))

def swap_cols(arr, frm, to):
    arr[:,[frm, to]] = arr[:,[to, frm]]

def myprint(msg, filepath=None):
    print(msg)
    if not filepath is None:
        with open(filepath, 'a') as f:
            f.write(msg + '\n')

def list_files_in_dir(dir):
    return [f for f in listdir(dir) if isfile(join(dir, f))]