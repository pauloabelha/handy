from os import listdir
from os.path import isfile, join
import time

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