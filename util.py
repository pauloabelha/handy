from os import listdir
from os.path import isfile, join

def swap_cols(arr, frm, to):
    arr[:,[frm, to]] = arr[:,[to, frm]]

def myprint(msg, filepath=None):
    print(msg)
    if not filepath is None:
        with open(filepath, 'a') as f:
            f.write(msg + '\n')

def list_files_in_dir(dir):
    return [f for f in listdir(dir) if isfile(join(dir, f))]