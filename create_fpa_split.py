import argparse
import fpa_io

parser = argparse.ArgumentParser(description='Create FPA paper splits')
parser.add_argument('-r', dest='dataset_root_folder', required=True, help='Root folder for dataset')
parser.add_argument('-v', dest='video_folder', default='Video_files/', help='Root folder for video (image) files')
parser.add_argument('--fpa-subj-split', default=False, action='store_true',
                    help='Whether to use the FPA paper cross-subject split')
parser.add_argument('--fpa-obj-split', default=False, action='store_true',
                    help='Whether to use the FPA paper cross-object split')
parser.add_argument('--all', default=False, action='store_true',
                    help='Create all splits')

args = parser.parse_args()

if args.all:
    fpa_io.create_split_file(args.dataset_root_folder,
                         args.video_folder,
                         perc_train=0.7, perc_valid=0.15,
                         only_with_obj_pose=False,
                         fpa_subj_split=True,
                         fpa_obj_split=False,
                         split_filename='fpa_split.p')
    fpa_io.create_split_file(args.dataset_root_folder,
                             args.video_folder,
                             perc_train=0.7, perc_valid=0.15,
                             only_with_obj_pose=False,
                             fpa_subj_split=False,
                             fpa_obj_split=True,
                             split_filename='fpa_split.p')
    fpa_io.create_split_file(args.dataset_root_folder,
                             args.video_folder,
                             perc_train=0.7, perc_valid=0.15,
                             only_with_obj_pose=False,
                             fpa_subj_split=False,
                             fpa_obj_split=False,
                             split_filename='fpa_split.p')
else:
    fpa_io.create_split_file(args.dataset_root_folder,
                             args.video_folder,
                             perc_train=0.7, perc_valid=0.15,
                             only_with_obj_pose=False,
                             fpa_subj_split=args.fpa_subj_split,
                             fpa_obj_split=args.fpa_obj_split,
                             split_filename='fpa_split.p')