%% variables
dataset_root = 'C:/Users/Administrator/Documents/Datasets/fpa_benchmark/';
gen_folder = 'gen_objs/';
image_folder = 'Video_files/';
subject_folder = 'Subject_1/';
action_folder = 'close_liquid_soap/';
seq_folder = '1/';
file_num = 2;
file_name = [num2str(file_num) '_recon.npy'];
filepath_recon = [dataset_root gen_folder subject_folder action_folder seq_folder file_name];

recon_img = readNPY(filepath_recon);
surf(recon_img);

filename_depth = ['depth_000' num2str(file_num) '.png'];
filepath_depth = [dataset_root image_folder subject_folder action_folder seq_folder 'depth/' filename_depth];
depth_img = imread(filepath_depth);
figure; 
surf(depth_img);