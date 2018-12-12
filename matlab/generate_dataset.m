%% cleaning and path adding
close all;
clear;
clc;

imagenet_root = 'C:\\Users\\Administrator\\Documents\\Datasets\\tiny-imagenet-200\\train\\';
% Get a list of all files and folders in this folder.
files = dir(imagenet_root);
% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];
% Extract only those that are directories.
imagenet_subfolders_temp = files(dirFlags);
imagenet_subfolders_temp = imagenet_subfolders_temp(3:end);
imagenet_subfolders = cell(1, length(imagenet_subfolders_temp));
for k = 1 : length(imagenet_subfolders)
	imagenet_subfolders{k} = imagenet_subfolders_temp(k).name;
end

num_data = 100;
num_folders = numel(imagenet_subfolders);
num_images_per_folder = 450;


addpath(genpath('toolbox_calib'));%http://www.vision.caltech.edu/bouguetj/calib_doc/
addpath(genpath('../enzymes'));%https://github.com/pauloabelha/enzymes
addpath(genpath('softwareRenderer'));

resolution = [480, 640];

% generating image
obj_name = 'crackers_box';
root_path = ['paulo/data/' obj_name '/' 'test/'];


load('ycb_objs.mat');
meshes = {mesh_crackers_box};

for i=1:num_data
    %% get random imagenet image
    random_folder_idx = randi([1 num_folders],1,1);
    imagenet_image_fileroot = [imagenet_root imagenet_subfolders{random_folder_idx} '\\images\\' imagenet_subfolders{random_folder_idx} '_'];
    random_img_idx = num2str(randi([0 num_images_per_folder],1,1));
    imagenet_image_filepath = [imagenet_image_fileroot random_img_idx '.jpeg'];
    imagenet_image = imread(imagenet_image_filepath);
    imagenet_image = imresize(imagenet_image, resolution);  
    
    imagenet_image_fileroot = [imagenet_root imagenet_subfolders{random_folder_idx} '\\images\\' imagenet_subfolders{random_folder_idx} '_'];
    random_img_idx = num2str(randi([0 num_images_per_folder],1,1));
    imagenet_image_filepath = [imagenet_image_fileroot random_img_idx '.jpeg'];
    imagenet_image2 = imread(imagenet_image_filepath);
      
    %% get YCB image
    translation = randomWithinRange(3, -0.2, 0.1).';
    rotation = [randomWithinRange(4, 0, 3.14)].';
    
    obj_crackers_box = getObject(translation, rotation, 1);
    [rgbd_image, colourImage, depthMap, obj_mask]  = getRGBDImage( {obj_crackers_box}, meshes);    
    %% merge images    
    obj_idxs = obj_mask == 1;
    a = uint8(255 * colourImage);    
    if ndims(imagenet_image) == 2
        disp('Found imagenet image with only two dimensions; repeating chanels...');
        imagenet_image = repmat(imagenet_image,1,1,3);
    end
    merged_image = imagenet_image;
    disp(imagenet_image_filepath);
    for j=1:3
        a1 = a(:, :, j);
        disp(size(imagenet_image));
        b1 = imagenet_image(:, :, j);
        b1(obj_idxs == 1) = a1(obj_idxs == 1);
        merged_image(:, :, j) = b1;
    end
    %% write images
    dlmwrite([root_path obj_name '_pose_' num2str(i) '.txt'], [i translation rotation]);
    imwrite(merged_image,[root_path obj_name '_with_imagenet_' num2str(i) '.png']);
    imwrite(colourImage,[root_path obj_name '_colour_' num2str(i) '.png']);
    imwrite(depthMap,[root_path obj_name '_depth_' num2str(i) '.png']);
    imwrite(obj_mask,[root_path obj_name '_mask_' num2str(i) '.png']);
end


