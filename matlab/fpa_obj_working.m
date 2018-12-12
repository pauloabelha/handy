close all;
clear;
clc;
%% Add renderer for use generating images
addpath(genpath('../toolbox_calib'));%http://www.vision.caltech.edu/bouguetj/calib_doc/
addpath(genpath('../GitHub/enzymes/'));%https://github.com/pauloabelha/enzymes
addpath(genpath('extra'));
addpath(genpath('softwareRenderer'));
dataset_root = 'C:/Users/Administrator/Documents/Datasets/fpa_benchmark/';
obj_models_folder = 'Object_models/';
obj_models_path = [dataset_root obj_models_folder];
obj_model_name = 'juice_bottle_model';
obj_model_path =  [obj_models_path obj_model_name '/'];
%% Generate obj and mesh
ply_path = [obj_model_path obj_model_name '.ply'];
texture_path = [obj_model_path 'texture.jpg'];
disp('Loading mesh...');
[ mesh ] = loadMesh( ply_path, texture_path, 'meters' );
disp('Mesh loaded.');

obj.objectLibraryIndex = 1;
obj.translationVector = [-9.4921104e+01, 5.0128422e+01, 3.9714862e+02];
obj.rotationMatrix = [[ 1.8915400e-01 -9.7259200e-01 -1.3522200e-01];...
                     [-6.5815198e-01 -2.3375001e-02 -7.5252301e-01];...
                     [ 7.2873700e-01  2.3134001e-01 -6.4453501e-01]];
model.obj = obj;
model.mesh = mesh;
%% Generate camera
camera.cameraResolution = [1920, 1080];
camera.cameraIntrinsics.focalLengthValue = [1395.749023, 1395.749268];
camera.cameraIntrinsics.principalPointValue = [935.732544, 540.681030];
camera.cameraIntrinsics.skewValue = 0;
camera.cameraIntrinsics.distortionValue = [0,0,0,0,0];
camera.cameraExtrinsics.translationVectorValue = [25.7, 1.22, 3.902];
camera.cameraExtrinsics.rotationMatrix =...
    [[0.999988496304 -0.00468848412856 0.000982563360594];...
    [0.00469115935266 0.999985218048 -0.00273845880292];...
    [-0.000969709653873 0.00274303671904 0.99999576807]];
%% Generate images
[rgbd_image, colourImage, depthMap, obj_mask]  = GenerateObjImages(model, camera);
imshow(colourImage);