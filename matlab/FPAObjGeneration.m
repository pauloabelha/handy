close all;
clc;
%subpath_to_start = '';
meshes_loaded = exist('obj_meshes', 'var');
%% Add renderer for use generating images
addpath(genpath('../toolbox_calib'));%http://www.vision.caltech.edu/bouguetj/calib_doc/
addpath(genpath('../GitHub/enzymes/'));%https://github.com/pauloabelha/enzymes
addpath(genpath('../toolbox_calib'));%http://www.vision.caltech.edu/bouguetj/calib_doc/
addpath(genpath('../../hector_code/extra'));
addpath(genpath('../../hector_code/softwareRenderer'));
dataset_root = 'C:/Users/Administrator/Documents/Datasets/fpa_benchmark/';
gen_folder = 'gen_objs/';
obj_gt_folder = 'Object_6D_pose_annotation_v1/';
obj_models_folder = 'Object_models/';
obj_models_path = [dataset_root obj_models_folder];
%% Generate camera
% colour
camera_colour.cameraResolution = [1920, 1080];
camera_colour.cameraIntrinsics.focalLengthValue = [1395.749023, 1395.749268];
camera_colour.cameraIntrinsics.principalPointValue = [935.732544, 540.681030];
camera_colour.cameraIntrinsics.skewValue = 0;
camera_colour.cameraIntrinsics.distortionValue = [0,0,0,0,0];
camera_colour.cameraExtrinsics.translationVectorValue = [25.7, 1.22, 3.902];
camera_colour.cameraExtrinsics.rotationMatrix =...
    [[0.999988496304 -0.00468848412856 0.000982563360594];...
    [0.00469115935266 0.999985218048 -0.00273845880292];...
    [-0.000969709653873 0.00274303671904 0.99999576807]];
% depth
MAX_DEPTH = 1000; % mm
camera_depth.cameraResolution = [640, 480];
camera_depth.cameraIntrinsics.focalLengthValue = [475.065948, 475.065857];
camera_depth.cameraIntrinsics.principalPointValue = [315.944855, 245.287079];
camera_depth.cameraIntrinsics.skewValue = 0;
camera_depth.cameraIntrinsics.distortionValue = [0,0,0,0,0];
camera_depth.cameraExtrinsics.translationVectorValue = camera_colour.cameraExtrinsics.translationVectorValue;
camera_depth.cameraExtrinsics.rotationMatrix = camera_colour.cameraExtrinsics.rotationMatrix;
%% load meshes
if meshes_loaded
    disp('Meshes have already been loaded.');
else
    obj_names = {'juice_bottle', 'liquid_soap', 'milk', 'salt'};
    disp('Loading all meshes (this will use a parallel pool)...');
    obj_meshes = cell(1, numel(obj_names));
    for i=1:numel(obj_names)
        obj_model_path =  [obj_models_path obj_names{i} '_model/'];
        ply_path = [obj_model_path obj_names{i} '_model.ply'];
        texture_path = [obj_model_path 'texture.jpg'];
        disp([char(9) 'Loading mesh for ''' obj_names{i} '''...']);
        obj_meshes{i} = LoadMeshPaulo( ply_path, texture_path, 'meters' );
        disp([char(9) 'Mesh loaded for ''' obj_names{i} '''...']);
    end
end
%% Generate images
subjectsDirs = GetSubDirsFirstLevelOnly([dataset_root obj_gt_folder]);
obj_ix = -1;
arrived_subpath = 0;
for subj_ix=1:numel(subjectsDirs)
    subjectDir = [subjectsDirs{subj_ix} '/'];
    actionsDirs = GetSubDirsFirstLevelOnly([dataset_root obj_gt_folder subjectDir]);
    for action_ix=1:numel(actionsDirs)
        actionDir = [actionsDirs{action_ix} '/'];
        sequencesDirs = GetSubDirsFirstLevelOnly([dataset_root obj_gt_folder subjectDir actionDir]);
        % get current obj ix to get correct mesh for it
        obj_ix = -1;
        if contains(actionDir,'juice')
            obj_ix = 1;
        end
        if contains(actionDir,'soap')
            obj_ix = 2;
        end
        if contains(actionDir,'milk')
            obj_ix = 3;
        end
        if contains(actionDir,'salt')
            obj_ix = 4;
        end
        if obj_ix < 0
            disp(['Action does not have object mesh associated with it: ' actionDir]);
            continue;
        end
        obj_mesh = obj_meshes{obj_ix};
        for seq_ix=1:numel(sequencesDirs)
            seqDir = [sequencesDirs{seq_ix} '/'];
            subpath = [subjectDir actionDir seqDir];
            if ~arrived_subpath
                if strcmp(subpath_to_start,'')
                    arrived_subpath = 1;
                else
                    if strcmp(subpath,subpath_to_start)
                        disp(['Arrived at starting subpath: ' subpath_to_start]); 
                        arrived_subpath = 1;
                    else
                        disp(['Skipping ' subpath ' until ' subpath_to_start]);
                            continue; 
                    end
                end
            end                        
            obj_gt_path = [dataset_root obj_gt_folder subpath 'object_pose.txt'];
            disp(['Current object groundtruth path: ' obj_gt_path]);
            try
                M = dlmread(obj_gt_path);
            catch
                disp(['WARNING. Problem reading file: ' obj_gt_path]);
                continue;
            end
            gen_folder_path = [dataset_root gen_folder subpath];
            mkdir(gen_folder_path);
            parfor frame=1:size(M,1)
                image_prefix_path = [gen_folder_path num2str(frame - 1) '_'];
                disp(['Processing frame: ' subpath ' ' num2str(frame)]);
                M_curr = M(frame,:);
                obj_transf = reshape(M_curr(2:end), [4, 4]);
                [~, ~, depthImage, depth_obj_mask]  = GenerateObjImages(obj_transf(1:3, 4)',...
                        obj_transf(1:3, 1:3), obj_mesh, camera_depth);  
                A = find(depthImage ~= Inf);
                B = depthImage(A);
                C = [A B];    
                dlmwrite([gen_folder_path num2str(frame - 1) '_depth_ixs.csv'], C);   
                temp = depthImage / MAX_DEPTH;
                depthImage = reshape(temp,[480 640]);
                imwrite(depthImage, [image_prefix_path 'depth.jpg']);
                imwrite(depth_obj_mask, [image_prefix_path 'depth_mask.jpg']);
                % normalise depth
                temp = depthImage(:);
                temp(temp == Inf) = 0;  
                dlmwrite([gen_folder_path num2str(frame - 1) '_depth.csv'], reshape(temp,[480 640]));                
                [~, colourImage, ~, colour_obj_mask]  = GenerateObjImages(obj_transf(1:3, 4)',...
                        obj_transf(1:3, 1:3), obj_mesh, camera_colour);                            
                
                %mat2np(reshape(temp,[480 640]), [gen_folder_path num2str(frame - 1) '_depth.pkl'], 'float64'); 
                
                % write images
                image_name = strrep(subpath,'/','_');
                
                imwrite(colourImage, [image_prefix_path 'colour.jpg']);
                imwrite(colour_obj_mask, [image_prefix_path 'colour_mask.jpg']);
                
                disp(['Done for frame: ' subpath ' ' num2str(frame)]);
            end
        end        
    end        
end

