close all;
clear;
clc;
%% Add renderer for use generating images
addpath(genpath('../toolbox_calib'));%http://www.vision.caltech.edu/bouguetj/calib_doc/
addpath(genpath('../enzymes'));%https://github.com/pauloabelha/enzymes
addpath(genpath('softwareRenderer'));
%% Generate meshes
%[obj_cube1, mesh_cube1] = generateCube(1, [0 0 0], 1);
%[obj_cube2, mesh_cube2] = generateCube(1, [0.2 0 0], 2);
[obj_crackers_box, mesh_crackers_box] = readPCLToHectorMesh('/home/paulo/hector_code/paulo/ycb/003_cracker_box/google_16k/nontextured.ply', [0,0,0], 1);
obj1 = obj_crackers_box;
mesh1 = mesh_crackers_box;
[obj_bowl, mesh_bowl] = readPCLToHectorMesh('/home/paulo/hector_code/paulo/ycb/024_bowl/google_16k/nontextured.ply', [0.25, 0.25, 0], 2);
objs = {obj_crackers_box,obj_bowl};
meshes = {mesh_crackers_box,mesh_bowl};
%% Define camera parameters
[cameraResolution,cameraIntrinsics,cameraExtrinsics,cameraRotation] = generateCamera();
%% Generate image
lightIntensity = 1;
superSampling = 1;
deformationPixelTolerance = 0.5;
[ depthImages, labelImages, libraryLabelImages, sceneCoordinateImages, objectCoordinateImages, ...
    texturePixelCoordinates, diffuseShadingImages, specularShadingImages, ambientShadingImages,...
    surfaceNormalImages, surfaceReflectionImages, objectTextures, ...
    shadedTextureImages, triangleLabels,objectMasks] = ...
    softwareRenderLabelledSceneImage( {cameraResolution},...
    {cameraIntrinsics}, {cameraExtrinsics},...
    objs, {}, meshes, {}, ...
    lightIntensity, superSampling, deformationPixelTolerance);
depthMap = depthImages{1};
figure,imshow(depthMap,[]);
title('Depth map');
segmMaskRGB = genSegmMaskRGBFromBins(objectMasks{1});
imshow(segmMaskRGB);
title('Segmentation mask');
