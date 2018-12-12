close all;
clear;
clc;
%% Add renderer for use generating images
addpath(genpath('../toolbox_calib'));%http://www.vision.caltech.edu/bouguetj/calib_doc/
addpath(genpath('../GitHub/enzymes/'));%https://github.com/pauloabelha/enzymes
addpath(genpath('softwareRenderer'));
%% Generate meshess
%[obj_cube1, mesh_cube1] = generateCube(1, [0 0 0], 1);
%[obj_cube2, mesh_cube2] = generateCube(1, [0.2 0 0], 2);
[obj_crackers_box, mesh_crackers_box] = readPCLToHectorMesh('C:/Users/Administrator/Documents/hector_code/paulo/ycb/003_cracker_box/google_16k/nontextured.ply', [0,0,0], 1);
obj1 = obj_crackers_box;
mesh1 = mesh_crackers_box;
[obj_bowl, mesh_bowl] = readPCLToHectorMesh('C:/Users/Administrator/Documents/hector_code/paulo/ycb/024_bowl/google_16k/nontextured.ply', [0.25, 0.25, 0], 2);
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
shadingImage = diffuseShadingImages{1} + specularShadingImages{1} + ambientShadingImages{1};
shadingImage = shadingImage/max(shadingImage(:));
figure,imshow(shadingImage);
title('Shading image');
colourImage = shadedTextureImages{1}/max(shadedTextureImages{1}(:));
figure,imshow(colourImage);
title('Colour image');
%% Choose voxel grid parameters
voxelGridMinimum = [-0.3,-0.3,-0.3];
voxelGridMaximum = [0.3,0.3,0.3];
voxelLengthSize = 0.01;
%% Generate internal samples for object
meshVolume = calculateMeshVolume(mesh1);
averageMeshSampleVolume = voxelLengthSize^3/10;
numberOfInternalSamples = ceil(meshVolume/averageMeshSampleVolume);
objectSamples = generateSamplesFromMesh(mesh1,numberOfInternalSamples);
%% Generate occupancy voxel grids
labelEmptySpace = true;
[ classLabelVoxelGrid, instanceLabelVoxelGrid ] = generateVoxelOccupancyForObjects( {obj1}, {objectSamples},...
   averageMeshSampleVolume, voxelGridMinimum, voxelGridMaximum, voxelLengthSize, labelEmptySpace );
visualiseVoxelGridByClass(classLabelVoxelGrid,0.5);
title('Class label voxel grid');
visualiseVoxelGridByClass(instanceLabelVoxelGrid,0.5);
title('Instance label voxel grid');
%% Generate voxel grid from depth map
labelMap = labelImages{1};
useUniformDistribution = false;
[ depthMapVoxelGrid ] = generateVoxelOccupancyForHardLabelledDepthMap( depthMap,labelMap,...
    cameraIntrinsics, cameraExtrinsics, ...
    voxelGridMinimum, voxelGridMaximum, voxelLengthSize, 1, ...
    labelEmptySpace, useUniformDistribution );
visualiseVoxelGridByClass(depthMapVoxelGrid,0.5);
title('Depth map voxel grid');
%% Make depth map noisy
erosionNoiseToAddEachIteration = 150;
numberOfIterations = 250;
randomNoise = 500;
[ noisyDepthMap ] = addNoiseToDepthMap( depthMap, labelMap, erosionNoiseToAddEachIteration, numberOfIterations, randomNoise );
figure,imshow(noisyDepthMap,[]);
%% Generate noisy voxel grid
[ noisyDepthMapVoxelGrid ] = generateVoxelOccupancyForHardLabelledDepthMap( noisyDepthMap,labelMap,...
    cameraIntrinsics, cameraExtrinsics, ...
    voxelGridMinimum, voxelGridMaximum, voxelLengthSize, 1, ...
    labelEmptySpace, useUniformDistribution );
visualiseVoxelGridByClass(noisyDepthMapVoxelGrid,0.5);
title('Noisy depth map voxel grid');