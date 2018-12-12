function [rgbd_image, colourImage, depthMap, obj_mask] = GenerateObjImages(obj_transl, obj_rot, mesh, camera, plot)
    if ~exist('plot', 'var')
        plot = 0;
    end    
    
    obj.objectLibraryIndex = 1;
    obj.translationVector = obj_transl;
    obj.rotationMatrix = obj_rot;
    model.obj = obj;
    model.mesh = mesh;
    
    %% Define camera parameters
    cameraResolution = camera.cameraResolution;
    cameraIntrinsics = camera.cameraIntrinsics;
    cameraExtrinsics = camera.cameraExtrinsics;
    %% Generate image
    lightIntensity = 1;
    superSampling = 1;
    deformationPixelTolerance = 0.5;
    [ depthImages, ~, ~, ~, ~, ...
        ~, ~, ~, ~,...
        ~, ~, ~, ...
        shadedTextureImages, ~,objectMasks] = ...
        softwareRenderLabelledSceneImage( {cameraResolution},...
        {cameraIntrinsics}, {cameraExtrinsics},...
        {model.obj}, {}, {model.mesh}, {}, ...
        lightIntensity, superSampling, deformationPixelTolerance);

    depthMap = depthImages{1};  
    colourImage = shadedTextureImages{1}/max(shadedTextureImages{1}(:));
    rgbd_image = zeros(size(depthMap,1),size(depthMap,2),4);
    rgbd_image(:, :, 1:3) = colourImage;
    rgbd_image(:, :, 4) = depthMap;
    obj_mask = objectMasks{1};    
    if plot
        %figure,imshow(depthMap,[]);
        %title('Depth map');        
        figure,imshow(colourImage);
        title('Colour image');
    end
    
    
end

