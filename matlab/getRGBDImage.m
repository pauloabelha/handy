function [rgbd_image, colourImage, depthMap, obj_mask] = getRGBDImage(objs, meshes, resolution, plot)
    if ~exist('plot', 'var')
        plot = 0;
    end
    if ~exist('resolution', 'var')
        resolution = [640, 480];
    end
    %% Define camera parameters
    [cameraResolution,cameraIntrinsics,cameraExtrinsics,cameraRotation] = generateCamera(resolution);
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
    labelMap = labelImages{1};
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

