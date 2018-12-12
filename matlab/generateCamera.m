function [cameraResolution,cameraIntrinsics,cameraExtrinsics,cameraRotation] = generateCamera(resolution)
    if ~exist('resolution','var')
        resolution = [640,480];
    end
    cameraResolution = resolution;
    cameraIntrinsics.focalLengthValue = [750,750];
    cameraIntrinsics.principalPointValue = [320,240];
    cameraIntrinsics.skewValue = 0;
    cameraIntrinsics.distortionValue = [0,0,0,0,0];
    cameraExtrinsics.translationVectorValue = [0,0,1];
    cameraRotation = [1,1,0,pi/4];
    cameraExtrinsics.rotationMatrix = axang2rotm(cameraRotation);
end

