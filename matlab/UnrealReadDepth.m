%%
%
%
%
% Assumes Unreal Pixel Format is PF_B8G8R8A8
% https://answers.unrealengine.com/questions/708727/what-does-devicedepth-option-under-scenecapturecom.html
% https://api.unrealengine.com/INT/API/Runtime/Core/EPixelFormat/index.html
%%
function [depth_img_norm] = UnrealReadDepth(depth_img_filepath, unreal_depth_correspondence, max_unreal_depth)
    
    depth  = getDepth(depth_img_filepath);

    [depth_img, ~, depth_img_A] = imread(depth_img_filepath);   
    depth_img = double(depth_img);
    A = double(depth_img_A);
    R = depth_img(:, :, 1);
    G = depth_img(:, :, 2);
    B = depth_img(:, :, 3);
    depth_img_out = ...
        R +...
        G / 256 +...
        B / 256^2;
    depth_img_out = depth_img_out / 255;
    depths = depth_img_out(:);
    dist_depths = pdist2(depths, unreal_depth_correspondence(:, 2));
    [~, min_dists_idxs] = min(dist_depths, [], 2);
    depth_img_out = reshape(unreal_depth_correspondence(min_dists_idxs, 1),size(depth,1),size(depth,2));
    depth_img_norm = depth_img_out / max_unreal_depth;
end

function [depthMatrix] = getDepth(ImagePath)
    % read the image
    im = imread(ImagePath);
    im = int16(im);
    
    % r -> Red channel
    % g -> Green channel
    % b -> Blue channel
    % f -> Far: the max render distance. 1000 meters in this case.

    % depth = ((r) + (g * 256) + (b * 256*256)) / ((256*256*256) - 1) * f

    depthMatrix = double( im(:,:,1) + (im(:,:,2) * 256) + (im(:,:,3) * 256 * 256) ) / double((256 * 256 * 256) - 1) * 1000;
    
end