function [depth_img] = FPAReadDepthImg(depth_img_filepath)
    depth_img = imread(depth_img_filepath);
    depth_img = double(depth_img) + 1;
    depth_img = depth_img / max(depth_img(:));
end

