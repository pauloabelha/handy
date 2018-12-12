function [segmMaskRGB] = genSegmMaskRGBFromBin(segmMaskBin, colorRGB, segmMaskRGB)
    if ~exist('segmMaskRGB','var')
       segmMaskRGB = zeros(size(segmMaskBin,1),size(segmMaskBin,2),3); 
    end    
    for i=1:3
        segmMask_color = segmMaskRGB(:,:,i);
        segmMask_color(segmMaskBin == 1) = colorRGB(i);
        segmMaskRGB(:,:,i) = segmMask_color;
    end
end

