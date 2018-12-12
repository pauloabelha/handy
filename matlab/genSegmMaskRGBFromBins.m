function [segmMaskRGB] = genSegmMaskRGBFromBins(segmMaskBins)
    colors = {[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255]};
    segmMaskRGB = genSegmMaskRGBFromBin(segmMaskBins(:,:,1), colors{1});
    for i=2:size(segmMaskBins,3)
        segmMaskRGB = genSegmMaskRGBFromBin(segmMaskBins(:,:,i), colors{i}, segmMaskRGB);
    end
end

