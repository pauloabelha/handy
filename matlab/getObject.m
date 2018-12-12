function [object] = getObject(translation, rotation, obj_idx)
    object.objectLibraryIndex = obj_idx;
    object.translationVector = translation;
    object.rotationMatrix = axang2rotm(rotation);
end

