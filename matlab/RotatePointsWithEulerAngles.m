function [mtx] = RotatePointsWithEulerAngles(mtx, eul_angles)
    mtx = [eul2rotm_(eul_angles) * mtx']';
end

