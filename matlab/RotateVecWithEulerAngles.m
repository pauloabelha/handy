function [vec] = RotateVecWithEulerAngles(vec,eul_angles)
    vec = eul2rotm_(eul_angles) * vec;
end

