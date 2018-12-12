function [unreal_vec] = CanonicaToUnrealVec(can_vec)
    unreal_vec = can_vec;
    unreal_vec(2) = -can_vec(2);
end

