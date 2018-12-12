function [can_vec] = UnrealVecToCanonical(vec)
    can_vec = vec;
    can_vec(2) = -vec(2);
end

