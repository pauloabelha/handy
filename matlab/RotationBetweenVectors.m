function [rot_mtx] = RotationBetweenVectors(u, v)
    u = u / norm(u);
    v = v / norm(v);
    rot_mtx = vrrotvec2mat(vrrotvec(u, v));
end

