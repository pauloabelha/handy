function [eul_angles] = EulerAnglesBetweenVectors(u, v, seq)
    if ~exist('seq','var')
        seq = 'XYZ';
    end
    u = u /norm(u);
    v = v / norm(v);
    eul_angles = rotm2eul_(vrrotvec2mat(vrrotvec(u, v)), seq);
end

