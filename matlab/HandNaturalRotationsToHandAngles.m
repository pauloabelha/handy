function [hand_angles] = HandNaturalRotationsToHandAngles(nat_rots)
    hand_angles = zeros(21, 3);
    nat_rots = reshape(nat_rots, [3,5])';
    hand_angles(2, :) = nat_rots(1, :);
    hand_angles(6, :) = nat_rots(2, :);
    hand_angles(10, :) = nat_rots(3, :);
    hand_angles(14, :) = nat_rots(4, :);
    hand_angles(18, :) = nat_rots(5, :);    
end

