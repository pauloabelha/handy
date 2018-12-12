function [bone_lengths] = GetHandBoneLengths(hand_pose)
    bone_lengths = zeros(1, 20);
    bone_idx = 1;
    for i=1:5
        joint_idx = (i * 4) - 2;
        bone_lengths(bone_idx) = norm(hand_pose(joint_idx, :) - hand_pose(1, :));
        bone_idx = bone_idx + 1;
        for j=joint_idx:joint_idx+2
            bone_lengths(bone_idx) = norm(hand_pose(j+1, :) - hand_pose(j, :));
            bone_idx = bone_idx + 1;
        end
    end
    
    
    
    
    
end

