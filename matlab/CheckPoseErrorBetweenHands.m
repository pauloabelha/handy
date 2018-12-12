function [error_pose] = CheckPoseErrorBetweenHands(bone_lentghs_1,hand_angles_1,bone_lengths_2,hand_angles_2)
    max_diff_bone_lengths = max(abs(bone_lentghs_1 - bone_lengths_2));
    hand_pose_1 = SetHandAngles(bone_lentghs_1, hand_angles_1);
    hand_pose_2 = SetHandAngles(bone_lengths_2, hand_angles_2);
    error_pose = sum(abs(hand_pose_1(:) - hand_pose_2(:))) / numel(hand_pose_1(:));
    if error_pose > max_diff_bone_lengths
        error('Pose error between hands is too large.');
    end
end

