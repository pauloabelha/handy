%% Hand pose is a 21 x 3 matrix
% Hand pose is in the following order:
%   wrist_x, wrist_y, wrist_z
%   thumb_mcp_x, thumb_mcp_y, thumb_mcp_z
%   thumb_pip_x, thumb_pip_y, thumb_pip_z
%   thumb_dip_x, thumb_dip_y, thumb_dip_z
%   thumb_tip_x, thumb_tip_y, thumb_tip_z
%   index_mcp_x, index_mcp_y, index_mcp_z
%   ...
% Hand pose values are in mm
% Hand angles is a 21 x 3 matrix with XYZ Euler angles for:
%   wrist XYZ
%   thumb root XYZ
%   thumb mcp XYZ
%   thumb pip XYZ
%   thumb dip XYZ
%   index root XYZ
%   ...
% Hand Angles are in Radians
function [hand_angles] = GetHandAngles(hand_pose)
    hand_pose = hand_pose - hand_pose(1, :);
    hand_angles = zeros(21, 3);
    %% get wrist angles
    hand_angles(1, :) = GetHandWristAngles(hand_pose);
    %% get finger joints angles
    for i=1:5
        root_idx = (i * 4) - 2;
        accum_joint_rot = GetJointRotation(hand_pose, root_idx, 1, inv(eye(3)));
        hand_angles(root_idx, :) = rotm2eul_(accum_joint_rot);
        idx = 1;
        for j=root_idx+1:root_idx+3
            joint_rot = GetJointRotation(hand_pose, j, j-1, inv(accum_joint_rot));
            hand_angles(root_idx+idx, :) = rotm2eul_(joint_rot);
            accum_joint_rot = accum_joint_rot * joint_rot;
            idx = idx + 1;
        end
    end    
end

function [joint_rot] = GetJointRotation(hand_pose, joint_idx, parent_idx, rot_parent)
    joint_vec = [hand_pose(joint_idx, :) - hand_pose(parent_idx, :)]';
    joint_vec_unit = joint_vec/norm(joint_vec);
    joint_vec_rot = rot_parent * joint_vec_unit;
    joint_align_rot = RotationBetweenVectors(joint_vec_rot, [1; 0; 0]);
    joint_rot = inv(joint_align_rot);
end
 
function [wrist_angles] = GetHandWristAngles(hand_pose)
     %% get all three wrist rotations and combine them into one
    % first rotation is as follow:
    % we get the cross product between the root bones for middle and index
    % we align this cross vector with the -1Z axis ([0; 0; -1])
    middle_root_vec = [hand_pose(10, :) - hand_pose(1, :)]';
    middle_root_vec_unit = middle_root_vec/norm(middle_root_vec);
    
    rot_ex_1 = RotationBetweenVectors(middle_root_vec_unit, [1; 0; 0]);
    index_root_vec = [hand_pose(6, :) - hand_pose(1, :)]';
    index_root_vec_unit = index_root_vec/norm(index_root_vec);
    index_root_vec_unit_rot = rot_ex_1 * index_root_vec_unit;
    wrist_rot_3 = eye(3);
    if index_root_vec_unit_rot(2) > 0
        wrist_rot_3 = eul2rotm_([pi, 0, 0]);
    end
    wrist_rot = inv(wrist_rot_3 * rot_ex_1);
    % convert wrist rotation to Euler XYZ angles
    wrist_angles = rotm2eul_(wrist_rot);
end