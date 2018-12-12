% Hand pose is returned from a given hand model and the required angles
% Hand pose is a 21 x 3 matrix
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
function [hand_pose] = SetHandAngles(bone_lengths, hand_angles)
    %% independently rotate fingers
    thumb = HandRotateFinger(bone_lengths(1:4), hand_angles(2:5, :));  
    index = HandRotateFinger(bone_lengths(5:8), hand_angles(6:9, :)); 
    middle = HandRotateFinger(bone_lengths(9:12), hand_angles(10:13, :)); 
    ring = HandRotateFinger(bone_lengths(13:16), hand_angles(14:17, :)); 
    little = HandRotateFinger(bone_lengths(17:20), hand_angles(18:21, :));     
    %% recompose hand
    hand_pose = [[0 0 0]; thumb; index; middle; ring; little];    
    %% rotate wrist
    %wrist_angles = hand_angles(1, :);
    %wrist_rot = eul2rotm_(wrist_angles);
    %hand_pose = [wrist_rot * hand_pose']';
end

function [finger] = HandRotateFinger(bone_lengths, finger_angles)
    X_vec = [1; 0; 0]; 
    % rotate finger    
    rot = eul2rotm_(finger_angles(1, :));
    finger_root_vec = rot * X_vec * bone_lengths(1);
    rot = rot * eul2rotm_(finger_angles(2, :));
    finger_mcp_vec = rot * X_vec * bone_lengths(2);
    rot = rot * eul2rotm_(finger_angles(3, :));
    finger_pip_vec = rot * X_vec * bone_lengths(3);
    rot = rot * eul2rotm_(finger_angles(4, :));
    finger_dip_vec = rot * X_vec * bone_lengths(4); 
    % position finger
    finger_mcp = finger_root_vec';
    finger_pip = finger_mcp_vec' + finger_mcp;
    finger_dip = finger_pip_vec' + finger_pip;
    finger_tip = finger_dip_vec' + finger_dip;
    finger = [finger_mcp; finger_pip; finger_dip; finger_tip];
end
