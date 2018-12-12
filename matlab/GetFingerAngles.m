function [finger_angles] = GetFingerAngles(finger_pose, hand_root)
    finger_angles = [0, 0, 0, 0];
    if ~exist('hand_root','var')
        hand_root = [0, 0, 0];
    end    
    % get fingers vectors
    finger_pose_mtx = [hand_root; finger_pose]';    
    finger_vecs = cell(1, 4);
    for i=1:4
        finger_vecs{i} = finger_pose_mtx(:, i+1) - finger_pose_mtx(:, i);
    end
    % get MCP angles
    [eul_angles, rot] = EulerAnglesFingerJoint(finger_vecs, 1);
    finger_angles(1) = -eul_angles(2);
    finger_angles(2) = eul_angles(3);
    % get DIP angle
    dip_vec = finger_pose(4, :) - finger_pose(3, :);
    pip_vec = finger_pose(3, :) - finger_pose(2, :);
    dip_angle = AngleBetweenVectors(pip_vec,dip_vec) * 180 / pi;
    %eul_angles = EulerAnglesFingerJoint(finger_vecs, 2);
    %dip_angle = eul_angles(2);
    finger_angles(3) = dip_angle;
    % get PIP angle
    mcp_vec = finger_pose(2, :) - finger_pose(1, :);
    pip_angle = AngleBetweenVectors(mcp_vec,pip_vec) * 180 / pi;
    %eul_angles = EulerAnglesFingerJoint(finger_vecs, 3);
    %pip_angle = eul_angles(2);
    finger_angles(4) = pip_angle;
    
    %disp(finger_angles);
    %PlotFinger(finger_pose);
end

function [eul_angles_degrees, rot] = EulerAnglesFingerJoint(finger_vecs, joint_idx)
    rot = RotationBetweenVectors(finger_vecs{joint_idx}, [1; 0; 0]);
    rot_bone = rot * (finger_vecs{joint_idx + 1}/norm(finger_vecs{joint_idx + 1}));
    eul_angles = EulerAnglesBetweenVectors([1; 0; 0], rot_bone);
    eul_angles_degrees = eul_angles * 180.0 / pi;
end

