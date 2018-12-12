function [angles] = get_angles_finger(finger_pose, hand_root)
	angles = [0, 0, 0, 0];
    finger_bone = finger_pose(1, :)' - hand_root';
    finger_bone =  finger_bone / norm(finger_bone);
    rot_finger_bone_to_X = vrrotvec2mat(vrrotvec(finger_bone, [1; 0; 0])); 
    mcp = finger_pose(2, :)' - finger_pose(1, :)';
    mcp = mcp / norm(mcp);    
    mcp_x = rot_finger_bone_to_X * mcp;    
    mcp_eul_angles = euler_between_vectors([1; 0; 0], mcp_x);   
    %disp(mcp_eul_angles);
    angles(1:2) = [-mcp_eul_angles(3), -mcp_eul_angles(2)] * 180.0 / 3.14159;        
    for j=1:2
        u = finger_pose(j+1, :)' - finger_pose(j, :)';
        v = finger_pose(j+2, :)' - finger_pose(j+1, :)';
        angles(j + 2) = AngleBetweenVectors(u, v) * 180. / pi;
    end
end

