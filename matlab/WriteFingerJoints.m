function WriteFingerJoints(finger_joint_pose, fileID)
    for joint_idx=1:min(3, size(finger_joint_pose,1))
        for dim_idx=1:size(finger_joint_pose,2)
            fprintf(fileID,[num2str(finger_joint_pose(joint_idx, dim_idx)) ',']);
        end
    end
end

