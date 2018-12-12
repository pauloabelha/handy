function [all_hand_poses_can] = FPAWriteHandPoseSequenceAnglesJSON(fpa_dataset_root, fpa_subpath, json_filepath)
    %% General required variables
    json_tab = '  ';
    hand_gt_folder = 'Hand_pose_annotation_v1/';
    unreal_hand_mesh_filepath = '"/Game/Meshes/male_hand_high_r.male_hand_high_r"';
    %% Read FPA hand pose    
    hand_gt_filepath = [fpa_dataset_root hand_gt_folder fpa_subpath 'skeleton.txt'];
    try
        hand_poses_fpa = dlmread(hand_gt_filepath);
    catch
        disp(['WARNING: Could not read FPA hand poses from: ' hand_gt_filepath]); 
        ret = 0;
        return;
    end
    %% Write JSON header
    fileID = fopen(json_filepath,'w');
    fprintf(fileID,'{');
    fprintf(fileID,'\n');
    fprintf(fileID,json_tab);fprintf(fileID,['"SequencePath": "' fpa_subpath '",']);fprintf(fileID,'\n');
    fprintf(fileID,json_tab);fprintf(fileID,['"Mesh": ' unreal_hand_mesh_filepath ',']);fprintf(fileID,'\n');
    fprintf(fileID,json_tab);fprintf(fileID,'"Frames": [');fprintf(fileID,'\n');
    all_hand_poses_can = {};
    % Write hand poses for all frames        
    for i=1:size(hand_poses_fpa,1)
        %% Get fpa canonical hand pose and angles
        hand_pose_fpa = reshape(hand_poses_fpa(i, 2:end), [3, 21])';
        hand_pose_can = FPAHandToCanonical(hand_pose_fpa);  
        hand_pose_can_root = hand_pose_can(1, :);
        hand_pose_can = hand_pose_can - hand_pose_can_root(1, :); 
        all_hand_poses_can{end+1} = hand_pose_can;
        hand_angles_can = GetHandAngles(hand_pose_can);
        hand_angles_can_degrees = hand_angles_can  * 180 / pi;
        %% Get unreal hand pose and check for errors
        % get FPA bone lengths for current subject 
        fpa_subpath_split = strsplit(fpa_subpath, '/');
        subject = fpa_subpath_split{1};
        [bone_lengths_fpa, ~] = FPASubjectHand(subject);
        % get unreal bone lengths and angles
        bone_lengths_unreal = GetUnrealMaleVRHandBoneLengths;
        error_pose = CheckPoseErrorBetweenHands(bone_lengths_fpa,hand_angles_can,bone_lengths_unreal,hand_angles_can);
        %% Map fro the euler angles rotations to hand parameterisation for Unreal
        % The Unreal hand being considered is parameterised by:
        %   Three angles for wrist
        %   Two angles for MCP joint
        %   One angle for PIP joint
        %   One angle for DIP joint
        % In total, there are 3 + 5*4 = 23 angles to parameterise a hand
        hand_angles_unreal = zeros(1, 23);
        % wrist
        hand_angles_unreal(1:3) = hand_angles_can_degrees(1, 1:3);
        hand_angles_unreal(1) = hand_angles_unreal(1);
        hand_angles_unreal(2) = hand_angles_unreal(2);
        hand_angles_unreal(3) = hand_angles_unreal(3);
        % thumb
        hand_angles_unreal(4) = -hand_angles_can_degrees(3, 2);
        hand_angles_unreal(5) = -hand_angles_can_degrees(3, 3) + 90; 
        hand_angles_unreal(6) = -hand_angles_can_degrees(4, 2);
        hand_angles_unreal(7) = -hand_angles_can_degrees(5, 2);        
        hand_angles_unreal(5) = (sign(hand_angles_can_degrees(3, 1))*hand_angles_can_degrees(3, 3));
        hand_angles_unreal(5) = hand_angles_unreal(5) + 110; 
        hand_angles_unreal(5) = hand_angles_unreal(5) / 2;
        hand_angles_unreal(4:7) = 30;
        disp(hand_angles_can_degrees(3:5, :));
        disp(hand_angles_unreal(4:7));
        disp('-------------------------------');        
        % index
        hand_angles_unreal(8) = -hand_angles_can_degrees(7, 2);
        hand_angles_unreal(9) = -hand_angles_can_degrees(7, 3);
        hand_angles_unreal(10) = -hand_angles_can_degrees(8, 2);
        hand_angles_unreal(11) = -hand_angles_can_degrees(9, 2);
        % middle
        hand_angles_unreal(12) = -hand_angles_can_degrees(11, 2);
        hand_angles_unreal(13) = -hand_angles_can_degrees(11, 3);
        hand_angles_unreal(14) = -hand_angles_can_degrees(12, 2);
        hand_angles_unreal(15) = -hand_angles_can_degrees(13, 2);
        % ring
        hand_angles_unreal(16) = -hand_angles_can_degrees(15, 2);
        hand_angles_unreal(17) = -hand_angles_can_degrees(15, 3);
        hand_angles_unreal(18) = -hand_angles_can_degrees(16, 2);
        hand_angles_unreal(19) = -hand_angles_can_degrees(17, 2);
        % little
        hand_angles_unreal(20) = -hand_angles_can_degrees(19, 2);
        hand_angles_unreal(21) = -hand_angles_can_degrees(19, 3);
        hand_angles_unreal(22) = -hand_angles_can_degrees(20, 2);
        hand_angles_unreal(23) = -hand_angles_can_degrees(21, 2);  
        %% Check joint limits
        %UnrealJointWithinLimits(hand_angles_unreal);
        %% Print info to JSON
        fprintf(fileID,[json_tab json_tab]);fprintf(fileID,'{');fprintf(fileID,'\n');    
        fprintf(fileID,[json_tab json_tab json_tab]);fprintf(fileID,['"Frame": ' num2str(i) ',']);fprintf(fileID,'\n');
        fprintf(fileID,[json_tab json_tab json_tab]);fprintf(fileID,'"WristPosition": [');
        % print wrist position (from mm to cm for Unreal)
        wrist_position = hand_pose_can_root(1, :) / 10;
        wrist_position(3) = wrist_position(3);
        for j=1:2
            fprintf(fileID,[num2str(wrist_position(1, j)) ', ']);
        end 
        fprintf(fileID,num2str(wrist_position(1, 3)));
        fprintf(fileID,'],\n');
        % print bone angles
        fprintf(fileID,[json_tab json_tab json_tab]);fprintf(fileID,'"BoneAngles": [');
        for j=1:22
            fprintf(fileID,[num2str(hand_angles_unreal(j)) ', ']);
        end
        fprintf(fileID,num2str(hand_angles_unreal(23)));
        fprintf(fileID,']\n'); 
        if i == size(hand_poses_fpa,1)
            fprintf(fileID,[json_tab json_tab]);fprintf(fileID,'}');fprintf(fileID,'\n');
        else
            fprintf(fileID,[json_tab json_tab]);fprintf(fileID,'},');fprintf(fileID,'\n');
        end
    end
    % print frame closing bracket
    fprintf(fileID,json_tab);fprintf(fileID,']');fprintf(fileID,'\n');
    fprintf(fileID,'}');            
    fclose(fileID);            
end

