function [bone_lengths] = PrintFPABoneLenghts(hand_pose, filepath)
    if ~exist('filepath','var')
        filepath = '';
    end
    fileID = 0;
    if ~strcmp(filepath,'')
        fileID = fopen(filepath,'w');
    end
    joint_conv_idx = [[2, 1]; [2,  7];  [7,   8]; [8, 9];...
                      [3, 1]; [3, 10];  [10, 11]; [11, 12];...
                      [4, 1]; [4, 13];  [13, 14]; [14, 15];...
                      [5, 1]; [5, 16];  [16, 17]; [17, 18];...
                      [6, 1]; [6, 19];  [19, 20]; [20, 21]
                     ];
    bone_names = GetCanonicalFPABoneNames();
    bone_lengths = zeros(20, 1);
    % for each finger
    for i=1:size(joint_conv_idx,1)
        joint1_idx = joint_conv_idx(i, 1);
        joint2_idx = joint_conv_idx(i, 2);
        bone_length = norm(hand_pose(joint1_idx, :) - hand_pose(joint2_idx, :));
        bone_lengths(i) = bone_length;
        %disp([bone_names{i} ' '  num2str(joint1_idx) ' '  num2str(joint2_idx) ' ' num2str(bone_length)]);
        if fileID
            bone_line = [bone_names{i} ',' num2str(bone_length)];
            fprintf(fileID,bone_line);
            if i < size(joint_conv_idx,1)
                fprintf(fileID,'\n');
            end
        end
    end
    if fileID
        fclose(fileID);
    end
end

