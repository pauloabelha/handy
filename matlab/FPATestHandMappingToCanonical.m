function [errors, errors_to_unreal_hand] = FPATestHandMappingToCanonical()
    MIN_ERROR = 1;
    errors = [];
    errors_to_unreal_hand = [];
    unreal_bone_lengths = GetUnrealMaleVRHandBoneLengths;
    
    msgID = 'MYFUN:BadIndex';
    msg = 'Error is too large';
    baseException = MException(msgID,msg);
    
    dataset_root = 'C:/Users/Administrator/Documents/Datasets/fpa_benchmark/';    
    hand_gt_folder = 'Hand_pose_annotation_v1/';
    subjectsDirs = GetSubDirsFirstLevelOnly([dataset_root hand_gt_folder]);
    for subj_ix=1:numel(subjectsDirs)
        subjectDir = [subjectsDirs{subj_ix} '/'];
        actionsDirs = GetSubDirsFirstLevelOnly([dataset_root hand_gt_folder subjectDir]);
        for action_ix=1:numel(actionsDirs)
            actionDir = [actionsDirs{action_ix} '/'];
            sequencesDirs = GetSubDirsFirstLevelOnly([dataset_root hand_gt_folder subjectDir actionDir]);
            for seq_ix=1:numel(sequencesDirs)
                seqDir = [sequencesDirs{seq_ix} '/'];
                % read hand pose
                hand_gt_filepath = [dataset_root hand_gt_folder subjectDir actionDir seqDir 'skeleton.txt'];
                try
                    M = dlmread(hand_gt_filepath);                    
                catch
                    disp(['WARNING: Could not read file to matrix: ' hand_gt_filepath]); 
                    continue;
                end
                for i=1:size(M,1)
                    fpa_hand_pose = reshape(M(i, 2:end), [3, 21])';
                    hand_pose = FPAHandToCanonical(fpa_hand_pose);                    
                    hand_angles = GetHandAngles(hand_pose);
                    bone_lengths = GetHandBoneLengths(hand_pose);
                    hand_pose_test = SetHandAngles(bone_lengths, hand_angles);
                    hand_pose_test = hand_pose_test + hand_pose(1, :);
                    error = sum(abs(hand_pose(:) - hand_pose_test(:))) / 63;
                    errors(end+1) = error;
                    if error > MIN_ERROR
                        disp(['Error ' num2str(error) ' > ' num2str(MIN_ERROR)]);
                        %PlotHand(hand_pose_test);
                        %throw(baseException);
                    end 
                    hand_pose_unreal = SetHandAngles(unreal_bone_lengths, hand_angles);
                    errors_to_unreal_hand(end+1) = sum(abs(hand_pose(:) - hand_pose_unreal(:))) / 63;
                end                
            end
        end
    end
end

