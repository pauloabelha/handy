%% variables
dataset_root = 'C:/Users/Administrator/Documents/Datasets/fpa_benchmark/';
gen_root = 'C:/Users/Administrator/Documents/Unreal Projects/UniBhamCV/Saved/Output/SingleHand/';
gen_folder = '';
hand_gt_folder = 'Hand_pose_annotation_v1/';
subject_folder = 'Subject_1/';
action_folder = 'close_juice_bottle/';
seq_folder = '1/';
file_name = 'skeleton.txt';
hand_gt_filepath = [dataset_root hand_gt_folder subject_folder action_folder seq_folder file_name];
image_gt_filepath = [dataset_root 'Video_files/' subject_folder action_folder seq_folder 'depth/depth_' ];
depth_img_filepath = [image_gt_filepath '0051.png'];
out_filepath_folder = [dataset_root 'gen_hands/'];
out_filename = 'HandPoseSequence.json';
disp(hand_gt_filepath);
json_tab = '  ';
num_bone_angles = 23;
%% camera colour transform (depth has same extrinsics as colour)
camera_colour_ext_transl = [25.7, 1.22, 3.902];
camera_colour_ext_rot =...
    [[0.999988496304 -0.00468848412856 0.000982563360594];...
    [0.00469115935266 0.999985218048 -0.00273845880292];...
    [-0.000969709653873 0.00274303671904 0.99999576807]]; 
[camera_depth.orientation, camera_depth.location] = ...
    extrinsicsToCameraPose_(camera_colour_ext_rot,camera_colour_ext_transl);
% from mm to cm
camera_depth.location = camera_depth.location * 0.1;
camera_depth_location_str = ['[' ...
    num2str(camera_depth.location(1)) ', ' ...
    num2str(camera_depth.location(2)) ', ' ...
    num2str(camera_depth.location(3)) ']'];
camera_depth_rotation_euler = rotm2eul_(camera_depth.orientation) * 180 / pi;
camera_depth_rotation_str = ['[' ...
    num2str(camera_depth_rotation_euler(1)) ', ' ...
    num2str(camera_depth_rotation_euler(2)) ', ' ...
    num2str(camera_depth_rotation_euler(3)) ']'];
%% generate poses
mkdir(out_filepath_folder);
%% read poses
thumb_idxs = [2 7 8 9];
index_idxs = [3 10 11 12];
middle_idxs = [4 13 14 15];
ring_idxs = [5 16 17 18];
little_idxs = [6 19 20 21];
subjectsDirs = GetSubDirsFirstLevelOnly([dataset_root hand_gt_folder]);
obj_ix = -1;
arrived_subpath = 0;

num_files_to_process = 0;
handposes_filepaths = {};
for subj_ix=1:numel(subjectsDirs)
    subjectDir = [subjectsDirs{subj_ix} '/'];
    actionsDirs = GetSubDirsFirstLevelOnly([dataset_root hand_gt_folder subjectDir]);
    for action_ix=1:numel(actionsDirs)
        actionDir = [actionsDirs{action_ix} '/'];
        sequencesDirs = GetSubDirsFirstLevelOnly([dataset_root hand_gt_folder subjectDir actionDir]);
        for seq_ix=1:numel(sequencesDirs)
            seqDir = [sequencesDirs{seq_ix} '/'];
            num_files_to_process = num_files_to_process + 1;
            out_filepath = [out_filepath_folder subjectDir actionDir seqDir];
            handposes_filepaths{end+1} = out_filepath;
        end
    end
end
%% create file with list of filepaths with hand poses
handposes_out_filepath = [out_filepath_folder 'HandPosesFilePaths.json'];
fileID = fopen(handposes_out_filepath,'w');
fprintf(fileID,'{');
fprintf(fileID,'\n');
fprintf(fileID,json_tab);fprintf(fileID,'"FilePaths": [');fprintf(fileID,'\n');
for i=1:numel(handposes_filepaths)-1
    fprintf(fileID,[json_tab json_tab '"' handposes_filepaths{i} '",']);  
    fprintf(fileID,'\n');
end
fprintf(fileID,[json_tab json_tab '"' handposes_filepaths{i} '"']);  
fprintf(fileID,'\n');
fprintf(fileID,[json_tab ']']);
fprintf(fileID,'}');
fclose(fileID);
idx_file = 0;

% hand translation (unreal coord system) fromorigin  to hand
unreal_hand_transf = [0, -11, -2.55];

for subj_ix=1:numel(subjectsDirs)
    subjectDir = [subjectsDirs{subj_ix} '/'];
    actionsDirs = GetSubDirsFirstLevelOnly([dataset_root hand_gt_folder subjectDir]);
    for action_ix=1:numel(actionsDirs)
        actionDir = [actionsDirs{action_ix} '/'];
        sequencesDirs = GetSubDirsFirstLevelOnly([dataset_root hand_gt_folder subjectDir actionDir]);
        for seq_ix=1:numel(sequencesDirs)
            idx_file = idx_file + 1;
            seqDir = [sequencesDirs{seq_ix} '/'];
            subpath = [subjectDir actionDir seqDir];
            % read hand pose
            hand_gt_filepath = [dataset_root hand_gt_folder subjectDir actionDir seqDir 'skeleton.txt'];
            try
                M = dlmread(hand_gt_filepath);
            catch
                disp(['WARNING: Could not read file to matrix: ' hand_gt_filepath]); 
                continue;
            end
            
            hand_transl_transf_str = ['"' ...
                'X=' num2str(unreal_hand_transf(1)) ' '...
                'Y=' num2str(unreal_hand_transf(2)) ' '...
                'Z=' num2str(unreal_hand_transf(3)) '"'];
            
            % create output file            
            out_filepath = [out_filepath_folder subjectDir actionDir seqDir];
            disp(hand_gt_filepath);
            mkdir(out_filepath);
            fileID = fopen([out_filepath out_filename],'w');
            fprintf(fileID,'{');
            fprintf(fileID,'\n');
            fprintf(fileID,json_tab);fprintf(fileID,['"SequencePath": "' subpath '",']);fprintf(fileID,'\n');
            fprintf(fileID,json_tab);fprintf(fileID,['"CameraLocation": ' camera_depth_location_str ',']);fprintf(fileID,'\n');
            fprintf(fileID,json_tab);fprintf(fileID,['"CameraRotation": ' camera_depth_rotation_str ',']);fprintf(fileID,'\n');
            fprintf(fileID,json_tab);fprintf(fileID,'"Type": "SingleHand",');fprintf(fileID,'\n');
            fprintf(fileID,json_tab);fprintf(fileID,'"Name": "SingleHandVRMale",');fprintf(fileID,'\n');
            fprintf(fileID,json_tab);fprintf(fileID,'"Mesh": "/Game/Meshes/male_hand_high_r.male_hand_high_r",');fprintf(fileID,'\n');
            fprintf(fileID,json_tab);fprintf(fileID,['"HandTranslationTransform": ' hand_transl_transf_str ',']);fprintf(fileID,'\n');
            fprintf(fileID,json_tab);fprintf(fileID,'"BoneUpdate": "BoneAngles",');fprintf(fileID,'\n');    
            fprintf(fileID,json_tab);fprintf(fileID,'"Frames": [');fprintf(fileID,'\n');
            
            first_hand_pose = [];
            all_hand_angles = zeros(size(M, 1), 23);
            all_wrist_positions = zeros(size(M, 1), 3);
            all_hand_poses = cell(1, size(M, 1));
            idx = 0;
            for i=1:size(M,1)  

                idx = idx + 1;

                fpa_hand_pose = reshape(M(i, 2:end), [3, 21])';                
                hand_pose = FPAHandToCanonical(fpa_hand_pose);
                % from mm to cm
                hand_pose = hand_pose * 0.1; 

                %% Get Hand Angles                
                hand_angles = zeros(1, 23);
                % wrist
                hand_angles_new = GetHandAngles(hand_pose);

                hand_angles_new_degrees = hand_angles_new * 180 / pi;
                %hand_angles(1:3) = hand_angles_new_degrees(1, 1:3);      
                hand_angles(1:3) = hand_angles_new_degrees(10, 1:3);    
                % thumb
                hand_angles(4) = -hand_angles_new_degrees(3, 2);
                hand_angles(5) = hand_angles_new_degrees(3, 3); 
                hand_angles(6) = -hand_angles_new_degrees(4, 2);
                hand_angles(7) = -hand_angles_new_degrees(5, 2);
                % index
                hand_angles(8) = -hand_angles_new_degrees(7, 2);
                hand_angles(9) = -hand_angles_new_degrees(7, 3);
                hand_angles(10) = -hand_angles_new_degrees(8, 2);
                hand_angles(11) = -hand_angles_new_degrees(9, 2);
                % middle
                hand_angles(12) = -hand_angles_new_degrees(11, 2);
                hand_angles(13) = -hand_angles_new_degrees(11, 3);
                hand_angles(14) = -hand_angles_new_degrees(12, 2);
                hand_angles(15) = -hand_angles_new_degrees(13, 2);
                % ring
                hand_angles(16) = -hand_angles_new_degrees(15, 2);
                hand_angles(17) = -hand_angles_new_degrees(15, 3);
                hand_angles(18) = -hand_angles_new_degrees(16, 2);
                hand_angles(19) = -hand_angles_new_degrees(17, 2);
                % little
                hand_angles(20) = -hand_angles_new_degrees(19, 2);
                hand_angles(21) = -hand_angles_new_degrees(19, 3);
                hand_angles(22) = -hand_angles_new_degrees(20, 2);
                hand_angles(23) = -hand_angles_new_degrees(21, 2);                  
                
                %% Unreal VR hand required adjustments
                % in Unreal, rotation around Y and Z is inverted
                hand_angles(1) = -hand_angles(1);
                hand_angles(2) = -hand_angles(2);         
                hand_angles(3) = -hand_angles(3);
                
                hand_angles(4) = (hand_angles(4)/2) + 30;
                hand_angles(5) = hand_angles(5) + 110; 
                
                hand_angles(4) = min(90, hand_angles(4));
                hand_angles(4) = max(10, hand_angles(4));
                hand_angles(5) = max(-45, hand_angles(4));
                hand_angles(5) = min(90, hand_angles(5));
                hand_angles(6) = max(-10, hand_angles(6));
                hand_angles(7) = min(90,hand_angles(7));
                hand_angles(7) = max(-15,hand_angles(7));
                hand_angles(8) = max(-45,hand_angles(8));
                hand_angles(9) = min(90,hand_angles(9));
                hand_angles(9) = max(-45,hand_angles(9));
                hand_angles(10) = max(-15,hand_angles(10));
                hand_angles(11) = max(-15,hand_angles(11));
                hand_angles(12) = max(-45,hand_angles(12));
                hand_angles(13) = min(90,hand_angles(13));
                hand_angles(13) = max(-15,hand_angles(13));
                hand_angles(14) = max(-15,hand_angles(14));
                hand_angles(15) = max(-15,hand_angles(15));
                hand_angles(17) = min(90,hand_angles(17));
                hand_angles(17) = max(-45,hand_angles(17));
                hand_angles(18) = max(-15,hand_angles(18));
                hand_angles(19) = max(-15,hand_angles(19));
                hand_angles(20) = max(-45,hand_angles(20));
                hand_angles(21) = min(90,hand_angles(21));
                hand_angles(21) = max(-45,hand_angles(21));
                hand_angles(22) = max(-15,hand_angles(22));
                hand_angles(23) = max(-15,hand_angles(23));
                
                % check joint limits
                UnrealJointWithinLimits(hand_angles);
                
                all_hand_angles(idx, :) = hand_angles;                
                %% Print to JSON
                %all_hand_angles(idx, :) = hand_angles;
                fprintf(fileID,[json_tab json_tab]);fprintf(fileID,'{');fprintf(fileID,'\n');    
                fprintf(fileID,[json_tab json_tab json_tab]);fprintf(fileID,['"Frame": ' num2str(i) ',']);fprintf(fileID,'\n');
                fprintf(fileID,[json_tab json_tab json_tab]);fprintf(fileID,'"WristPosition": [');
                
                wrist_position = hand_pose(1, :);
                wrist_position(1) = wrist_position(1);
                wrist_position(2) = -wrist_position(2);
                %aux = wrist_position(1);
                %wrist_position(1) = wrist_position(2);
                %wrist_position(2) = aux;                
                
                for j=1:2
                    fprintf(fileID,[num2str(wrist_position(1, j)) ', ']);
                end
                fprintf(fileID,num2str(wrist_position(1, 3)));
                all_wrist_positions(i, :) = wrist_position(1, :); 
                fprintf(fileID,'],\n');
                fprintf(fileID,[json_tab json_tab json_tab]);fprintf(fileID,'"BoneAngles": [');
                for j=1:22
                    fprintf(fileID,[num2str(hand_angles(j)) ', ']);
                end
                fprintf(fileID,num2str(hand_angles(23)));
                fprintf(fileID,'],\n'); 
                % write positions
                fprintf(fileID,[json_tab json_tab json_tab]);fprintf(fileID,'"BonePositions": [],\n');
                
                hand_angles(1) = hand_angles(1) + 180;
                hand_transl = eul2rotm_(hand_angles(1:3) * pi /180) * UnrealVecToCanonical(unreal_hand_transf');
                hand_transl = CanonicaToUnrealVec(hand_transl);
                hand_transl(abs(hand_transl)<0.01) = 0;
                hand_transl_str = ['"' ...
                'X=' num2str(hand_transl(1)) ' '...
                'Y=' num2str(hand_transl(2)) ' '...
                'Z=' num2str(hand_transl(3)) '"'];
                fprintf(fileID,[json_tab json_tab json_tab]);fprintf(fileID,'"HandTranslation": ');
                fprintf(fileID,hand_transl_str);  
                fprintf(fileID,'\n');
                
                fprintf(fileID,[json_tab json_tab]);
                fprintf(fileID,'}');
                if i < size(M,1)
                    fprintf(fileID,',');
                end
                fprintf(fileID,'\n');    
                
                
                
                %disp(hand_pose);                
            end
            fprintf(fileID,json_tab);fprintf(fileID,']');fprintf(fileID,'\n');
            fprintf(fileID,'}');            
            fclose(fileID);
        end
    end
end


