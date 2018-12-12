function [hand_angles, hand_pose_fpa] = FPAGetHandAngles_old(M, i)
    %% read poses
    thumb_idxs = [2 7 8 9];
    index_idxs = [3 10 11 12];
    middle_idxs = [4 13 14 15];
    ring_idxs = [5 16 17 18];
    little_idxs = [6 19 20 21];
    hand_angles = zeros(1, 23);

    hand_pose_fpa = reshape(M(i, 2:end), [3, 21])';  
    hand_pose_fpa= hand_pose_fpa - hand_pose_fpa(1, :);
    
    thumb_pose = hand_pose_fpa(thumb_idxs, :);
    index_pose = hand_pose_fpa(index_idxs, :);
    middle_pose = hand_pose_fpa(middle_idxs, :);
    ring_pose = hand_pose_fpa(ring_idxs, :);
    little_pose = hand_pose_fpa(little_idxs, :);
    
    % calculate rotation to get hand normal (cross between rig and middle
    % inger first bones)to align with the Z axis
    ring_vec_unit = [ring_pose(1,:)/norm(ring_pose(1,:))]';
    middle_vec_unit = [middle_pose(1,:)/norm(middle_pose(1,:))]';
    cross_hand = cross(ring_vec_unit, middle_vec_unit);
    cross_hand = cross_hand / norm(cross_hand);
    %disp(ring_vec_unit');
    %disp(middle_vec_unit');
    %disp(cross_hand');
    %disp('---------------------------------------');
    rot_hand_1 = vrrotvec2mat(vrrotvec(cross_hand, [0; 0; 1]));
    min_angle = 1e-3;
    %rot_hand_1(rot_hand_1 > -min_angle & rot_hand_1 < min_angle) = 0;
    % calculate second hand rotation to align middle finger with the X axis
    rot_middle = rot_hand_1 * [middle_pose(1,:)/norm(middle_pose(1,:))]';
    rot_hand_2 = vrrotvec2mat(vrrotvec(rot_middle, [1; 0; 0]));
    %rot_hand_2(rot_hand_2 > -min_angle & rot_hand_2 < min_angle) = 0;
    rot_hand_3 = eye(3);
    % check if hand has palm facing down (get rotation to make it face down)
    hand_pose_rot_1_2 = [rot_hand_2 * rot_hand_1 * hand_pose_fpa']';
    if hand_pose_rot_1_2(2, 3) < 0
        rot_hand_3 = eul2rotm_([pi, 0, 0]);
    end
    rot_hand = rot_hand_3 * rot_hand_2 * rot_hand_1;
    %rot_hand_3(rot_hand_3 > -min_angle & rot_hand_3 < min_angle) = 0;
    %get total hand rotation for finalalignment
    rot_hand_inv = inv(rot_hand);
    wrist_rotation = rotm2eul_(rot_hand_inv) * 180.0 / pi;  
  
    % for debugging
    hand_pose_align = [rot_hand * hand_pose_fpa']';
    hand_pose_inv = [rot_hand_inv * hand_pose_align']';
    rot_error = sum(hand_pose_inv(:) - hand_pose_fpa(:)) / size(hand_pose_inv, 1);
    if rot_error > 1e-10
        hand_angles = -1;
        return;
    end    
    
    hand_angles(1:3) = wrist_rotation;
    hand_angles(4:7) = GetFingerAngles(thumb_pose);
    
    % adjustments for Unreal
    hand_angles(4) = hand_angles(4) / 4;
    hand_angles(4) = abs(hand_angles(4));
    hand_angles(5) = abs(hand_angles(5)) - 120;
    hand_angles(8:11) = GetFingerAngles(index_pose);
    hand_angles(9) = hand_angles(9) - 30;
    hand_angles(12:15) = GetFingerAngles(middle_pose);
    hand_angles(13) = hand_angles(9) - 10;
    hand_angles(16:19) = GetFingerAngles(ring_pose);
    hand_angles(17) = hand_angles(17) - 30;
    hand_angles(20:23) = GetFingerAngles(little_pose);
    hand_angles(21) = hand_angles(21)  - 15;
    
    %plot_fpa_hand(hand_pose_fpa);   
    
 end
