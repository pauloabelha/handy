function [hand_pose, hand_vectors] = PositionHandNaturalPose(hand)
    %% position hand in natural state (see FPASubjectHand header comments)
    % also collects 3 vectors per finger (for each mcp, pip, dip bone)
    X_vec = [1; 0; 0];
    hand_pose = zeros(21, 3);  
    nat_rots = hand(1:15);
    bone_lengths = hand(16:end);
    hand_vectors = zeros(15, 3);
    idx_hand_vec = 1;
    for i=1:5
        % position joint mcp
        idx_nat = (i*3) - 2;
        idx_mcp = (i*4) - 2;
        nat_rot = eul2rotm_(nat_rots(idx_nat:idx_nat+2));
        mcp = nat_rot * X_vec * bone_lengths(idx_mcp-1);        
        hand_pose(idx_mcp,1:3) = mcp;
        hand_vectors(idx_hand_vec, :) = mcp;
        idx_hand_vec = idx_hand_vec + 1;
        % position joints pip, dip, tip
        prev_translation = mcp;
        for j=1:3
            joint = nat_rot * X_vec * bone_lengths(idx_mcp-1+j);    
            hand_vectors(idx_hand_vec, :) = joint;
            idx_hand_vec = idx_hand_vec + 1;
            hand_pose(idx_mcp+j,1:3) = joint + prev_translation;            
            prev_translation = prev_translation + joint;
        end
    end  
    hand_vectors = hand_vectors';
end

