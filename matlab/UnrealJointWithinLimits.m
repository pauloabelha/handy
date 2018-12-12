function [within_limits] = UnrealJointWithinLimits(hand_angles)
    %% limits that are euqal per joints type
    %% limits that are equal per joints type
    mcp_bend_min = -45;
    mcp_spread_min = -45;
    pip_bend_min = -15;
    dip_bend_min = -15;
    mcp_bend_max = 90;
    mcp_spread_max = 90;
    pip_bend_max = 90;
    dip_bend_max = 90;
    %% min
    hand_angles_min = zeros(1, 23);
    hand_angles_min(:) = -361;
    % thumb
    hand_angles_min(4) = -15;
    hand_angles_min(5) = -45;
    hand_angles_min(6) = pip_bend_min;
    hand_angles_min(7) = dip_bend_min;
    % index
    hand_angles_min(8) = mcp_bend_min;
    hand_angles_min(9) = mcp_spread_min;
    hand_angles_min(10) = pip_bend_min;
    hand_angles_min(11) = dip_bend_min;
    % middle
    hand_angles_min(12) = mcp_bend_min;
    hand_angles_min(13) = mcp_spread_min;
    hand_angles_min(14) = pip_bend_min;
    hand_angles_min(15) = dip_bend_min;
    % ring
    hand_angles_min(16) = mcp_bend_min;
    hand_angles_min(17) = mcp_spread_min;
    hand_angles_min(18) = pip_bend_min;
    hand_angles_min(19) = dip_bend_min;
    % little
    hand_angles_min(20) = mcp_bend_min;
    hand_angles_min(21) = mcp_spread_min;
    hand_angles_min(22) = pip_bend_min;
    hand_angles_min(23) = dip_bend_min;
    %% max
    hand_angles_max = zeros(1, 23);
    hand_angles_max(:) = 361;
    % thumb
    hand_angles_max(4) = mcp_bend_max;
    hand_angles_max(5) = 91;
    hand_angles_max(6) = pip_bend_max;
    hand_angles_max(7) = dip_bend_max;
    % index
    hand_angles_max(8) = mcp_bend_max;
    hand_angles_max(9) = mcp_spread_max;
    hand_angles_max(10) = pip_bend_max;
    hand_angles_max(11) = dip_bend_max;
    % middle
    hand_angles_max(12) = mcp_bend_max;
    hand_angles_max(13) = mcp_spread_max;
    hand_angles_max(14) = pip_bend_max;
    hand_angles_max(15) = dip_bend_max;
    % ring
    hand_angles_max(16) = mcp_bend_max;
    hand_angles_max(17) = mcp_spread_max;
    hand_angles_max(18) = pip_bend_max;
    hand_angles_max(19) = dip_bend_max;
    % little
    hand_angles_max(20) = mcp_bend_max;
    hand_angles_max(21) = mcp_spread_max;
    hand_angles_max(22) = pip_bend_max;
    hand_angles_max(23) = dip_bend_max;
    
    %% check limits
    limits_check = ...
        hand_angles >= hand_angles_min &...
        hand_angles <= hand_angles_max;
    limit_invalid_idx = find(~limits_check);
    if isempty(limit_invalid_idx)
        return;
    else
        error(sprintf(['Joints [' num2str(limit_invalid_idx) '] are outside their allowed limit.\nValues are ['...
            num2str(hand_angles(limit_invalid_idx)) ']\nMinima: [' ...
            num2str(hand_angles_min(limit_invalid_idx)) ']\nMaxima: [' ...
            num2str(hand_angles_max(limit_invalid_idx)) ']']));
    end        
    
end

