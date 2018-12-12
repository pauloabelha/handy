%% A hand is a 35-element vector
% A hand natural position (all angles set to 0) is:
%   palm facing up, wrist at origin and middle finger along the X axis
%   this natural position incldues the finger roots' natural rotations
%   (bones from your wrist to each first knucle have natural angles between)
% bone_lengths is a 20-element vector containing bone lengths:
%   thumb_mcp, thumb_pip, thumb_dip, thumb_tip, index_mcp ...
% Angles are in radians and lengths in mm
% The second return are the hand angles:
% hand_angles is a 21 x 3 matrix with XYZ Euler angles for:
%   wrist XYZ
%   thumb root XYZ
%   thumb mcp XYZ
%   thumb pip XYZ
%   thumb dip XYZ
%   index root XYZ
%   ...
function [bone_lengths, hand_angles] = FPASubjectHand(subject)
    bone_lengths = -1;
    %% Subject 1
    if strcmp(subject, 'Subject_1')
        nat_rots = FPASubject1NaturalRotations;
        bone_lengths = [...
            15.769    57.084    37.278    28.000...
            80.000    42.666    28.990    20.000...
            77.000    49.000    30.000    24.000...
            73.000    44.000    28.000    25.000...
            76.000    32.928    20.525    22.000];
    end    
    hand_angles = HandNaturalRotationsToHandAngles(nat_rots);
end

