%% Gotten from measuring bone vectors in Unreal for the Male VR Hand
% values for tip lengths are invented (hand does not have them as bones,
% so it is not posible to infer location
function [bone_lengths] = GetUnrealMaleVRHandBoneLengths()
    little_mcp = 9.635697;
	little_pip = 3.730112;
	little_dip = 2.463038;
    little_tip = 2;
    little = [little_mcp little_pip little_dip little_tip];
    thumb_mcp = 3.818611;
    thumb_dip = 5.136038;
    thump_pip = 3.176646;
    thumb_tip = 3;
    thumb = [thumb_mcp thumb_dip thump_pip thumb_tip];
    middle_mcp = 10.297134;
    middle_pip = 5.110246;
    middle_dip = 3.071;
    middle_tip = 2;
    middle = [middle_mcp middle_pip middle_dip middle_tip];
    ring_mcp = 10.140307;
    ring_pip = 4.358661;
    ring_dip = 2.969887;
    ring_tip = 2;
    ring = [ring_mcp ring_pip ring_dip ring_tip];
    index_mcp = 11.148599;
    index_pip = 4.676986;
    index_tip = 2.376295;
    index_tip = 2;
    index = [index_mcp index_pip index_tip index_tip];
    bone_lengths = [thumb index middle ring little];   
    % cm to mm
    bone_lengths = bone_lengths * 10;
end

