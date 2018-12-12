function [h] = FPAPlotHand(hand_pose)
    thumb_idxs = [2 7 8 9];
    index_idxs = [3 10 11 12];
    middle_idxs = [4 13 14 15];
    ring_idxs = [5 16 17 18];
    little_idxs = [6 19 20 21];

    thumb_pose = hand_pose(thumb_idxs, :);
    index_pose = hand_pose(index_idxs, :);
    middle_pose = hand_pose(middle_idxs, :);
    ring_pose = hand_pose(ring_idxs, :);
    little_pose = hand_pose(little_idxs, :);
    
    hand_pose = [hand_pose(1, :); thumb_pose; index_pose; middle_pose; ring_pose; little_pose];
    hand_pose_canonical = hand_pose;
    hold on;   
    for j=1:5
        start_idx = (j * 4) - 2;
        bone = [hand_pose(1, :); hand_pose(start_idx, :)];
        plot3(bone(1:2, 1), bone(1:2, 2), bone(1:2, 3), 'color', 'k');
    end
    for j=1:3
        plot3(thumb_pose(j:j+1, 1), thumb_pose(j:j+1,2), thumb_pose(j:j+1, 3), 'color', 'r');
    end
    for j=1:3
        plot3(index_pose(j:j+1, 1), index_pose(j:j+1,2), index_pose(j:j+1, 3), 'color', 'g');
    end
    for j=1:3
        plot3(middle_pose(j:j+1, 1), middle_pose(j:j+1,2), middle_pose(j:j+1, 3), 'color', 'b');
    end
    for j=1:3
        plot3(ring_pose(j:j+1, 1), ring_pose(j:j+1,2), ring_pose(j:j+1, 3), 'color', 'y');
    end
    for j=1:2
        plot3(little_pose(j:j+1, 1), little_pose(j:j+1,2), little_pose(j:j+1, 3), 'color', 'm');
    end
    j = 3;
    h = plot3(little_pose(j:j+1, 1), little_pose(j:j+1,2), little_pose(j:j+1, 3), 'color', 'm');
    axis equal;
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    view(0, 90);
    hold off;    
end

