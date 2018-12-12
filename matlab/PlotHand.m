function [h] = PlotHand(hand_pose)
    thumb_pose = hand_pose(2:5, :);
    index_pose = hand_pose(6:9, :);
    middle_pose = hand_pose(10:13, :);
    ring_pose = hand_pose(14:17, :);
    little_pose = hand_pose(18:21, :);
    
    figure;
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
    hold off;    
end

