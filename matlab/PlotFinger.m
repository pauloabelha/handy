function [] = PlotFinger(finger_pose, hand_root)
    if ~exist('hand_root','var')
        hand_root = [0, 0, 0];
    end
    colors = {'k', 'r', 'g', 'b'};
    figure;
    axis equal;
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    hold on;
    finger_pose = [hand_root; finger_pose];
    for j=1:4
        plot3(finger_pose(j:j+1, 1), finger_pose(j:j+1,2), finger_pose(j:j+1, 3), 'color', colors{j});
    end
    
    hold off;
end

