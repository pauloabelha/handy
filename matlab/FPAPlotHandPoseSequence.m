function [] = FPAPlotHandPoseSequence(hand_poses)
    plot_fpa_hand(hand_poses{1});
    for i=2:numel(hand_poses)
        pause(0.1);
        cla reset;
        plot_fpa_hand(hand_poses{i}); 
        view(0.7, 0.7);
        title(num2str(i));
        drawnow;
    end
    close all;
end

