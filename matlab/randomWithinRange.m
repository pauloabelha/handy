function [ret] = randomWithinRange(num_samples, range_min, range_max)
    ret = (range_max-range_min).*rand(num_samples,1) + range_min;
end

