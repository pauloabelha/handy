v = [10.1 10.5 11 12 15 20 30 40 50 60 70 80 90 100 200;...
    0.004971 0.043047 0.086757 0.163170 0.331097 0.498744...
    0.6661 0.7497 0.7998 0.8332 0.8570 0.8749 0.8888 0.899943 0.949984];
v = fliplr(v');
base_start = 1;
base_step = 0.001;
base_end=2;
exp_divider_start = 8;
exp_divider_step = 0.001;
exp_divider_end = 10;
const_start = 0.1;
const_step = 0.1;
const_end = 10;
min_dist = 1e10;
min_base= -1;
min_exp_divider = -1;
min_res = -1;
dists = [];
for base=base_start:base_step:base_end
    for exp_divider=exp_divider_start:exp_divider_step:exp_divider_end
        
            res = 1 - 1 ./ base.^((v(:,2)-10)./exp_divider);
            dist = sum(abs(v(:,1)-res))/size(v,1);
            dists(end+1) = dist;
            if dist <= min_dist
                min_dist = dist;
                min_base = base;
                min_exp_divider = exp_divider;
                min_res = res;
            end

    end
    a = 0;
end


