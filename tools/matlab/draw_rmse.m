function draw_rmse(est_files)
line_styles = {{'-r', '-g', '-b'}, {'--r', '--g', '--b'},...
    {'-.r', '-.g', '-.b'}, {':r', ':g', ':b'}, ...
    {'-m', '-y', '-k'}, {'--m', '--y', '--k'}};

indices = 2:4;
draw_rmse_columns(est_files, line_styles, indices, {'-px', '-py', '-pz'});
indices = 5:7;
draw_rmse_columns(est_files, line_styles, indices, {'-qx', '-qy', '-qz'});
indices = 8:10;
draw_rmse_columns(est_files, line_styles, indices, {'-vx', '-vy', '-vz'});
end

function draw_rmse_columns(est_files, line_styles, indices, labels)
figure;
legend_list = {};
for i = 1:length(est_files)
    est_file = est_files{i};
    [result_dir, name_est, ext] = fileparts(est_file);
    est_data = readmatrix(est_file, 'NumHeaderLines', 1);
    est_line_style = line_styles{i};
    draw_data_columns(est_data, indices, 1.0, false, est_line_style);
    legend_list = {legend_list{:}, [name_est, labels{1}], ...
        [name_est, labels{2}], [name_est, labels{3}]};
end
legend(legend_list);
end
