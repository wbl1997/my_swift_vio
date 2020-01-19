function draw_rmse(est_file, cmp_file)
if nargin < 2
    cmp_file = est_file;
end
est_line_style = {'r', 'g', 'b'};
cmp_line_style = {'--r', '--g', '--b'};
[result_dir, name_est, ext] = fileparts(est_file);
[result_dir, name_cmp, ext] = fileparts(cmp_file);
indices = 2:4;
est_data = readmatrix(est_file, 'NumHeaderLines', 1);
cmp_data = readmatrix(cmp_file, 'NumHeaderLines', 1);
figure;
draw_data_columns(est_data, indices, 1.0, false, est_line_style);
draw_data_columns(cmp_data, indices, 1.0, false, cmp_line_style);
leg = legend([name_est, '-px'], [name_est, '-py'], [name_est, '-pz'], ...
    [name_cmp, '-px'], [name_cmp, '-py'], [name_cmp, '-pz']);
set(leg,'Interpreter', 'none');

indices = 5:7;
figure;
draw_data_columns(est_data, indices, 1.0, false, est_line_style);
draw_data_columns(cmp_data, indices, 1.0, false, cmp_line_style);
leg = legend([name_est, '-qx'], [name_est, '-qy'], [name_est, '-qz'], ...
    [name_cmp, '-qx'], [name_cmp, '-qy'], [name_cmp, '-qz']);
set(leg,'Interpreter', 'none');

indices = 8:10;
figure;
draw_data_columns(est_data, indices, 1.0, false, est_line_style);
draw_data_columns(cmp_data, indices, 1.0, false, cmp_line_style);
leg = legend([name_est, '-vx'], [name_est, '-vy'], [name_est, '-vz'], ...
    [name_cmp, '-vx'], [name_cmp, '-vy'], [name_cmp, '-vz']);
set(leg,'Interpreter', 'none');
end