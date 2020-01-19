function draw_nees(est_file, cmp_file)
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
leg = legend([name_est, '-p'], [name_est, '-q'], [name_est, '-T'], ...
    [name_cmp, '-p'], [name_cmp, '-q'], [name_cmp, '-T']);
set(leg,'Interpreter', 'none');
end