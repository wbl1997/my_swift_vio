function draw_nees(est_file, num_runs, cmp_file, label_est, label_cmp)
export_fig_path = '/media/jhuai/Seagate/jhuai/tools/export_fig/';
addpath(export_fig_path);
if nargin < 5
    label_cmp = 'cmp';
end
if nargin < 4
    label_est = 'est';
end
 
if nargin < 3
    cmp_data = [];
else
    cmp_data = readmatrix(cmp_file, 'NumHeaderLines', 1);
end

est_line_style = {'r', 'g', 'b'};
cmp_line_style = {'--r', '--g', '--b'};
[result_dir, ~, ~] = fileparts(est_file);

indices = 2:4;

est_data = readmatrix(est_file, 'NumHeaderLines', 1);
startTime = est_data(1, 1);

est_data(:, 1) = est_data(:, 1) - startTime;

Q = 0.05;
[Tl, Tr] = two_sided_prob_region_nees(Q, 6, num_runs);
[rl, rr] = two_sided_prob_region_nees(Q, 3, num_runs);
[ql, qr] = two_sided_prob_region_nees(Q, 3, num_runs);

close all;
figure;
draw_data_columns(est_data, indices, 1.0, false, est_line_style);
if ~isempty(cmp_data)
    cmp_data(:, 1) = cmp_data(:, 1) - startTime;
    draw_data_columns(cmp_data, indices, 1.0, false, cmp_line_style);
end
xlim([-10, est_data(end, 1) + 10]);
ylim([0, 20]);
xlabel('time (sec)');
ylabel('NEES (1)');

% plot(est_data([1, end], 1), [3, 3], 'r--');
% plot(est_data([1, end], 1), [rl, rl], 'r-.');
% plot(est_data([1, end], 1), [rr, rr], 'r-.');
% plot(est_data([1, end], 1), [3, 3], 'g--');
% plot(est_data([1, end], 1), [rl, rl], 'g-.');
% plot(est_data([1, end], 1), [rr, rr], 'g-.');
% plot(est_data([1, end], 1), [6, 6], 'b--');
% plot(est_data([1, end], 1), [Tl, Tl], 'b-.');
% plot(est_data([1, end], 1), [Tr, Tr], 'b-.');

% find average of last 10 secs.
avg_period = 10;
time_tol = 0.05;
startIndex = find(abs(est_data(end, 1) - avg_period - est_data(:,1)) <...
    time_tol, 1);
if isempty(startIndex)
    startIndex = size(est_data, 1) - 100;
end
startIndex
size(est_data, 1)
est_avg = mean(est_data(startIndex:end, indices), 1)
cmp_avg = mean(cmp_data(startIndex:end, indices), 1)

leg = legend([label_est, '-Position'], [label_est, '-Orientation'], [label_est, '-Pose'], ...
    [label_cmp, '-Position'], [label_cmp, '-Orientation'], [label_cmp, '-Pose']);
set(leg,'Interpreter', 'none');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'nees_fej_vs_naive.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);
end
