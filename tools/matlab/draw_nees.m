function draw_nees(est_files, num_runs, labels)
% sim_dir = '/media/jhuai/Seagate/jhuai/temp/';
% draw_nees({[sim_dir, 'msckf_simul_test_nofej/MSCKF_WavyCircle_NEES.txt'], ...
% [sim_dir, 'msckf_simul_wave/MSCKF_WavyCircle_NEES.txt'], ...
% [sim_dir, 'msckf_simul_ball/MSCKF_Ball_NEES.txt']}, ...
% 1000, {'Naive', 'Wave', 'Torus'});

export_fig_path = '/media/jhuai/Seagate/jhuai/tools/export_fig/';
addpath(export_fig_path);

line_styles = {{':r', ':g', ':b'}, {'-r', '-g', '-b'}, {'--r', '--g', '--b'}};
[result_dir, ~, ~] = fileparts(est_files{1});

indices = 2:4;
Q = 0.05;
[Tl, Tr] = two_sided_prob_region_nees(Q, 6, num_runs);
[rl, rr] = two_sided_prob_region_nees(Q, 3, num_runs);
[ql, qr] = two_sided_prob_region_nees(Q, 3, num_runs);

close all;
figure;
    
startTime = 0;

for i = 1:length(est_files)
    est_data = readmatrix(est_files{i}, 'NumHeaderLines', 1);
    if startTime == 0
        startTime = est_data(1, 1);
    end
    est_data(:, 1) = est_data(:, 1) - startTime;
    
    draw_data_columns(est_data, indices, 1.0, false, line_styles{i});
   
    % find average of last 10 secs.
    avg_period = 10;
    time_tol = 0.05;
    startIndex = find(abs(est_data(end, 1) - avg_period - est_data(:,1)) <...
        time_tol, 1);
    if isempty(startIndex)
        startIndex = size(est_data, 1) - 100;
    end
    est_avg = mean(est_data(startIndex:end, indices), 1)
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
label_list = cell(1, length(est_files) * 3);
for i = 1:length(est_files)
    label_list(1, (i-1) * 3 + (1:3)) = {[labels{i}, '-Position'], ...
        [labels{i}, '-Orientation'], [labels{i}, '-Pose']};
end
leg = legend(label_list);
set(leg,'Interpreter', 'none');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'nees.eps'];
if exist(outputfig, 'file')==2
    delete(outputfig);
end
export_fig(outputfig);
end
