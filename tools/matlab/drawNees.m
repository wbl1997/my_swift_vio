function drawNees(est_files, labels, num_runs, export_fig_path)
% sim_dir = '';
% draw_nees({[sim_dir, 'msckf_simul_test_nofej/MSCKF_WavyCircle_NEES.txt'], ...
% [sim_dir, 'msckf_simul_wave/MSCKF_WavyCircle_NEES.txt'], ...
% [sim_dir, 'msckf_simul_ball/MSCKF_Ball_NEES.txt']}, ...
% 1000, {'Naive', 'Wave', 'Torus'}, '/tools/export_fig');

addpath(export_fig_path);

line_styles = {{':r', ':g', ':b'}, {'-r', '-g', '-b'}, {'--r', '--g', '--b'}, {'--m', '--c', '--k'}};
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
    
    drawColumnsInMatrix(est_data, indices, false, 1.0, line_styles{i});
   
    % find average of last 10 secs.
    avgPeriod = 10;
    timeTol = 0.05;
    [avg, ~, ~] = averageAtOneEnd(est_data, avgPeriod, indices, 1, timeTol);
    disp(['average of position, ori., pose, in the last 10 secs: ', num2str(avg)]);
end

% xlim([-10, est_data(end, 1) + 10]);
% ylim([0, 20]);
xlabel('time (sec)');
ylabel('NEES (1)');

label_list = cell(1, length(est_files) * 3);
for i = 1:length(est_files)
    label_list(1, (i-1) * 3 + (1:3)) = {[labels{i}, '-Position'], ...
        [labels{i}, '-Orientation'], [labels{i}, '-Pose']};
end
leg = legend(label_list);
set(leg,'Interpreter', 'none');
% set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'nees.eps'];
if exist(outputfig, 'file')==2
    delete(outputfig);
end
export_fig(outputfig);
end
