function drawNees(est_files, labels, export_fig_path, num_runs, drawOnlyPose)
% This function requires Matlab signal processing toolbox for downsample()

% Example use:
% sim_dir = '';
% draw_nees({[sim_dir, 'simul_test_nofej/MSCKF_WavyCircle_NEES.txt'], ...
% [sim_dir, 'simul_wave/MSCKF_WavyCircle_NEES.txt'], ...
% [sim_dir, 'simul_ball/MSCKF_Ball_NEES.txt']}, ...
% {'Naive', 'Wave', 'Torus'}, '/tools/export_fig', 100);
if nargin < 5
    drawOnlyPose = 1;  % 0 draw position, orientation, and pose.
end
if nargin < 4
    num_runs = 100;
end
if nargin < 3
    export_fig_path = '/path/to/export_fig';
end

addpath(export_fig_path);

line_styles = {{'-.r', '-.g', '-.b'}, {'--r', '--g', '--b'}, ...
    {'-r', '-g', '-b'}, {'--m', '--c', '--k'}};
line_widths = {{2, 2, 2}, {2, 2, 2}, {1, 1, 1}, {2, 2, 2}};
[result_dir, ~, ~] = fileparts(est_files{1});

downsamplefactor = 10; % 1 original data.
backgroundColor = 'None'; % 'w' for debugging, 'None' for production.
single_line_styles={'r-', 'k-.', 'g--', 'b:'};
single_line_widths={1, 1, 1, 1.5};
indices = 2:7;
Q = 0.05;
[Tl, Tr] = twoSidedProbabilityRegionNees(Q, 6, num_runs);
[rl, rr] = twoSidedProbabilityRegionNees(Q, 3, num_runs);
[ql, qr] = twoSidedProbabilityRegionNees(Q, 3, num_runs);

close all;
figure;
hold on;
for i = 1:length(est_files)
    est_data = readmatrix(est_files{i}, 'NumHeaderLines', 1);
    est_data(:, 1) = est_data(:, 1) - est_data(1, 1);
    est_data = downsample(est_data, downsamplefactor);
    if drawOnlyPose == 1
        plot(est_data(:, 1), est_data(:, indices(3)), ...
            single_line_styles{i}, 'LineWidth', single_line_widths{i}, 'MarkerSize', 1.5);
    else
        drawColumnsInMatrix(est_data, indices(1:3), false, 1.0, line_styles{i}, line_widths{i});
    end

    % find average of last 10 secs.
    avgPeriod = 10;
    timeTol = 0.05;
    [avg, ~, ~] = averageAtOneEnd(est_data, avgPeriod, indices, 1, timeTol);
    disp(['average of position, ori., pose, in the last 10 secs: ', num2str(avg)]);
end

xlim([-10, est_data(end, 1) + 10]);
ylim([0, 20]);
xlabel('time (sec)');
ylabel('NEES (1)');

if drawOnlyPose == 1
    label_list = cell(1, length(est_files));
    for i = 1:length(est_files)
        
        label_list(1, i) = {labels{i}};
    end
else
    label_list = cell(1, length(est_files) * 3);
    for i = 1:length(est_files)
        label_list(1, (i-1) * 3 + (1:3)) = {[labels{i}, ' Position'], ...
            [labels{i}, ' Orientation'], [labels{i}, ' Pose']};
    end
end
leg = legend(label_list, 'Location', 'Best');
set(leg,'Interpreter', 'none');
set(gcf, 'Color', backgroundColor);
grid on;
outputfig = [result_dir, '/', 'nees.eps'];
if exist(outputfig, 'file')==2
    delete(outputfig);
end
export_fig(outputfig);
end
