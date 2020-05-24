function draw_rmse(est_files, est_labels)
line_styles = {{'-r', '-g', '-b', '-k'}, {'--r', '--g', '--b', '--k'},...
    {'-.r', '-.g', '-.b', '-.k'}, {':r', ':g', ':b', ':k'}, ...
    {'-m', '-y', '-k', '-c'}, {'--m', '--y', '--k', '--c'}};
close all;

[result_dir, ~, ~] = fileparts(est_files{1});
format short;
indices = 2:4;

draw_rmse_columns(est_files, line_styles, indices, est_labels, {'x', 'y', 'z'});
ylabel('$\delta \mathbf{t}_{WB}$ (m)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_t_WB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

indices = 5:7;
draw_rmse_columns(est_files, line_styles, indices, est_labels, {'x', 'y', 'z'}, 180/pi);
ylabel('$\delta \mathbf{\theta}_{WB} (^{\circ})$', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_theta_WB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

indices = 8:10;
draw_rmse_columns(est_files, line_styles, indices, est_labels, {'x', 'y', 'z'});
ylabel('$\delta \mathbf{v}_{WB}$ (m/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_v_WB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

indices = 11:13;
draw_rmse_columns(est_files, line_styles, indices, est_labels, {'x', 'y', 'z'});
ylabel('$\delta \mathbf{b}_{g}$ (rad/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_b_g.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

indices = 14:16;
draw_rmse_columns(est_files, line_styles, indices, est_labels, {'x', 'y', 'z'});
ylabel('$\delta \mathbf{b}_{a}$ (m/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_b_a.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

indices = 17 + [0, 4, 8];
draw_rmse_columns(est_files, line_styles, indices, est_labels, {'x', 'y', 'z'});
ylabel('$\delta \mathbf{T}_{g}$ (m/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_T_g.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

indices = 26 + [0, 4, 8];
draw_rmse_columns(est_files, line_styles, indices, est_labels, {'x', 'y', 'z'});
ylabel('$\delta \mathbf{T}_{s}$ (m/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_T_s.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

indices = 35 + [0, 4, 8];
draw_rmse_columns(est_files, line_styles, indices, est_labels, {'x', 'y', 'z'});
ylabel('$\delta \mathbf{T}_{a}$ (m/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_T_a.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

indices = 44:46;
draw_rmse_columns(est_files, line_styles, indices, est_labels, {'x', 'y', 'z'});
ylabel('$\delta \mathbf{p}_{CB}$ (m/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_p_CB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

indices = 47:50;
draw_rmse_columns(est_files, line_styles, indices, est_labels, {'fx', 'fy', 'cx', 'cy'});
ylabel('$\delta \mathbf{f-c}$ (m/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_fc.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

indices = 51:54;
draw_rmse_columns(est_files, line_styles, indices, est_labels, {'k1', 'k2', 'p1', 'p2'});
ylabel('$\delta \mathbf{k-p}$ (m/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_k-p.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

indices = 55:56;
draw_rmse_columns(est_files, line_styles, indices, est_labels, {'td', 'tr'});
ylabel('$\delta \mathbf{td-tr}$ (m/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_td-tr.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);
end

function draw_rmse_columns(est_files, line_styles, indices, ...
    est_labels, column_labels, scalar)
if nargin < 6
    scalar = 1.0;
end
figure;
legend_list = cell(1, length(est_files) * length(column_labels));
startTime = 0;
dataBegin = 0;
dataEnd = 0;
for i = 1:length(est_files)
    est_file = est_files{i};
    est_label = est_labels{i};
    est_data = readmatrix(est_file, 'NumHeaderLines', 1);
    if startTime < 1e-7
        startTime = est_data(1, 1);
        dataBegin = est_data(1, 1) - startTime;
        dataEnd = est_data(end, 1) - startTime;
    end
    est_data(:, 1) = est_data(:, 1) - startTime;
    
    avg_period = 10;
    time_tol = 0.05;
    startIndex = find(abs(est_data(end, 1) - avg_period - est_data(:,1)) <...
        time_tol, 1);
    if isempty(startIndex)
        startIndex = size(est_data, 1) - 100;
    end

    est_avg = sqrt(sum(est_data(startIndex:end, indices).^2, 2));
    est_label
    [indices, startIndex, size(est_data, 1)]
    est_avg(end)

    est_line_style = line_styles{i};
    draw_data_columns(est_data, indices, scalar, false, est_line_style);
    for j = 1: length(column_labels)
        legend_list((i - 1) * length(column_labels) + j) = ...
            {[est_label, '-', column_labels{j}]};
    end
end
legend(legend_list);
xlabel('time (sec)');
xlim([dataBegin-10, dataEnd+10]);
end
