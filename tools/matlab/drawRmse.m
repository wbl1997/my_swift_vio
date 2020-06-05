function drawRmse(dataFiles, fileLabels)
% sim_dir = '';
% draw_rmse({[sim_dir, 'msckf_simul_wave/MSCKF_WavyCircle_RMSE.txt'], ...
% [sim_dir, 'msckf_simul_ball/MSCKF_Ball_RMSE.txt']}, ...
% {'Wave', 'Torus'});

fileColumnStyles = {{'-r', '-g', '-b', '-k'}, {'--r', '--g', '--b', '--k'},...
    {'-.r', '-.g', '-.b', '-.k'}, {':r', ':g', ':b', ':k'}, ...
    {'-m', '-y', '-k', '-c'}, {'--m', '--y', '--k', '--c'}};
close all;

[result_dir, ~, ~] = fileparts(dataFiles{1});
format short;

matrices = cell(length(dataFiles), 1);
for i = 1:length(dataFiles)
    file = dataFiles{i};
    matrices{i} = readmatrix(file, 'NumHeaderLines', 1);
end

figure;
indices = 2:4;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'x', 'y', 'z'}, indices, fileColumnStyles);
for j = 1:length(matrices)
    est_avg = sqrt(sum(matrices{j}(end-100:end, indices).^2, 2));
    disp(['RMSE in position of ', fileLabels{j}, ': ', num2str(est_avg(end))]);
end

ylabel('$\delta \mathbf{t}_{WB}$ (m)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_t_WB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 5:7;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'x', 'y', 'z'}, indices, fileColumnStyles, 180/pi);
for j = 1:length(matrices)
    est_avg = sqrt(sum(matrices{j}(end-100:end, indices).^2, 2)) * 180 / pi;
    disp(['RMSE in orientation of ', fileLabels{j}, ' in degrees: ', num2str(est_avg(end))]);
end
ylabel('$\delta \mathbf{\theta}_{WB} (^{\circ})$', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_theta_WB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 8:10;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'x', 'y', 'z'}, indices, fileColumnStyles);
ylabel('$\delta \mathbf{v}_{WB}$ (m/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_v_WB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 11:13;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'x', 'y', 'z'}, indices, fileColumnStyles);
ylabel('$\delta \mathbf{b}_{g}$ (rad/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_b_g.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 14:16;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'x', 'y', 'z'}, indices, fileColumnStyles);
ylabel('$\delta \mathbf{b}_{a}$ (m/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_b_a.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 17 + [0, 4, 8];
drawColumnsInMultipleMatrices(matrices, fileLabels, {'x', 'y', 'z'}, indices, fileColumnStyles);
ylabel('$\delta \mathbf{T}_{g}$ (m/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_T_g.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 26 + [0, 4, 8];
drawColumnsInMultipleMatrices(matrices, fileLabels, {'x', 'y', 'z'}, indices, fileColumnStyles);
ylabel('$\delta \mathbf{T}_{s}$ (m/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_T_s.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 35 + [0, 4, 8];
drawColumnsInMultipleMatrices(matrices, fileLabels, {'x', 'y', 'z'}, indices, fileColumnStyles);
ylabel('$\delta \mathbf{T}_{a}$ (m/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_T_a.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 44:46;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'x', 'y', 'z'}, indices, fileColumnStyles);
ylabel('$\delta \mathbf{p}_{CB}$ (m/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_p_CB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 47:50;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'fx', 'fy', 'cx', 'cy'}, indices, fileColumnStyles);
ylabel('$\delta \mathbf{f-c}$ (m/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_fc.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 51:54;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'k1', 'k2', 'p1', 'p2'}, indices, fileColumnStyles);
ylabel('$\delta \mathbf{k-p}$ (m/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_k-p.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 55:56;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'td', 'tr'}, indices, fileColumnStyles);
ylabel('$\delta \mathbf{td-tr}$ (m/s)', 'Interpreter', 'latex');
set(gcf, 'Color', 'None');
grid on;
outputfig = [result_dir, '/', 'rmse_td-tr.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);
end

