function drawRmse(dataFiles, fileLabels, exportFigPath)
% sim_dir = '';
% draw_rmse({[sim_dir, 'simul_wave/MSCKF_WavyCircle_RMSE.txt'], ...
% [sim_dir, 'simul_ball/MSCKF_Ball_RMSE.txt']}, ...
% {'Wave', 'Torus'});

if nargin < 3
    exportFigPath = '/tools/export_fig/';
end
addpath(exportFigPath);
backgroundColor = 'None'; % 'w' for debugging, 'None' for production.
subPlot = 1;

probeEpochs = [0, 3, 10, 30, 100, 300];
fileColumnStyles =  {
        {'--r', '-g', '-b', '-k', '.k', '.b', ...
        '-c', '-m', '-y'}, ...
        {'-.k', '--g', '--b', '--k', '-.k', '-.b', ...
        '--c', '--m', '--y'}, ...
        {'-b', '-.g', '-.b', ':k', '-.r', '-.g', ...
        ':c', ':m', ':y'}, ...
        {':g', '--c', '--k', '.k', '-.r', '-.g', ...
        ':c', ':m', ':y'}};
matrixColumnLineWidths = {{1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1}, ...
        {1,1,1,1,1,1,1,1,1}, {2,1,1,1,1,1,1,1,1}};
close all;
[result_dir, ~, ~] = fileparts(dataFiles{1});
format short;

matrices = cell(length(dataFiles), 1);
for i = 1:length(dataFiles)
    file = dataFiles{i};
    matrices{i} = readmatrix(file, 'NumHeaderLines', 1);
end

% find the indices for probe epochs
startTime = matrices{1}(1, 1);
timeTol = 0.055;
epochIndexList = zeros(1, length(probeEpochs));
for j = 1:length(probeEpochs)
    epoch = startTime + probeEpochs(j);
    index = find(abs(epoch - matrices{1}(:,1)) <...
        timeTol, 1);
    if isempty(index)
        index = find(abs(epoch - matrices{1}(:,1)) <...
            timeTol * 2, 1);
        assert(~isempty(index));
    end
    epochIndexList(j) = index;
end
fprintf(['Checking rmse at indices ', num2str(epochIndexList), ...
    ' of time ', num2str(probeEpochs), ...
    ' for matrix of rows ', num2str(size(matrices{1}, 1)), '\n']);
componentIndex = 0;
componentList = cell(1, 20);
formatList = cell(1, 20);
rmseList = cell(length(matrices), 1);
for i = 1 : length(matrices)
    rmseList{i} = zeros(length(epochIndexList), 20);
end

figure;
indices = 2:4;
labels = {'x (m)', 'y (m)', 'z (m)'};
drawColumnsInMultipleMatrices(matrices, fileLabels, labels, ...
    indices, 0, fileColumnStyles, matrixColumnLineWidths, 1.0, subPlot);
estAvg = probeRmse(matrices, fileLabels, indices, 'position', epochIndexList);
recordRmse(estAvg, 'position');
formatList{componentIndex} = '%.3f';
if ~subPlot
ylabel('$\delta \mathbf{p}_{WB} (m)$', 'Interpreter', 'latex');
end
set(gcf, 'Color', backgroundColor);
grid on;
outputfig = [result_dir, '/', 'rmse_t_WB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 5:7;
scale = 180 / pi;
drawColumnsInMultipleMatrices(matrices, fileLabels, {['x (', char(176), ')'], ['y (', char(176), ')'], ['z (', char(176), ')']}, ...
    indices, 0, fileColumnStyles, matrixColumnLineWidths, scale, subPlot);
estAvg = probeRmse(matrices, fileLabels, indices, 'orientation (deg)', epochIndexList, scale);
recordRmse(estAvg, 'orientation (deg)');
formatList{componentIndex} = '%.3f';
if ~subPlot
ylabel('$\delta \mathbf{\theta}_{WB} (^{\circ})$', 'Interpreter', 'latex');
end
set(gcf, 'Color', backgroundColor);
grid on;
outputfig = [result_dir, '/', 'rmse_theta_WB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 8:10;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'x (m/s)', 'y (m/s)', 'z (m/s)'}, ...
    indices, 0, fileColumnStyles, matrixColumnLineWidths, 1.0, subPlot);
estAvg = probeRmse(matrices, fileLabels, indices, 'velocity', epochIndexList);
recordRmse(estAvg, 'velocity');
formatList{componentIndex} = '%.3f';
if ~subPlot
ylabel('$\delta \mathbf{v}_{WB} (m/s)$', 'Interpreter', 'latex');
end
set(gcf, 'Color', backgroundColor);
grid on;
outputfig = [result_dir, '/', 'rmse_v_WB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 11:13;
scale = 180/pi;
drawColumnsInMultipleMatrices(matrices, fileLabels, {['x (', char(176), '/s)'], ['y (', char(176), '/s)'], ['z (', char(176), '/s)']}, ...
    indices, 0, fileColumnStyles, matrixColumnLineWidths, scale, subPlot);
estAvg = probeRmse(matrices, fileLabels, indices, 'bg (deg/s)', epochIndexList, scale);
recordRmse(estAvg, 'bg (deg/s)');
formatList{componentIndex} = '%.2f';
if ~subPlot
ylabel('$\delta \mathbf{b}_{g} (^{\circ}/s)$', 'Interpreter', 'latex');
end
set(gcf, 'Color', backgroundColor);
grid on;
outputfig = [result_dir, '/', 'rmse_b_g.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 14:16;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'x (m/s^2)', 'y (m/s^2)', 'z (m/s^2)'}, ...
    indices, 0, fileColumnStyles, matrixColumnLineWidths, 1.0, subPlot);
estAvg = probeRmse(matrices, fileLabels, indices, 'ba', epochIndexList);
recordRmse(estAvg, 'ba');
formatList{componentIndex} = '%.3f';
if ~subPlot
ylabel('$\delta \mathbf{b}_{a} (m/s^2)$', 'Interpreter', 'latex');
end
set(gcf, 'Color', backgroundColor);
grid on;
outputfig = [result_dir, '/', 'rmse_b_a.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

if ~shouldTerminate(matrices, 17)
figure;
indices = 17 + (0:8);
scale = 1000;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'x', 'y', 'z'}, ...
    indices, 0, fileColumnStyles, matrixColumnLineWidths, scale);
estAvg = probeRmse(matrices, fileLabels, indices, 'Tg (0.001)', epochIndexList, scale);
recordRmse(estAvg, 'Tg (0.001)');
formatList{componentIndex} = '%.2f';
ylabel('$\delta \mathbf{T}_{g} (0.001)$', 'Interpreter', 'latex');
set(gcf, 'Color', backgroundColor);
grid on;
outputfig = [result_dir, '/', 'rmse_T_g.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 26 + (0:8);
scale = 1000;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'x', 'y', 'z'}, ...
    indices, 0, fileColumnStyles, matrixColumnLineWidths, scale);
estAvg = probeRmse(matrices, fileLabels, indices, 'Ts (0.001)', epochIndexList, scale);
recordRmse(estAvg, 'Ts (0.001)');
formatList{componentIndex} = '%.2f';
ylabel('$\delta \mathbf{T}_{s} (0.001 \frac{rad/s}{m/s^2})$', 'Interpreter', 'latex');
set(gcf, 'Color', backgroundColor);
grid on;
outputfig = [result_dir, '/', 'rmse_T_s.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 35 + (0:8);
scale = 1000;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'x', 'y', 'z'}, ...
    indices, 0, fileColumnStyles, matrixColumnLineWidths, scale);
estAvg = probeRmse(matrices, fileLabels, indices, 'Ta (0.001)', epochIndexList, scale);
recordRmse(estAvg, 'Ta (0.001)');
formatList{componentIndex} = '%.2f';
ylabel('$\delta \mathbf{T}_{a} (0.001)$', 'Interpreter', 'latex');
set(gcf, 'Color', backgroundColor);
grid on;
outputfig = [result_dir, '/', 'rmse_T_a.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 44:46;
scale = 100;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'x', 'y', 'z'}, ...
    indices, 0, fileColumnStyles, matrixColumnLineWidths, scale);
estAvg = probeRmse(matrices, fileLabels, indices, 'p_CB (cm)', epochIndexList, scale);
recordRmse(estAvg, 'p_CB (cm)');
formatList{componentIndex} = '%.2f';
ylabel('$\delta \mathbf{p}_{CB} (cm)$', 'Interpreter', 'latex');
set(gcf, 'Color', backgroundColor);
grid on;
outputfig = [result_dir, '/', 'rmse_p_CB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 47:48;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'fx', 'fy'}, ...
    indices, 0, fileColumnStyles);
estAvg = probeRmse(matrices, fileLabels, indices, 'fxy', epochIndexList);
recordRmse(estAvg, 'fxy');
formatList{componentIndex} = '%.2f';
ylabel('$\delta \mathbf{f} (pixel)$', 'Interpreter', 'latex');
set(gcf, 'Color', backgroundColor);
grid on;
outputfig = [result_dir, '/', 'rmse_fxy.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 49:50;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'cx', 'cy'}, ...
    indices, 0, fileColumnStyles);
estAvg = probeRmse(matrices, fileLabels, indices, 'cxy', epochIndexList);
recordRmse(estAvg, 'cxy');
formatList{componentIndex} = '%.2f';
ylabel('$\delta \mathbf{c} (pixel)$', 'Interpreter', 'latex');
set(gcf, 'Color', backgroundColor);
grid on;
outputfig = [result_dir, '/', 'rmse_cxy.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 51:52;
scale = 1000;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'k1', 'k2'}, ...
    indices, 0, fileColumnStyles, matrixColumnLineWidths, scale);
estAvg = probeRmse(matrices, fileLabels, indices, 'k12 (0.001)', epochIndexList, scale);
recordRmse(estAvg, 'k12 (0.001)');
formatList{componentIndex} = '%.2f';
ylabel('$\delta \mathbf{k} (0.001)$', 'Interpreter', 'latex');
set(gcf, 'Color', backgroundColor);
grid on;
outputfig = [result_dir, '/', 'rmse_k.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 53:54;
scale = 1000;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'p1', 'p2'}, ...
    indices, 0, fileColumnStyles, matrixColumnLineWidths, scale);
estAvg = probeRmse(matrices, fileLabels, indices, 'p12 (0.001)', epochIndexList, scale);
recordRmse(estAvg, 'p12 (0.001)');
formatList{componentIndex} = '%.2f';
ylabel('$\delta \mathbf{p} (0.001)$', 'Interpreter', 'latex');
set(gcf, 'Color', backgroundColor);
grid on;
outputfig = [result_dir, '/', 'rmse_p.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
indices = 55:56;
scale = 1000;
drawColumnsInMultipleMatrices(matrices, fileLabels, {'t_d', 't_r'}, ...
    indices, 0, fileColumnStyles, matrixColumnLineWidths, scale);
estAvg = probeRmse(matrices, fileLabels, indices(1), 'td (ms)', epochIndexList, scale);
recordRmse(estAvg, 'td (ms)');
formatList{componentIndex} = '%.2f';
estAvg = probeRmse(matrices, fileLabels, indices(2), 'tr (ms)', epochIndexList, scale);
recordRmse(estAvg, 'tr (ms)');
formatList{componentIndex} = '%.2f';
ylabel('$\delta t_d \delta t_r (ms)$', 'Interpreter', 'latex');
set(gcf, 'Color', backgroundColor);
grid on;
outputfig = [result_dir, '/', 'rmse_td-tr.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);
end

% record the rmse at several epochs.
rmseOutputFile = [result_dir, '/', 'rmse_at_epochs.txt'];
fid = fopen(rmseOutputFile, 'w');
delimiter = ' ';
logRmseStartIndex = 1;
fprintf(fid, ['time(sec)', delimiter]);
for j = logRmseStartIndex:componentIndex
    fprintf(fid, ['%s', delimiter], componentList{j});
end
fprintf(fid, '\n');

for i = 1:length(rmseList)
    fprintf(fid, '%s\n', dataFiles{i});
    for k = 1 : size(rmseList{i}, 1)
        fprintf(fid, ['%.1f', delimiter], probeEpochs(k));
        for j = logRmseStartIndex : componentIndex
            fprintf(fid, [formatList{j}, delimiter], rmseList{i}(k, j));
        end
        fprintf(fid, '\n');
    end
end
fclose(fid);
fprintf('Saved rmse probe results to %s\n', rmseOutputFile);

    function recordRmse(dataAvgList, componentLabel)
        componentIndex = componentIndex + 1;
        for dataIndex = 1:length(dataAvgList)
            rmseList{dataIndex}(:, componentIndex) = dataAvgList{dataIndex};
        end
        componentList{componentIndex} = componentLabel;
    end
end

function estAvgList = probeRmse(matrices, fileLabels, columnIndices, componentLabel, ...
    epochIndexList, scale)
if nargin < 6
    scale = 1.0;
end
estAvgList = cell(length(matrices), 1);
for j = 1:length(matrices)
    estAvg = sqrt(sum(matrices{j}(epochIndexList, columnIndices).^2, 2)) * scale;
    fprintf(['RMSE in ', componentLabel, ' of ', fileLabels{j}, '\n']);
    disp(num2str(estAvg));
    estAvgList{j} = estAvg;
end
end

function quit = shouldTerminate(matrices, nextIndex)
quit = 0;
for i = 1:length(matrices)
    cols = size(matrices, 2);
    if cols < nextIndex
        quit = 1;
        return;
    end
end
end
