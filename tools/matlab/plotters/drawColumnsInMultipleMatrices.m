function drawColumnsInMultipleMatrices(matrices, matrixLabels, ...
    columnLabels, columnIndices, plot3d, ...
    matrixColumnStyles, matrixColumnLineWidths, dataMultiplier, subPlot)
% draw multiple columns in multiple matrices in one figure.
% warn: this function will modify matrices by aligning all timestamps in
% matrices to a common startTime.
% The first column of each matrix is timestamp in secs.
% matrixLabels for each matrix
% columnLabels for each column
% matrixColumnStyles for each column in each matrix,
% e.g., {{'r', 'g'}, {'b', 'k'}}
% dataMultiplier scale factor
% subPlot if 1, plot dimensions in subplots and this voids plot3d.

if nargin < 9
    subPlot = 0;
end

if nargin < 8
    dataMultiplier = 1.0;
end
if nargin < 7
    matrixColumnLineWidths = {{1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1}, ...
        {1,1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1,1}};
end
if nargin < 6
    matrixColumnStyles =  {
        {'-r', '-g', '-b', '-k', '.k', '.b', ...
        '-c', '-m', '-y'}, ...
        {'--r', '--g', '--b', '--k', '-.k', '-.b', ...
        '--c', '--m', '--y'}, ...
        {':r', ':g', ':b', ':k', '-.r', '-.g', ...
        ':c', ':m', ':y'}, ...
        {'.r', '.g', '.b', '.k', '-.r', '-.g', ...
        ':c', ':m', ':y'}};
end
if nargin < 5
    plot3d = 0;
end
legendList = cell(1, length(matrices) * length(columnLabels));
startTime = 0;
dataBegin = 0;
dataEnd = 0;

% data preparation
for i = 1:length(matrices)
    if startTime < 1e-7
        startTime = matrices{i}(1, 1);
        dataBegin = matrices{i}(1, 1) - startTime;
        dataEnd = matrices{i}(end, 1) - startTime;
    end
    matrices{i}(:, 1) = matrices{i}(:, 1) - startTime;
end

if subPlot == 1
    numPanels = length(columnIndices);
    
    axes = zeros(1, numPanels);
    for p = 1:numPanels
        axes(p) = subplot(numPanels, 1, p);
        for i = 1:length(matrices)
            label = matrixLabels{i};
            lineStyle = matrixColumnStyles{i};
            drawColumnsInMatrix(matrices{i}, columnIndices(p), plot3d, ...
                dataMultiplier, lineStyle, matrixColumnLineWidths{i});
            
            legendList((p - 1) * length(matrices) + i) = {label};
        end
        ylabel(columnLabels{p});
        if p == 1
            legend(legendList{(p - 1) * length(matrices) + (1: length(matrices))}, 'Location', 'Best');
        end
    end
    linkaxes(axes, 'x');
else
    for i = 1:length(matrices)
        label = matrixLabels{i};        
        lineStyle = matrixColumnStyles{i};
        drawColumnsInMatrix(matrices{i}, columnIndices, plot3d, ...
            dataMultiplier, lineStyle, matrixColumnLineWidths{i}); hold on;
        for j = 1: length(columnLabels)
            legendList((i - 1) * length(columnLabels) + j) = ...
                {[label, '-', columnLabels{j}]};
        end
    end
    legend(legendList);
end
if ~plot3d
    xlabel('time (sec)');
    xlim([dataBegin-10, dataEnd+10]);
end
end