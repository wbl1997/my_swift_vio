function drawColumnsInMultipleMatrices(matrices, matrixLabels, ...
    columnLabels, columnIndices, matrixColumnStyles, dataMultiplier)
% draw multiple columns in multiple matrices in one figure.
% warn: this function will modify matrices by aligning all timestamps in 
% matrices to a common startTime.
% The first column of each matrix is timestamp in secs.
% matrixLabels for each matrix
% columnLabels for each column
% matrixColumnStyles for each column in each matrix, 
% e.g., {{'r', 'g'}, {'b', 'k'}}
% dataMultiplier scale factor

if nargin < 6
    dataMultiplier = 1.0;
end
legend_list = cell(1, length(matrices) * length(columnLabels));
startTime = 0;
dataBegin = 0;
dataEnd = 0;
for i = 1:length(matrices)
    est_label = matrixLabels{i};
    if startTime < 1e-7
        startTime = matrices{i}(1, 1);
        dataBegin = matrices{i}(1, 1) - startTime;
        dataEnd = matrices{i}(end, 1) - startTime;
    end
    matrices{i}(:, 1) = matrices{i}(:, 1) - startTime;

    est_line_style = matrixColumnStyles{i};
    drawColumnsInMatrix(matrices{i}, columnIndices, dataMultiplier, ...
        false, est_line_style); hold on;
    for j = 1: length(columnLabels)
        legend_list((i - 1) * length(columnLabels) + j) = ...
            {[est_label, '-', columnLabels{j}]};
    end
end
legend(legend_list);
xlabel('time (sec)');
xlim([dataBegin-10, dataEnd+10]);
end