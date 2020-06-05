function drawColumnsInMultipleFiles(dataFiles, fileLabels, ...
    columnLabels, columnIndices, fileColumnStyles, dataMultiplier)
matrices = cell(length(dataFiles), 1);
for i = 1:length(dataFiles)
    file = dataFiles{i};
    matrices{i} = readmatrix(file, 'NumHeaderLines', 1);
end
drawColumnsInMultipleMatrices(matrices, fileLabels, ...
    columnLabels, columnIndices, fileColumnStyles, dataMultiplier);
end