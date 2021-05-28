function drawImuData(imuFileList, labelList, accelIndices, gyroIndices)
numImuFiles = length(imuFileList);
dataList = cell(1, numImuFiles);
for i = 1:numImuFiles
    data = readmatrix(imuFileList{i}, 'NumHeaderLines', 1);
    dataList{i} = data;
end
line_styles = {{'-r', '-g', '-b', '-k'},...
    {'-.m', '-.c', '-.k', '-.y'}, ...
    {'-.r', '-.g', '-.b', '-.k'}};

close all;
figure;
legends = cell(3*length(labelList), 1);
for i = 1:numImuFiles
    drawColumnsInMatrix(dataList{i}, gyroIndices, 0, 180/pi, line_styles{i});
    legends{(i-1) * 3 + 1} = [labelList{i}, '-x'];
    legends{(i-1) * 3 + 2} = [labelList{i}, '-y'];
    legends{(i-1) * 3 + 3} = [labelList{i}, '-z'];
end
ylabel('gyro (deg/s)');
legend(legends);

figure;
for i = 1:numImuFiles
    drawColumnsInMatrix(dataList{i}, accelIndices, 0, 1, line_styles{i});
end
ylabel('accel (m/s^2)');
legend(legends);
end
