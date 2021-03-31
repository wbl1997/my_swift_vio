function drawTrajectoryWithCoordinateFrames(matrixList, labels, txyzIndices, ...
    qxyzwIndices)
% draw a trajectory and tips of local coordinate frames at specific 
% locations
% matrixList a cell of matrices, each row of each matrix containing a pose.
% at most four matrices are supported.
% labels are cell array 1xn for each matrix.
% txyzIndices indices of txyz in all matrices.
% qxyzwIndices indices of qxyzw in all matrices.

if nargin < 3
    qxyzwIndices = 5:8;
end
if nargin < 2
    txyzIndices = 2:4;
end
colors = {'r', 'g', 'b', 'k'};
maxCoordinateFrames = 400;

count = min(length(matrixList), 4);
lineHandles = zeros(1, count);

for j = 1:count
    lineHandles(j) = drawColumnsInMatrix(matrixList{j}, txyzIndices, 1, ...
        1, colors(j));
    hold on;
    increment = floor(size(matrixList{j}, 1) / maxCoordinateFrames);
    for i = 1:increment:size(matrixList{j}, 1)
        rot = rotmat(quaternion(matrixList{j}(i, [qxyzwIndices(4), ...
            qxyzwIndices(1:3)])), 'point');
        drawCoordinateFrame(matrixList{j}(i, txyzIndices)', rot, 1);
    end
end
legend(lineHandles, labels);
end
