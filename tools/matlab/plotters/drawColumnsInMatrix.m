function lineHandles = drawColumnsInMatrix(data, indices, plot3d, ...
    scale, lineStyles, lineWidths)
% first column of data is x-axis
% indices identify columns for y-axis
if nargin < 6
    lineWidths = {1, 1, 1, 1, 1, 1, 1, 1, 1};
end
if nargin < 5
    lineStyles = {'-r', '-g', '-b', '-k', '.k', '.b', '-c', '-m', '-y'};
end
if nargin < 4
    scale = 1.0;
end
if nargin < 3
    plot3d = 0;
end

dimen = length(indices);
if plot3d
    lineHandles = plot3(data(:, indices(1)), data(:, indices(2)), ...
        data(:, indices(3)), lineStyles{1}, 'LineWidth', lineWidths{1}); 
    grid on; axis equal;
    xlabel('x (m)');
    ylabel('y (m)');
    zlabel('z (m)');
    return;
end
lineHandles = zeros(1, dimen);
lineHandles(1) = plot(data(:,1), data(:, indices(1))*scale, lineStyles{1}, 'LineWidth', lineWidths{1});
hold on;
for i=2:dimen
    lineHandles(i) = plot(data(:,1), data(:, indices(i))*scale, lineStyles{i}, 'LineWidth', lineWidths{i});
end
end
