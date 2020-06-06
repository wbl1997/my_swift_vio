function lineHandles = drawColumnsInMatrix(data, indices, scale, ...
    plot3d, lineStyles)
% first column of data is x-axis
% indices identify columns for y-axis
if nargin < 5
    lineStyles = {'-r', '-g', '-b', '-k', '.k', '.b', '-c', '-m', '-y'};
end
if nargin < 4
    plot3d = 0;
end
if nargin < 3
    scale = 1.0;
end

dimen = length(indices);
if plot3d
    lineHandles = plot3(data(:, indices(1)), data(:, indices(2)), ...
        data(:, indices(3)), lineStyles{1}); 
    grid on; axis equal;
    xlabel('x');
    ylabel('y');
    zlabel('z');
    return;
end
lineHandles = zeros(1, dimen);
lineHandles(1) = plot(data(:,1), data(:, indices(1))*scale, lineStyles{1});
hold on;
for i=2:dimen
    lineHandles(i) = plot(data(:,1), data(:, indices(i))*scale, lineStyles{i});
end
end
