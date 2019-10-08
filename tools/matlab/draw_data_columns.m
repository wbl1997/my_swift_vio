function draw_data_columns(data, triple_index, scalar, plot3d)
if nargin < 4
    plot3d = 0;
end
if nargin < 3
    scalar = 1.0;
end

dimen = length(triple_index);
line_styles = {'-r', '-g', '-b', '-k', '.k', '.b', '-c', '-m', '-y'};
if plot3d
    plot3(data(:, triple_index(1)), data(:, triple_index(2)), ...
        data(:, triple_index(3)), line_styles{1}); 
    grid on; axis equal;
    xlabel('x');
    ylabel('y');
    zlabel('z');
    return;
end
plot(data(:,1), data(:, triple_index(1))*scalar, line_styles{1}); hold on;
for i=2:dimen
    plot(data(:,1), data(:, triple_index(i))*scalar, line_styles{i});
end
end
