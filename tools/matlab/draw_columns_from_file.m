function draw_columns_from_file(data_file, indices, plot3d, line_styles)
% draw columns of data loaded from data_file
% the first row of the data file will be removed as a header
% the first column serves as the x axis
% indices are range of columns to draw, 1 based, e.g., 2:4
% plot3d if 1, plot3 will be used for the first 3 dim of indices,
% default 0
% line_styles line styles for each dim.
if nargin < 4
    line_styles = {'-r', '-g', '-b', '-k', '.k', '.b', '-c', '-m', '-y'};
end
if nargin < 3
    plot3d = 0;
end
data = readmatrix(data_file, 'NumHeaderLines', 1);
if data(1, 1) > 1e9
    data(:, 1) = data(:, 1) * 0.000000001;
end
data(:, 1) = data(:, 1) - data(1, 1);
dimen = length(indices);
draw_data_columns(data, indices, 1.0, plot3d, line_styles);
if plot3d
    legend('xyz');
else if dimen == 3
        legend('x', 'y', 'z');
    else
        labels = cell(dimen, 1);
        for j = 1:dimen
            labels{j} = num2str(j);
        end
        legend(labels);
    end
end
grid on;
end
