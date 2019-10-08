function draw_columns_from_file(data_file, indices, plot3d)
if nargin < 3
    plot3d = 0;
end
data = readmatrix(data_file, 'NumHeaderLines', 1);
close all;
dimen = length(indices);
figure;
draw_data_columns(data, indices, 1.0, plot3d);
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
