function line_handles = draw_ekf_triplet_with_std(data, triple_index, ...
    triple_std_index, scalar, iserror, line_styles, std_line_styles)
% dimen should be smaller than 5
if nargin < 7
    std_line_styles = {'--r', '--g', '--b', '--k', '-.k', '-.b', ...
    '--c', '--m', '--y'};
end
if nargin < 6
    line_styles = {'-r', '-g', '-b', '-k', '.k', '.b', '-c', '-m', '-y'};
end
if nargin < 5
    iserror = 0;
end
if nargin < 4
    scalar = 1.0;
end
dimen = length(triple_index);
line_handles = zeros(1, dimen * 3);
line_handles(1:dimen) = draw_data_columns(data, triple_index, scalar, 0, line_styles);

if iserror
    for i=1:dimen
        line_handles(dimen + i) = plot(data(:,1), (3*data(:, triple_std_index(i)))*scalar, std_line_styles{i});
        line_handles(dimen*2 + i) = plot(data(:,1), (-3*data(:,triple_std_index(i)))*scalar, std_line_styles{i});
    end
else
    for i=1:dimen
        line_handles(dimen + i) = plot(data(:,1), (3*data(:, triple_std_index(i)) + ...
            data(:, triple_index(i)))*scalar, std_line_styles{i});
        line_handles(dimen*2 + i) = plot(data(:,1), (-3*data(:,triple_std_index(i)) + ...
            data(:, triple_index(i)))*scalar, std_line_styles{i});
    end
end

set(gcf, 'Color', 'w');
xlabel('time (sec)');
if dimen == 3
    legend('x', 'y', 'z', '3\sigma_x', '3\sigma_y', '3\sigma_z');
else
    if dimen == 9
    legend('1', '2', '3', '4', '5', '6', '7', '8', '9', ...
        '3\sigma_1', '3\sigma_2', '3\sigma_3', '3\sigma_4', ...
        '3\sigma_5', '3\sigma_6', '3\sigma_7', '3\sigma_8', '3\sigma_9');
    end
end
grid on;
end

