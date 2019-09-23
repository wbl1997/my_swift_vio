function draw_ekf_triplet_with_std(data, triple_index, ...
    triple_std_index, scalar)
% dimen should be smaller than 5
if nargin < 4
    scalar = 1.0;
end
dimen = length(triple_index);

line_styles = {'-r', '-g', '-b', '-k', '.k', '.b', '-c', '-m', '-y'};
std_line_styles = {'--r', '--g', '--b', '--k', '-.k', '-.b', ...
    '--c', '--m', '--y'};

plot(data(:,1), data(:, triple_index(1))*scalar, line_styles{1}); hold on;
for i=2:dimen
    plot(data(:,1), data(:, triple_index(i))*scalar, line_styles{i});
end

for i=1:dimen
    plot(data(:,1), (3*data(:, triple_std_index(i)) + ...
        data(:, triple_index(i)))*scalar, std_line_styles{i});
    plot(data(:,1), (-3*data(:,triple_std_index(i)) + ...
        data(:, triple_index(i)))*scalar, std_line_styles{i});
end

set(gcf, 'Color', 'w');
xlabel('time[sec]');
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

