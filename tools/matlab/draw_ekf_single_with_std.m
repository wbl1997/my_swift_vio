
function draw_ekf_single_with_std(data, single_index, ...
    single_std_index, scalar)
if nargin < 4
    scalar = 1.0;
end
plot(data(:,1), data(:, single_index)*scalar, '-r'); hold on;

plot(data(:,1), (3*data(:, single_std_index) + data(:, single_index))*scalar, '--g');
plot(data(:,1), (-3*data(:,single_std_index) + data(:, single_index))*scalar, '--g');
set(gcf, 'Color', 'w');
xlabel('time[sec]');
legend('x', '3\sigma_x');
grid on;
end

