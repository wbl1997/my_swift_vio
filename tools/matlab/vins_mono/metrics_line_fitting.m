function [delta_o, delta_s, delta] = metrics_line_fitting(...
    est_file, position_indices, data_duration, ref_length)
% data_duration in secs
% output
% delta_o loop error
% delta_s scale error
% delta transverse error
visualize_result = 1;
delta_o = 1e8;
delta_s = 1e8;
delta = 1e8;
ref_length = 5;
fid = fopen(est_file);
firstline = fgetl(fid);
fclose(fid);
headerlines = 0;
if contains(firstline, '%') || contains(firstline, '#')
    headerlines = 1;
end
data = readmatrix(est_file, 'NumHeaderLines', headerlines);
if data(1, 1) > 1e9
    data(:, 1) = data(:, 1) * 0.000000001;
end
position_data = data(:, [1, position_indices]);
if (data(end, 1) - data(1, 1)) < 0.9 * data_duration
    return
end
delta_o = sqrt(sum((position_data(end, 2:4) - position_data(1, 2:4)).^2));
endpoints = fit3DLine(position_data(:, 2:4)');
[d, foot] = point_to_line(position_data(:, 2:4), endpoints(:, 1)', endpoints(:, 2)');
dist2_to_start_projection = sum((foot - repmat(foot(1, :), size(foot, 1), 1)).^2, 2);
[xval, xindex] = max(dist2_to_start_projection);
est_length = sqrt(xval);
delta_s = abs(est_length - ref_length) / ref_length;
[maxdelta, dindex] = max(d);
delta = mean(d);

if visualize_result == 1
close all;

figure;
plot3(position_data(:, 2), position_data(:, 3), position_data(:, 4), 'b-');
hold on;
plot3(endpoints(1, :), endpoints(2, :), endpoints(3, :), 'k');
% draw the estimated length projection
plot3(foot(1, 1), foot(1, 2), foot(1, 3), 'ro');
perp_farthest = [position_data(xindex, 2:4);...
    foot(xindex, 1:3)];
plot3(perp_farthest(:, 1), perp_farthest(:, 2), perp_farthest(:, 3), '-rs');
% draw the max length perpendicular line segment
perp = [position_data(dindex, 2:4);...
    foot(dindex, 1:3)];
plot3(perp(:, 1), perp(:, 2), perp(:, 3), '-b+');
axis equal;
grid on;

xlabel('x');
ylabel('y');
zlabel('z');
titlestr = 'line_fitting';
[output_dir,name,ext] = fileparts(est_file);
outputfig = [output_dir, '/', titlestr, '.eps'];
if exist(outputfig, 'file')==2
    delete(outputfig);
end
export_fig(outputfig);
end
end