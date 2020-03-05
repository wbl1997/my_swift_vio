function [delta_o, range, delta_s, delta] = metrics_line_fitting(...
    est_file, position_indices, data_duration, ref_length, visualize_result)
% input
% data_duration in secs
% output
% delta_o loop error
% range: max distance between trajectory points and the starting point
% delta_s scale error
% delta transverse error

if nargin < 5
    visualize_result = 0;
end
if nargin < 4
    ref_length = 5;
end

delta_o = 1e8;
range=1e8;
delta_s = 1e8;
delta = 1e8;
fid = fopen(est_file);
firstline = fgetl(fid);
fclose(fid);
headerlines = 0;
if firstline == -1
    return;
end
if contains(firstline, '%') || contains(firstline, '#')
    headerlines = 1;
end
data = readmatrix(est_file, 'NumHeaderLines', headerlines);
if isempty(data)
    return;
end
if data(1, 1) > 1e9
    data(:, 1) = data(:, 1) * 0.000000001;
end
position_data = data(:, [1, position_indices]);
if (data(end, 1) - data(1, 1)) < 0.9 * data_duration
    return;
end
delta_o = sqrt(sum((position_data(end, 2:4) - position_data(1, 2:4)).^2));
datalength = size(position_data, 1);
range = max(sqrt(sum((position_data(:, 2:4) - repmat(position_data(1, 2:4), datalength, 1)).^2, 2)));

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
    endpoints = extend_endpoints(endpoints);
    plot3(endpoints(1, :), endpoints(2, :), endpoints(3, :), '--k');
    % draw the loop closure error
    start_end = position_data([1, end], 2:4);
    markerSize = 4;
    plot3(start_end(:, 1), start_end(:, 2), start_end(:, 3), '--b');
    plot3(start_end(1, 1), start_end(1, 2), start_end(1, 3), 'mo', 'MarkerSize', markerSize);
    plot3(start_end(2, 1), start_end(2, 2), start_end(2, 3), 'ms', 'MarkerSize', markerSize);
    annotate_line_fitting_iros2020(est_file);
    
    % draw the estimated length projection
    start_projection = [position_data(1, 2:4);...
        foot(1, 1:3)];
    plot3(start_projection(:, 1), start_projection(:, 2), start_projection(:, 3), '-r', 'LineWidth', 1);
    perp_farthest = [position_data(xindex, 2:4);...
        foot(xindex, 1:3)];
    plot3(perp_farthest(:, 1), perp_farthest(:, 2), perp_farthest(:, 3), '-r', 'LineWidth', 1);
    estimated_line = foot([1, xindex], 1:3);
    plot3(estimated_line(:, 1), estimated_line(:, 2), estimated_line(:, 3), '-ks', 'MarkerSize', markerSize);
    
    % draw the max length perpendicular line segment
%     perp = [position_data(dindex, 2:4);...
%         foot(dindex, 1:3)];
%     plot3(perp(:, 1), perp(:, 2), perp(:, 3), '-m+');
    
    % draw the parallal deviation bound
    
    upper_endpoints = parallel_line(endpoints, delta);
    lower_endpoints = parallel_line(endpoints, -delta);
    plot3(upper_endpoints(1, :), upper_endpoints(2, :), upper_endpoints(3, :), '-.k');
    plot3(lower_endpoints(1, :), lower_endpoints(2, :), lower_endpoints(3, :), '-.k');
    
    axis equal;
    grid on;
    xlabel('x (m)');
    ylabel('y (m)');
    zlabel('z (m)');
    view([90, 90]);
    titlestr = 'line_fitting';
    set(gcf, 'Color', 'None');
    [output_dir,name,ext] = fileparts(est_file);
    outputfig = [output_dir, '/', titlestr, '.eps'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
end
end
function new_endpoints = extend_endpoints(endpoints)
% endpoint 3x2
% extend left and right each by 20%
d = norm(endpoints(:, 2) - endpoints(:, 1));
direction = normalize(endpoints(:, 2) - endpoints(:, 1), 'norm');
new_endpoints = [endpoints(:, 1) - direction * d * 0.2, endpoints(:, 2) + direction * d * 0.2];
end

function parallel_endpoints = parallel_line(endpoints, offset)
% assume the normal vector has zero component along z axis.
l = normalize(endpoints(:, 2) - endpoints(:, 1), 'norm');
nx = l(2) / sqrt((1- l(3)^2)^2 + (l(2) * l(3))^2 + (l(1) * l(3))^2);
ny = - nx * l(1)/l(2);
n = [nx; ny; 0];
assert(abs(norm(n) - 1) < 1e-7);
assert(n' * l < 1e-7);
assert(abs(norm(cross(n, l)) - 1) < 1e-7);

parallel_endpoints = endpoints + repmat(n, 1, 2) * offset; 
end
