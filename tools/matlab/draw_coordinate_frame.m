function draw_coordinate_frame(t_WC, R_WC, scale)
% t_WC 3x1
% R_WC 3x3
% scale side length
origin = t_WC;
axis_tips = repmat(origin, 1, 3) + R_WC * scale;
colors = {'r', 'g', 'b'};
for j = 1:3
    axis_segment = [origin, axis_tips(:, j)];
    plot3(axis_segment(1, :), axis_segment(2, :), ...
        axis_segment(3, :), colors{j});
end
end
