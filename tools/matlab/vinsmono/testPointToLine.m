function testPointToLine()
endpoints = [0, -5, 0;
    3, 0, 1];
pt = [-4, 5, 2];
[d, foot] = pointToLine(pt, endpoints(1, :), endpoints(2, :));
figure;
direction = endpoints(2, :) - endpoints(1, :);
direction = normalize(direction, 'norm');
endpoints_ex = [endpoints(1, :) - 5 * direction;
    endpoints(2, :) + 5*direction];
plot3(endpoints_ex(:, 1), endpoints_ex(:, 2), ...
    endpoints_ex(:, 3), 'k');
hold on;
plot3(pt(1), pt(2), pt(3), 'bs', 'MarkerSize', 10);
projectionline = [pt; foot];
plot3(foot(1), foot(2), foot(3), 'bo', 'MarkerSize', 10);
plot3(projectionline(:, 1), projectionline(:, 2), projectionline(:, 3), 'b');
axis equal; grid on;
xlabel('x');
ylabel('y');
zlabel('z');
end