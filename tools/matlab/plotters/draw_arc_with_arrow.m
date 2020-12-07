function draw_arc_with_arrow()
% a sample showing how to use arc_with_arrow.
[arc_point, arc_arrow] = arc_with_arrow(5, -pi/2, pi/2, 1);
figure;
plot3(arc_point(1, :), arc_point(2, :), arc_point(3, :), 'b'); hold on;
plot3(arc_arrow(1, :), arc_arrow(2, :), arc_arrow(3, :), 'b');
xlabel('x');
ylabel('y');
zlabel('z');
axis equal;
grid on;

end
