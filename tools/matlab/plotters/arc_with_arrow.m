function [arc_point, arc_arrow] = arc_with_arrow(...
    radius, start_angle, end_angle, dir)
% dir = 1; % 1 anticlock, -1 clock
% arc_point 3xn
% arc_arrow at the end of the arc 3x3

steps = 20;
delta = (end_angle - start_angle) / steps;
arc_point = zeros(3, steps + 1);
arc_arrow = zeros(3, 2);

arrow_anchor_index = 0;
if dir == 1
    arrow_anchor_index = steps;
end
for i = 0 : steps
    theta = i * delta + start_angle;
    ct = cos(theta);
    st = sin(theta);
    arc_point(:, i + 1) = [ct, st, 0]';
    if i == arrow_anchor_index
        % arrows parameters
        h = 0.05*radius; % height
        w = 0.05*radius; % width
       
        a = [-w/2 0 w/2;
            -dir*h  0 -dir*h;
            0, 0, 0];
        
        R = [ct -st 0;
            st  ct 0;
            0  0  1];
        arc_arrow = repmat(arc_point(:, i + 1), 1, 3) + R*a;
    end
end
end