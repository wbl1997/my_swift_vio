function circular_curve_and_features()
% simulate a circular sinusoidal trajectory and points on a wall.
% refer to Camera-IMU-based localization: Observability
% analysis and consistency improvement.

% motion
% translation part
step = 2*pi/200;
t = 0:step:2*pi;
[x, y, z, ~, ~, ~] = trajectory_positions(t);
plot3(x, y, z); hold on;

% draw the directions
step = 2*pi/10;
t_sparse = 0:step:2*pi;
[x_sparse, y_sparse, z_sparse, u, v, w] = trajectory_positions(t_sparse);
quiver3(x_sparse, y_sparse, z_sparse, u, v, w);
axis equal;
xlabel('x [m]');
ylabel('y [m]');
zlabel('z [m]');

% rotation part
rotations = trajectory_orientations(t);

for i = 1:size(rotations, 2)
    % draw local frame
    if mod(i, 10) == 0
        origin = [x(i); y(i); z(i)];
        axis_tips = repmat(origin, 1, 3) + rotations{i} * nearest_depth();
        colors = {'r', 'g', 'b'};
        for j = 1:3
            axis_segment = [origin, axis_tips(:, j)];
            plot3(axis_segment(1, :), axis_segment(2, :), axis_segment(3, :), colors{j});
        end
    end
end

% structure
step = 2*pi/40;
theta = 0:step:2*pi;
fx = get_wall_radius() * sin(theta);
fy = get_wall_radius() * cos(theta);

for i = 1:7
    fz = ones(1, size(theta, 2)) * (i - 4) * get_wall_height() * 0.5 / 3.0;
    noise = (rand(size(theta)) - 0.5) * 0.1;
    % add noise to fz if you will. But it seems not necessary.
    plot3(fx, fy, fz, 'rx')
end
end

function [x, y, z, u, v, w] = trajectory_positions(t)
% speed about 1.2 m/s
trajectory_radius = 5;
angular_rate = 1.2 / trajectory_radius;
epochs = t/angular_rate;
disp(['One round takes time ', num2str(epochs(end)), ' secs.']);
wall_radius = get_wall_radius();
halfz = get_wall_height() * 0.5;
nearest_depth = sqrt(wall_radius * wall_radius - trajectory_radius * trajectory_radius);
tan_vertical_half_Fov = halfz / nearest_depth;

frequency_num = 10;
% decrease the coefficient to make more point visible.
coeff = 0.9;
wave_height = coeff * (tan_vertical_half_Fov * trajectory_radius / frequency_num);
x = trajectory_radius*cos(t);
y = trajectory_radius*sin(t);
z = wave_height * cos(frequency_num*t);
% gradient
u = - sin(t);
v = cos(t);
w = -wave_height * frequency_num * sin(frequency_num*t) / trajectory_radius;
end

function d = nearest_depth()
trajectory_radius = 5;
wall_radius = get_wall_radius();
d = sqrt(wall_radius * wall_radius - trajectory_radius * trajectory_radius);
end

function wr = get_wall_radius()
wr = 6;
end

function z = get_wall_height()
z=2;
end

function rotations = trajectory_orientations(t)
trajectory_radius = 5;
wall_radius = get_wall_radius();
halfz = 1;
nearest_depth = sqrt(wall_radius * wall_radius - trajectory_radius * trajectory_radius);
tan_vertical_half_Fov = halfz / nearest_depth;

frequency_num = 10;
% decrease the coefficient to make more point visible.
coeff = 0.9;
wave_height = coeff * (tan_vertical_half_Fov * trajectory_radius / frequency_num);

rotations = cell(size(t));

for i = 1:size(t, 2)
    F = [- trajectory_radius*sin(t(i));
        trajectory_radius*cos(t(i));
        -wave_height * frequency_num * sin(frequency_num*t(i));];
    F = normalize(F, 'norm');
    L = [-trajectory_radius*cos(t(i));
        -trajectory_radius*sin(t(i));
        0];
    L = normalize(L, 'norm');
    
    assert(dot(F, L) < 1e-7);
    U = cross(F, L);
    U = normalize(U, 'norm');
    R_WB = [F, L, U];
    Rx = rotx(30 * pi/180 * sin(5 * t(i))); % add rotation about another axis.
    rotations{i} = R_WB * Rx;
end
end

function Rx = rotx(theta)
ct = cos(theta);
st = sin(theta);
Rx = [1, 0, 0; 0, ct, st; 0, -st, ct];
end

