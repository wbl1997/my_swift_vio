function circular_curve_and_features()
% simulate a circular sinusoidal trajectory and points on a wall.
% refer to Camera-IMU-based localization: Observability
% analysis and consistency improvement.
close all;
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

box_landmark_grid();

end

% structure 1
function cylinder_landmark_grid()
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

% structure 2
function box_landmark_grid()
xyincrement = 1.0;
zincrement = 0.5;
wr = get_wall_radius();
h = get_wall_height();
xycoordinates = -wr:xyincrement:wr;
zcoordinates = -h*0.5:zincrement:h*0.5;
assert(size(zcoordinates, 2)==7);
[XY, Z] = meshgrid(xycoordinates, zcoordinates);
X = - get_wall_radius() * ones(size(XY));
plot3(X, XY, Z, 'rx'); % left
plot3(-X, XY, Z, 'rx'); % right
plot3(XY, X, Z, 'rx'); % back
plot3(XY, -X, Z, 'rx'); % front
end

function [x, y, z, u, v, w] = trajectory_positions(t)
% speed about 1.2 m/s
trajectory_radius = get_trajectory_radius();
angular_rate = 1.2 / trajectory_radius;
epochs = t/angular_rate;
disp(['One round takes time ', num2str(epochs(end)), ' secs.']);
wh = wave_height();
frequency_num = get_frequency_number();
[x, y, z] = position(t, trajectory_radius, wh, frequency_num);
% gradient
u = - sin(t);
v = cos(t);
w = -wh * frequency_num * sin(frequency_num*t) / trajectory_radius;
end

function [x, y, z] = position(t, trajectory_radius, wave_height, frequency_num)
% t_WB_B
x = trajectory_radius*cos(t);
y = trajectory_radius*sin(t);
z = wave_height * cos(frequency_num*t);
end

function d = nearest_depth()
trajectory_radius = get_trajectory_radius();
wall_radius = get_wall_radius();
d = sqrt(wall_radius * wall_radius - trajectory_radius * trajectory_radius);
end

function wh = wave_height()
trajectory_radius = get_trajectory_radius();
halfz = 0.5*get_wall_height();
nd = nearest_depth();
tan_vertical_half_Fov = halfz / nd;

frequency_num = get_frequency_number();
% decrease the coefficient to make more point visible.
coeff = 0.9;
wh = coeff * (tan_vertical_half_Fov * trajectory_radius / frequency_num);
disp(['wave height in rotation ', num2str(wh)]);
end

function wr = get_wall_radius()
wr = 5;
end

function z = get_wall_height()
z=3;
end

function tr = get_trajectory_radius()
tr = 4;
end

function fn = get_frequency_number()
fn = 10;
end

function Rx = rotx(theta)
ct = cos(theta);
st = sin(theta);
Rx = [1, 0, 0; 0, ct, st; 0, -st, ct];
end

function R_WB = orientation(t, trajectory_radius, wave_height, frequency_num)
F = [- trajectory_radius*sin(t);
        trajectory_radius*cos(t);
        -wave_height * frequency_num * sin(frequency_num*t);];
    F = F / sqrt(F' * F);
    L = [-cos(t); -sin(t); 0];
    U = cross(F, L);
    U = U / sqrt(U' * U);
    R_WB = [F, L, U];
    Rx = rotx(30 * pi/180 * sin(5 * t)); % add rotation about another axis.
    R_WB = R_WB * Rx;
end

function rotations = trajectory_orientations(t)
wh = wave_height();
rotations = cell(size(t));
for i = 1:size(t, 2)
    rotations{i} = orientation(t(i), get_trajectory_radius(), wh, ...
        get_frequency_number());
end
end



