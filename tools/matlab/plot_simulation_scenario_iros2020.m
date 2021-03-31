function plot_simulation_scenario_iros2020(data_dir, scene)
% to generate the figures for the iros 2020 paper
% plot_simulation_scenario('/RSS-2020/results/msckf_simulation', 'Dot')
% plot_simulation_scenario('/RSS-2020/results/msckf_simulation', 'Motionless')
% plot_simulation_scenario('/RSS-2020/results/msckf_simulation', 'Squircle')
% plot_simulation_scenario('/RSS-2020/results/msckf_simulation', 'WavyCircle')
export_fig_path = '/media/jhuai/Seagate/jhuai/tools/export_fig/';
addpath(export_fig_path);
landmark_txt = [data_dir, '/', scene, '_Points.txt'];
landmarks = readmatrix(landmark_txt, 'NumHeaderLines', 1);
trajectory_txt = [data_dir, '/', scene, '.txt'];
trajectory = readmatrix(trajectory_txt, 'NumHeaderLines', 1);
r = [3:5];
q = [6:9];
% body frame FLU, camera frame forward motion
T_BC = [0, 0, 1, 0;
    -1, 0, 0, 0;
    0, -1, 0, 0;
    0, 0, 0, 1];
T_WB = eye(4);
anchorIndex = 37;
T_WB(1:3, 4) = trajectory(anchorIndex, r)';
T_WB(1:3, 1:3) = rotmat(quaternion([trajectory(anchorIndex, q(4)), ...
    trajectory(anchorIndex, q(1:3))], 'point');
T_WC = T_WB * T_BC;
t_WC = T_WC(1:3, 4);
R_WC = T_WC(1:3, 1:3);

figure;

plot3(trajectory(:, r(1)), trajectory(:, r(2)), ...
    trajectory(:, r(3)), '-k', 'LineWidth', 1); hold on;
plot3(landmarks(:, 2), landmarks(:, 3), landmarks(:, 4), 'xb');
w = 6.4;
h = 3.6;
f = 5.0;
scale = 0.2;
draw_camera_frustum(t_WC, R_WC, w, h, f, scale);
zlim([-2.0, 2.0]);
if strcmp(scene, 'Dot')
    [arc_point, arc_arrow] = arc_with_arrow(5, -210 / 180 * pi, -60 / 180 * pi, 1);
    plot3(arc_point(1, :), arc_point(2, :), arc_point(3, :), 'b');
    plot3(arc_arrow(1, :), arc_arrow(2, :), arc_arrow(3, :), 'b', 'LineWidth', 1);
end
legend_list = {'trajectory', 'landmarks'};
% legend(legend_list);
% title([scene, ' p_{GB}']);
xlabel('x (m)');
ylabel('y (m)');
zlabel('z (m)');
set(gcf, 'Color', 'None');
axis equal;
grid on;
outputfig = [data_dir, '/', scene, '_p_GB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

end
