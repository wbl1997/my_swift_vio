function draw_simul_results(est_file)
[result_dir, name_est, ext] = fileparts(est_file);
traj_est = find_traj_name(name_est);

algo_est = find_algo_name(name_est);
truth_file=[result_dir, '/', traj_est, '.txt'];
est_data = readmatrix(est_file, 'NumHeaderLines', 1);
if est_data(1, 1) > 1e9
    est_data(:, 1) = est_data(:, 1) * 0.000000001;
end

maxindex = max(size(est_data, 1));
data_range=1:maxindex;
truth_data = readmatrix(truth_file, 'NumHeaderLines', 1);
close all;
figure;
plot3(truth_data(:, 3), truth_data(:, 4), ...
    truth_data(:, 5)); hold on;
plot3(est_data(1, 3), est_data(1, 4), ...
    est_data(1, 5), 'ro', 'MarkerSize', 8);
steps = 10;
plot3(est_data(steps, 3), est_data(steps, 4), ...
    est_data(steps, 5), 'rs', 'MarkerSize', 8);
plot3(est_data(data_range, 3), est_data(data_range, 4), ...
    est_data(data_range, 5));


grid on; axis equal;
xlabel('x');
ylabel('y');
zlabel('z');
legend('truth', 'start', 'direction', algo_est);
title(traj_est);


figure;
index = 19:21;
drawColumnsInFile(est_file, index, 0, ...
    1, {'r', 'g', 'b'});
legend('p_x', 'p_y', 'p_z');

figure;
index = 22:24;
drawColumnsInFile(est_file, index, 0, ...
    1, {'r', 'g', 'b'});
legend('\theta_x', '\theta_y', '\theta_z');

figure;
index = 25:27;
drawColumnsInFile(est_file, index, 0, ...
    1, {'r', 'g', 'b'});
legend('v_x', 'v_y', 'v_z');

end

function traj_name = find_traj_name(name_est)
traj_name = '';
traj = {'Ball', 'Torus2', 'Torus', 'WavyCircle', 'Dot', 'Squircle', 'Circle', 'Motionless'};
for i = 1 : size(traj, 2)
    if contains(name_est, traj{i})
        traj_name = traj{i};
        return;
    end
end
end


function algo_name = find_algo_name(name_est)
algo_name = '';
algo = {'OKVIS', 'MSCKFEpi', 'MSCKF', 'TFVIO', 'General', 'SlidingWindowSmoother'};
for i = 1 : size(algo, 2)
    if contains(name_est, algo{i})
        algo_name = algo{i};
        return;
    end
end
end

