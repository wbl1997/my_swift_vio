function draw_simul_results(est_file, cmp_file, maxindex)

[result_dir, name_est, ext] = fileparts(est_file);
[~, name_cmp, ext] = fileparts(cmp_file);
traj_est = find_traj_name(name_est);
traj_cmp = find_traj_name(name_cmp);
if ~strcmp(traj_est, traj_cmp)
    disp(['Different traj in two files ', traj_est, ' and ', traj_cmp]);
end
algo_est = find_algo_name(name_est);
algo_cmp = find_algo_name(name_cmp);
truth_file=[result_dir, '/', traj_est, '.txt'];
est_data = readmatrix(est_file, 'NumHeaderLines', 1);
if est_data(1, 1) > 1e9
    est_data(:, 1) = est_data(:, 1) * 0.000000001;
end
draw_cmp = 0;
if isfile(cmp_file)
    cmp_data = readmatrix(cmp_file, 'NumHeaderLines', 1);
    if cmp_data(1, 1) > 1e9
        cmp_data(:, 1) = cmp_data(:, 1) * 0.000000001;
    end
    draw_cmp = 1;
else
    cmp_data = [];
end
data_range=1:maxindex;
truth_data = readmatrix(truth_file, 'NumHeaderLines', 1);
close all;
figure;
plot3(truth_data(:, 3), truth_data(:, 4), ...
    truth_data(:, 5)); hold on;
plot3(est_data(data_range, 3), est_data(data_range, 4), ...
    est_data(data_range, 5)); hold on;
if draw_cmp
plot3(cmp_data(data_range, 3), cmp_data(data_range, 4), ...
    cmp_data(data_range, 5));
end
grid on; axis equal;
xlabel('x');
ylabel('y');
zlabel('z');
legend('truth', algo_est, algo_cmp);
title(traj_est);

plot_dim(3, 'px', truth_data, est_data, cmp_data, ...
    algo_est, algo_cmp, traj_est);
plot_dim(4, 'py', truth_data, est_data, cmp_data, ...
    algo_est, algo_cmp, traj_est);
plot_dim(5, 'pz', truth_data, est_data, cmp_data, ...
    algo_est, algo_cmp, traj_est);


plot_dim(6, 'qx', truth_data, est_data, cmp_data, ...
    algo_est, algo_cmp, traj_est);
plot_dim(7, 'qy', truth_data, est_data, cmp_data, ...
    algo_est, algo_cmp, traj_est);
plot_dim(8, 'qz', truth_data, est_data, cmp_data, ...
    algo_est, algo_cmp, traj_est);
plot_dim(9, 'qw', truth_data, est_data, cmp_data, ...
    algo_est, algo_cmp, traj_est);

plot_dim(10, 'vx', truth_data, est_data, cmp_data, ...
    algo_est, algo_cmp, traj_est);
plot_dim(11, 'vy', truth_data, est_data, cmp_data, ...
    algo_est, algo_cmp, traj_est);
plot_dim(12, 'vz', truth_data, est_data, cmp_data, ...
    algo_est, algo_cmp, traj_est);

end

function traj_name = find_traj_name(name_est)
traj_name = '';
traj = {'Ball', 'Torus', 'Circle', 'Dot', 'Squircle'};
for i = 1 : size(traj, 2)
  if contains(name_est, traj{i})
      traj_name = traj{i};
      return;
  end
end
end


function algo_name = find_algo_name(name_est)
algo_name = '';
algo = {'OKVIS', 'MSCKFEpi', 'MSCKF', 'TFVIO', 'General'};
for i = 1 : size(algo, 2)
  if contains(name_est, algo{i})
      algo_name = algo{i};
      return;
  end
end
end

function plot_dim(index, labely, truth_data, est_data, cmp_data, ...
    algo_est, algo_cmp, traj_name)
figure;
plot(truth_data(:, 1), truth_data(:, index), 'g-'); hold on;
plot(est_data(:, 1), est_data(:, index), 'k-');
if ~isempty(cmp_data)
    plot(cmp_data(:, 1), cmp_data(:, index), 'b-');
    legend('truth', algo_est, algo_cmp);
else
    legend('truth', algo_est);
end
ylabel(labely);
title(traj_name);
end
