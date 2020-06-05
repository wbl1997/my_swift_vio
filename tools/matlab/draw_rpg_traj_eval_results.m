function draw_rpg_traj_eval_results(res_dir, algo, sessions, trials)
% draw ground truth and traj estimates organized in rpg traj evalution tool
% format.
% res_dir, algo, sessions, trials are related by
% for j in trials:
% est_filename{j} = [res_dir, algo, '/laptop_', algo, '_', sessions{i}, ...
%                 '/stamped_traj_estimate', num2str(j), '.txt'];

close all;
runs = 0:trials - 1;
for i = 1:length(sessions)
    est_files = cell(trials + 1, 1);
    if trials == 1
        est_files{1} = [res_dir, algo, '/laptop_', algo, '_', sessions{i}, ...
            '/stamped_traj_estimate.txt'];
    else
        for j = runs
            est_files{j+1} = [res_dir, algo, '/laptop_', algo, '_', sessions{i}, ...
                '/stamped_traj_estimate', num2str(j), '.txt'];
        end
    end
    est_files{end} = [res_dir, algo, '/laptop_', algo, '_', sessions{i}, '/stamped_groundtruth.txt'];
    styles = {'r', 'b', 'k', 'm', 'c'};
    legends = cell(length(est_files), 1);
    figure('units','normalized','outerposition',[0 0 1 1]);
    for j = 1:trials
        draw_columns_from_file(est_files{j}, 2:4, 1, {styles{j}}); hold on;
        legends{j} = ['est ', num2str(j-1)];
    end
    draw_columns_from_file(est_files{trials + 1}, 2:4, 1, {'g'}); 
    legends{trials + 1} = 'gt';
    legend(legends);
    title([algo, '-', sessions{i}]);
%     x = input("Enter anything except for '0' to continue:");
    x = 1;
    if x == 0
        disp('Exiting.');
        break;
    end
end
