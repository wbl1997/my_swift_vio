function drawRpgTrajectoryEvaluationResults(...
    res_dir, algo, sessions, trials, excludeResultOfHugeDrift)
% draw ground truth and traj estimates organized in rpg traj evalution tool
% format.
% res_dir, algo, sessions, trials are related by
% for j in trials:
% est_filename{j} = [res_dir, algo, '/laptop_', algo, '_', sessions{i}, ...
%                 '/stamped_traj_estimate', num2str(j), '.txt'];

if nargin < 5
    excludeResultOfHugeDrift = 1;
end

driftTolerance = 1000;
runs = 0:(trials - 1);
for i = 1:length(sessions)
    est_files = cell(trials + 1, 1);
    if trials == 1
        est_files{1} = [res_dir, algo, '/laptop_', algo, '_', ...
            sessions{i}, '/stamped_traj_estimate.txt'];
    else
        for j = runs
            est_files{j+1} = [res_dir, algo, '/laptop_', algo, '_', ...
                sessions{i}, '/stamped_traj_estimate', num2str(j), '.txt'];
        end
    end
    est_files{end} = [res_dir, algo, '/laptop_', algo, '_', ...
        sessions{i}, '/stamped_groundtruth.txt'];
    styles = {'r', 'b', 'k', 'm', 'c'};
    legends = cell(length(est_files), 1);
    figure('units','normalized','outerposition',[0 0 1 1]);
    successfulTrials = 0;
    for j = 1:trials
        if ~isfile(est_files{j})
            continue;
        end
        if excludeResultOfHugeDrift
            data = readmatrix(est_files{j}, 'NumHeaderLines', 1);
            if max(max(abs(data(:, 2:4)))) > driftTolerance
                continue;
            end
        end
        drawColumnsInFile(est_files{j}, 2:4, 1, 1, styles(j));
        hold on;
        successfulTrials = successfulTrials + 1;
        legends{successfulTrials} = ['est ', num2str(j-1)];
    end
    if isfile(est_files{trials + 1})
        drawColumnsInFile(est_files{trials + 1}, 2:4, 1, 1, {'g'});
        successfulTrials = successfulTrials + 1;
        legends{successfulTrials} = 'gt';
    end
    legend(legends{1:successfulTrials});
    title([algo, '-', sessions{i}], 'Interpreter', 'none');
%     x = input("Enter anything except for '0' to continue:");
    x = 1;
    if x == 0
        disp('Exiting.');
        break;
    end
end
