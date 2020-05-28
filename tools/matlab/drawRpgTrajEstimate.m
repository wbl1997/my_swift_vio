close all;
% draw the results structured according to rpg traj evaluation tool.
ref_dir = ['/uzh_fpv_competition/results/submission', ...
    '_5_11_2020/laptop_MSCKF_nframe_'];
result_dir = ['/uzh_fpv_jac/vio/laptop/MSCKF_n_part_fix/laptop_MSCKF_n_part_fix_'];
session_names = {'in_45_3',  'in_45_16', 'in_fwd_11', ...
    'in_fwd_12', 'out_fwd_9', 'out_fwd_10'};

trials = 5;
for i = 1:size(session_names, 2)
    drawVioResultsForSession(ref_dir, result_dir, session_names{i}, trials);
end

function drawVioResultsForSession(ref_dir, result_dir, session, trials)
% draw vio results from ref_dir, and result_dir for one data session.
% result_dir may contain VIO results of multiple trials for the session. 
figure;
draw_columns_from_file([ref_dir, session, '/stamped_traj_estimate.txt'], 2:4, 1);
hold on;
linestyles = {'k-', 'g-', 'b-', 'y-', 'm-'};
datalabels= cell(1, trials + 1);
datalabels{1} = 'ref';
validresults = 1;
for index = 1:trials
    if trials == 1
        result_file = [result_dir, session, '/stamped_traj_estimate.txt'];
    else
        result_file = [result_dir, session, '/stamped_traj_estimate', num2str(index-1), '.txt'];
    end
    if isfile(result_file)
        draw_columns_from_file(result_file, 2:4, 1, {linestyles{index}});
        validresults = validresults + 1;
        datalabels{validresults} = num2str(index-1);        
    end
end
datalabels = datalabels(1:validresults);
legend(datalabels);
title(session, 'Interpreter', 'none');
axis equal;
grid on;
end