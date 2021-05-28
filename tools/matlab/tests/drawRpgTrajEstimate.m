close all;
% draw vio results from ref_dir, and result_dir for one data session.
% result_dir may contain VIO results of multiple trials for the session. 
ref_dir = ['/keyframe_based_filter_2020/uzh_fpv_competition/results/', ...
    'submission_4_26_2020/laptop_MSCKF_nframe_'];
res_dir = ['/keyframe_based_filter_2020/uzh_fpv_competition/results/', ...
    'submission_5_11_2020/'];
algo = 'MSCKF_nframe';
sessions = {'in_45_3',  'in_45_16', 'in_fwd_11', ...
    'in_fwd_12', 'out_fwd_9', 'out_fwd_10'};

trials = 1;
for i = 1:length(sessions)
    drawRpgTrajectoryEvaluationResults(res_dir, algo, sessions(i), trials, 1000);
    drawColumnsInFile(...
        [ref_dir, sessions{i}, '/stamped_traj_estimate.txt'], ...
        2:4, 1, 1, {'--c'});
end
