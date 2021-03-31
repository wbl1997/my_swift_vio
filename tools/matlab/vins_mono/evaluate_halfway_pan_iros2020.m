function evaluate_halfway_pan_iros2020(export_fig_path)
% evaluate estimators' results for data sessions of sideways motion with halfway
% panning, only panning, or only standstill.
addpath(export_fig_path);
% root_dit is the folder of estimator results for a batch of data.
result_dir='/PAVIO_cvpr2020/results/post_processed';
root_dir_list = {
    [result_dir, '/hailun_house/halfway_pan_init'], ...
    [result_dir, '/hailun_house/halfway_pan'], ...
    [result_dir, '/hailun_house/halfway_pan_af_off'], ...
    [result_dir, '/hailun_house/pan'], ...
    [result_dir, '/hailun_house/pan_af_off'], ...
    [result_dir, '/hailun_house/stationary']};

data_dir='/media/jhuai/OldWin8OS/jhuai/data';
ref_dist = 5;

for k = 1:length(root_dir_list)
    root_dir = root_dir_list{k};
    msckf_results = dir([root_dir, '/**/msckf_estimates.csv']);
    vins_results = dir([root_dir, '/**/vins_result_*.csv']);
    est_results = [msckf_results; vins_results];
    
    folderlist = {est_results.folder}';
    namelist = {est_results.name}';
    durationlist = zeros(size(folderlist));
    
    % find the corresponding data session duration
    for i = 1:length(folderlist)
        [folderpath, ~, ~] = fileparts(folderlist{i});
        data_session_folder = strrep(folderpath, result_dir, data_dir);
        frame_time_file = [data_session_folder, '/frame_timestamps.txt'];
        frame_times = readmatrix(frame_time_file, 'NumHeaderLines', 1);
        durationlist(i) = (frame_times(end) - frame_times(1)) * 0.000000001;
        assert(durationlist(i) > 25);
        assert(durationlist(i) < 600);
    end
    
    % compute the metrics
    output_file = [root_dir, '/full_metric_values.csv'];
    fileID = fopen(output_file,'w');
    fprintf(fileID, ['%%folder,estimator_file,data_uration,delta_o,range,delta_s,'...
        'delta, The last two are irrelevant for pan and standstill data.\n']);
    for i = 1:length(folderlist)
        if contains(namelist{i}, 'vins_result_ex.csv')
            continue
        end
        if contains(folderlist{i}, 'halfway')
            visualize_result = 1;
        else
            visualize_result = 0;
        end
        if contains(namelist{i}, 'vins_result')
            position_indices = 2:4;
        else
            assert(contains(namelist{i}, 'msckf_estimates.csv'));
            position_indices = 3:5;
        end
        est_file = fullfile(folderlist{i}, namelist{i});
        fprintf('working on %s\n', est_file);
        [delta_o, range, delta_s, delta] = metrics_line_fitting(...
            est_file, position_indices, durationlist(i), ref_dist, ...
            visualize_result);
        fprintf(fileID,'%s,%s,%6.8f,%6.5f,%6.5f,%6.5f,%6.5f\n', folderlist{i}, ...
            namelist{i}, durationlist(i), delta_o, range, delta_s, delta);
    end
    fclose(fileID);
    fprintf('The metric values are saved at %s\n', output_file);
end
end
