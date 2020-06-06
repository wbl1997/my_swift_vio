function drawPqvAndStdBound(msckf_traj_list, gt_traj, traj_labels, ...
 misalignment_dim, extrinsic_dim, project_intrinsic_dim, ...
    distort_intrinsic_dim, fix_extrinsic, fix_intrinsic)
% result_dir = '/keyframe_based_filter_2020/results/simulation';
% results = {[result_dir, '/msckf_simul_wave/MSCKF_WavyCircle_0.txt'], ...
% [result_dir, '/msckf_simul_wave_nofej/MSCKF_WavyCircle_0.txt']};
% traj_labels = {'FEJ', 'Naive'};
% gt_traj = [result_dir, '/msckf_simul_wave_nofej/WavyCircle.txt'];
% drawPqvAndStdBound(results, gt_traj, traj_labels, 27, 3, 4, 4, 0, 0);

line_styles = {{'-r', '-g', '-b', '-k'},...
    {'-.m', '-.c', '-.k', '-.y'}, ...
    {'-.r', '-.g', '-.b', '-.k'}};
std_line_styles = {{'--r', '--g', '--b', '--k'},...
    {':m', ':c', ':k', ':y'},...
   {':r', ':g', ':b', ':k'}};

gt_index.r = 3:5;
gt_index.q = 6:9;
gt_index.v = 10:12;

msckf_index_server = Msckf2Constants(misalignment_dim, extrinsic_dim, ...
    project_intrinsic_dim, distort_intrinsic_dim, fix_extrinsic, fix_intrinsic);

sec_to_nanos = 1e9;

[output_dir, ~, ~] = fileparts(msckf_traj_list{1});
disp(['output_dir is set to ', output_dir]);

data = readmatrix(msckf_traj_list{1}, 'NumHeaderLines', 1);
data(:, 1) = data(:, 1) / sec_to_nanos;

startTime = data(1, 1);
endTime = data(end, 1);
endIndex = size(data, 1);
% determine start and end time
gt = readmatrix(gt_traj, 'NumHeaderLines', 1);
index= find(abs(gt(:,1) - startTime)<5e-2);
gt = gt(index:end,:);
if(endTime>gt(end,1))
    endTime = gt(end,1);
    endIndex= find(abs(endTime - data(:,1))<2.5e-2, 1);
else
    index= find(abs(gt(:,1) - endTime)<5e-2);
    gt= gt(1:index,:);
end
data= data(1:endIndex,:);

% associate the data by timestamps, discrepancy less than 0.02sec
starter =1;
assocIndex = zeros(size(data,1), 1);
for i=1:size(data,1)
    index = find(abs(gt(starter:end,1) - data(i,1))<2.5e-2);
    [~, uniqIndex] = min(abs(gt(starter+ index -1,1) - data(i,1)));
    assocIndex(i,1)= index(uniqIndex)+starter-1;
    starter = index(uniqIndex) + starter-1;
end
% association end
gt(:,1) = gt(:,1)- startTime;
data_diff_list = cell(1, length(msckf_traj_list));
for j = 1:length(msckf_traj_list)
    data = readmatrix(msckf_traj_list{j}, 'NumHeaderLines', 1);
    data(:, 1) = data(:, 1) / sec_to_nanos - startTime;
    data= data(1:endIndex,:);
    data_diff = data;
    data_diff(:, msckf_index_server.r) = ...
        data_diff(:, msckf_index_server.r) - gt(assocIndex, gt_index.r);
    alpha= zeros(size(data,1),3);
    for i=1:size(data,1)
        qs2w = gt(assocIndex(i), [gt_index.q(4), gt_index.q(1:3)]);
        qs2w_hat = data(i, [msckf_index_server.q(4), msckf_index_server.q(1:3)]);
        alpha(i,:) = unskew(rotqr2ro(qs2w')*rotqr2ro(qs2w_hat')'-eye(3))';
    end
    data_diff(:, msckf_index_server.q(1:3)) = alpha;
    if size(gt, 2) >= gt_index.v(3)
        data_diff(:, msckf_index_server.v) = ...
            data_diff(:, msckf_index_server.v) - gt(assocIndex, gt_index.v);
    end
    data_diff_list{j} = data_diff;
end

close all;
background_color='None';
time_range = [-10, 310];
figure;

num_results = length(msckf_traj_list);
select_line_handles = zeros(1, 6 * num_results);
legends_template = {'$x$', '$y$', '$z$', '$3\sigma_x$', '$3\sigma_y$', '$3\sigma_z$'};
legends = cell(1, 6 * num_results);

for j = 1:length(msckf_traj_list)
    line_handles = drawMeanAndStdBound(data_diff_list{j}, msckf_index_server.r, ...
        msckf_index_server.r_std, 1, 1, line_styles{j}, std_line_styles{j});
    select_line_handles((j-1) * 3 + (1:3)) = line_handles(1:3);
    select_line_handles(3 * num_results + (j-1) * 3 + (1:3)) = line_handles(4:6);
    for l = 1:3
        legends{(j-1) * 3 + l} = [traj_labels{j}, ' ', legends_template{l}];
        legends{3 * num_results + (j-1) * 3 + l} = [traj_labels{j}, ' ', legends_template{3 + l}];
    end
end
hLeg = legend(select_line_handles, legends);
set(hLeg,'visible','off');
xlim(time_range);
ylabel('$\delta \mathbf{t}_{WB}$ (m)', 'Interpreter', 'Latex');
set(gcf, 'Color', background_color);
outputfig = [output_dir, '/Error p_WB.eps'];
if exist(outputfig, 'file')==2
    delete(outputfig);
end
export_fig(outputfig);

figure;
select_line_handles = zeros(1, 6 * num_results);
for j = 1:length(msckf_traj_list)
    line_handles = drawMeanAndStdBound(data_diff_list{j}, msckf_index_server.q(1:3), ...
        msckf_index_server.q_std, 180/pi, 1, line_styles{j}, std_line_styles{j});
    select_line_handles((j-1) * 3 + (1:3)) = line_handles(1:3);
    select_line_handles(3 * num_results + (j-1) * 3 + (1:3)) = line_handles(4:6);
end
legend(select_line_handles, legends, 'Interpreter', 'Latex');
xlim([-10, 410]);
ylabel('$\delta \mathbf{\theta}_{WB}{} (^{\circ})$', 'Interpreter', 'Latex');
set(gcf, 'Color', background_color);
outputfig = [output_dir, '/Error q_WB.eps'];
if exist(outputfig, 'file')==2
    delete(outputfig);
end
export_fig(outputfig);

figure;
select_line_handles = zeros(1, 6 * num_results);
for j = 1:length(msckf_traj_list)
    line_handles = drawMeanAndStdBound(data_diff_list{j}, msckf_index_server.v, ...
        msckf_index_server.v_std, 1, 1, line_styles{j}, std_line_styles{j});
    select_line_handles((j-1) * 3 + (1:3)) = line_handles(1:3);
    select_line_handles(3 * num_results + (j-1) * 3 + (1:3)) = line_handles(4:6);
end
legend(select_line_handles, legends, 'Interpreter', 'Latex');
xlim(time_range);
ylabel('$\delta \mathbf{v}_{WB} (m/s)$', 'Interpreter', 'Latex');
set(gcf, 'Color', background_color);
outputfig = [output_dir, '/Error v_WB.eps'];
if exist(outputfig, 'file')==2
    delete(outputfig);
end
export_fig(outputfig);

end