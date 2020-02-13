function plotMSCKF2Result(msckf_csv, export_fig_path, voicebox_path, output_dir, ...
    cmp_data_file, gt_file, avg_since_start, avg_trim_end, ...
    misalignment_dim, extrinsic_dim, project_intrinsic_dim, ...
    distort_intrinsic_dim, fix_extrinsic, fix_intrinsic)
if nargin < 1
    disp(['Usage:plotMSCKF2Result msckf_csv ...']);
    return;
end
if ~exist('fix_extrinsic', 'var')
    fix_extrinsic = 0;
end
if ~exist('fix_intrinsic', 'var')
    fix_intrinsic = 0;
end

close all;
if ~exist('export_fig_path','var')
    export_fig_path = input('path of export_fig:', 's');
end
if isempty(export_fig_path)
    export_fig_path = '/media/jhuai/Seagate/jhuai/tools/export_fig/';
end
if ~exist('voicebox_path','var')
    voicebox_path = input('path of voicebox:', 's');
end
if isempty(voicebox_path)
    voicebox_path = '/media/jhuai/Seagate/jhuai/tools/voicebox/';
end
addpath(export_fig_path);
addpath(voicebox_path);

filename = msckf_csv;
if isempty(filename)
    return
end
if ~exist('output_dir','var')
    output_dir = input('output_dir, if empty, set to dir of the csv:', 's');
end
if isempty(output_dir)
    [filepath, ~, ~] = fileparts(filename);
    output_dir = filepath;
    disp(['output_dir is set to ', output_dir]);
end

if ~exist('misalignment_dim','var')
    misalignment_dim_str = input('dim of IMU misalignment:', 's');
if isempty(misalignment_dim_str)
    misalignment_dim = 27;
else
    misalignment_dim = str2double(misalignment_dim_str);
end
end
if ~exist('extrinsic_dim','var')
    extrinsic_dim_str = input('dim of camera extrinsics:', 's');
if isempty(extrinsic_dim_str)
    extrinsic_dim = 3;
else
    extrinsic_dim = str2double(extrinsic_dim_str);
end
end

if ~exist('project_intrinsic_dim','var')
    project_intrinsic_dim_str = input('dim of camera projection intrinsics:', 's');

if isempty(project_intrinsic_dim_str)
    project_intrinsic_dim = 4;
else
    project_intrinsic_dim = str2double(project_intrinsic_dim_str);
end
end

if ~exist('distort_intrinsic_dim','var')
    distort_intrinsic_dim_str = input('dim of camera distortion intrinsics:', 's');
if isempty(distort_intrinsic_dim_str)
    distort_intrinsic_dim = 4;
else
    distort_intrinsic_dim = str2double(distort_intrinsic_dim_str);
end
end
msckf_index_server = Msckf2Constants(misalignment_dim, extrinsic_dim, ...
    project_intrinsic_dim, distort_intrinsic_dim, fix_extrinsic, fix_intrinsic);

fontsize = 18;
msckf_estimates = readmatrix(filename, 'NumHeaderLines', 1);
data = msckf_estimates;
original_data = data;
startTime = data(1, 1);
endTime = data(end, 1);

nominal_intrinsics = data(1, msckf_index_server.fxy_cxy);
disp('The nominal fxy cxy is set to ');
disp(nominal_intrinsics);

sec_to_nanos = 1e9;
if ~exist('cmp_data_file','var')
    % eg., 'Seagate/temp/parkinglot/opt_states.txt';
    cmp_data_file = input('okvis_classic:', 's'); 
end
if ~exist('gt_file','var')
    % ground truth file must have the same number of rows as output data
    gt_file = input('Ground truth csv:', 's');
end

if (gt_file)
    gt = csvread(gt_file);
    index= find(abs(gt(:,1) - startTime)<10000);
    gt = gt(index:end,:);
    if(endTime>gt(end,1))
        endTime = gt(end,1);
        index= find(abs(endTime - data(:,1))<2.5e7, 1);
        data= data(1:index,:);
    else
        index= find(abs(gt(:,1) - endTime)<10000);
        gt= gt(1:index,:);
    end
    
    % associate the data by timestamps, discrepancy less than 0.02sec
    starter =1;
    assocIndex = zeros(size(data,1), 1);
    for(i=1:size(data,1))
        index = find(abs(gt(starter:end,1) - data(i,1))<2.5e7);
        [~, uniqIndex] = min(abs(gt(starter+ index -1,1) - data(i,1)));
        assocIndex(i,1)= index(uniqIndex)+starter-1;
        starter = index(uniqIndex) + starter-1;
    end
    % association end
    
    gt(:,1) = (gt(:,1)- startTime)/1e9;
    data(:,1)= (data(:,1)- startTime)/1e9;
    % umeyama transform to gt
    src = data(100:end, 3:5);
    dst = gt(assocIndex(100:end), 2:4);
    [R_res, t_res] = umeyama(src',dst');
    data(:,3:5) = (R_res*data(:, 3:5)'+repmat(t_res, 1, size(data,1)))';
    
    alpha= zeros(size(data,1),3);
    for i=1:size(data,1)
        qs2w= gt(assocIndex(i), 5:8);
        qs2w_hat = data(i, [9,6:8]);
        alpha(i,:)= unskew(rotqr2ro(qs2w')*rotqr2ro(qs2w_hat')'-eye(3))';
    end
    
    q_res= rotro2qr(R_res);
    for i=1:size(data,1)
        res4= quatmult_v001(q_res, [data(i,9), data(i, 6:8)]', 0);
        data(i,6:8) = res4(2:4);
        data(i,9) = res4(1);
    end
    data(:,10:12) = (R_res*data(:, 10:12)')';
else
    gt = [];
    if data(1, 1) > sec_to_nanos
        data(:,1) = (data(:,1) - startTime) / sec_to_nanos;
    else
        data(:,1) = data(:,1) - startTime;
    end
end

figure;
plot3(data(:, msckf_index_server.r(1)), data(:, msckf_index_server.r(2)), ...
    data(:, msckf_index_server.r(3)), '-b'); hold on;
plot3(data(1, msckf_index_server.r(1)), data(1, msckf_index_server.r(2)), ...
    data(1, msckf_index_server.r(3)), '-or');
plot3(data(end, msckf_index_server.r(1)), data(end, msckf_index_server.r(2)), ...
    data(end, msckf_index_server.r(3)), '-sr');
legend_list = {'msckf2', 'start', 'finish'};

if(~isempty(gt))
    plot3(gt(:,2), gt(:,3), gt(:,4), '-r');
    legend_list{end+1} = 'gt';
end

if (cmp_data_file)
    cmp_data = readmatrix(cmp_data_file, 'NumHeaderLines', 1);
    plot3(cmp_data(:, 4), cmp_data(:, 5), cmp_data(:, 6), '-g');
    legend_list{end+1} = 'okvis';
end
legend(legend_list);
title('p_B^G');
xlabel('x[m]');
ylabel('y[m]');
zlabel('z[m]');
axis equal;
grid on;
outputfig = [output_dir, '/p_GB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

if(~isempty(gt))
    % compute totla distance, max error, rmse
    distance = totalDistance(gt(assocIndex, 2:4));
    errorPosition = data(:,3:5)- gt(assocIndex, 2:4);
    absError = sqrt(sum(errorPosition.^2,2));
    [maxError, idx]= max(absError)
    maxError/distance
    rmse = sqrt(sum(sum(errorPosition.^2,2))/size(data,1))
    rmse/distance
    
    % error plot
    figNumber = figNumber +1;
    figure(figNumber);
    plot(data(:,1), data(:,3)- gt(assocIndex, 2), '-r'); hold on;
    plot(data(:,1), data(:,4)- gt(assocIndex, 3), '-g');
    plot(data(:,1), data(:,5)- gt(assocIndex, 4), '-b');
    legend('x','y','z');
    title 'position error';
    xlabel 'time [sec]';
    ylabel '[m]';
    grid on;
    set(gca,'FontSize',fontsize);
    
    figNumber = figNumber +1;
    figure(figNumber);
    plot(data(:,1), alpha(:,1), '-r'); hold on;
    plot(data(:,1), alpha(:,2), '-g');
    plot(data(:,1), alpha(:,3), '-b');
    legend('x','y','z');
    title 'attitude error';
    xlabel 'time [sec]';
    ylabel 'rad/sec';
    grid on;
    set(gca,'FontSize',fontsize);
end

figure;
plot(data(:,1), data(:, msckf_index_server.q(1)), '-r');
hold on;
plot(data(:,1), data(:, msckf_index_server.q(2)), '-g');
plot(data(:,1), data(:, msckf_index_server.q(3)), '-b');

if(~isempty(gt))
    plot(gt(:,1), gt(:,6), '--r');
    plot(gt(:,1), gt(:,7), '--g');
    plot(gt(:,1), gt(:,8), '--b');
end
xlabel('time[sec]');
legend('qx', 'qy', 'qz', 'gt qx', 'gt qy', 'gt qz');
grid on;
outputfig = [output_dir, '/qxyz_GB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
draw_ekf_triplet_with_std(data, msckf_index_server.v, msckf_index_server.v_std);
ylabel('v_{GB}[m/s]');
outputfig = [output_dir, '/v_GB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
draw_ekf_triplet_with_std(data, msckf_index_server.b_g, msckf_index_server.b_g_std, 180/pi);
ylabel(['b_g[' char(176) '/s]']);
outputfig = [output_dir, '/b_g.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
draw_ekf_triplet_with_std(data, msckf_index_server.b_a, msckf_index_server.b_a_std, 1.0);
ylabel('b_a[m/s^2]');
outputfig = [output_dir, '/b_a.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

if ~isempty(msckf_index_server.p_BC_std)
figure;

draw_ekf_triplet_with_std(data, msckf_index_server.p_BC, msckf_index_server.p_BC_std, 100.0);

ylabel('p_{BC}[cm]');
outputfig = [output_dir, '/p_BC.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);
end

if ~isempty(msckf_index_server.fxy_cxy_std)
figure;
data(:, msckf_index_server.fxy_cxy) = data(:, msckf_index_server.fxy_cxy) - ...
    repmat(nominal_intrinsics, size(data, 1), 1);
draw_ekf_triplet_with_std(data, msckf_index_server.fxy_cxy, ...
    msckf_index_server.fxy_cxy_std);
legend('f_x','f_y','c_x','c_y','3\sigma_f_x', '3\sigma_f_y','3\sigma_c_x','3\sigma_c_y');
ylabel('deviation from nominal values ($f_x$, $f_y$), ($c_x$, $c_y$)[px]', 'Interpreter', 'Latex');
outputfig = [output_dir, '/fxy_cxy.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);
end

if ~isempty(msckf_index_server.k1_k2_std)
figure;
draw_ekf_triplet_with_std(data, msckf_index_server.k1_k2, msckf_index_server.k1_k2_std);
ylabel('k_1 and k_2[1]');
legend('k_1','k_2', '3\sigma_{k_1}', '3\sigma_{k_2}');
outputfig = [output_dir, '/k1_k2.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);
end

if ~isempty(msckf_index_server.p1_p2_std)
figure;
p1p2Scale = 1e2;
draw_ekf_triplet_with_std(data, msckf_index_server.p1_p2, msckf_index_server.p1_p2_std, p1p2Scale);
ylabel('p_1 and p_2[0.01]');
legend('p_1','p_2', '3\sigma_{p_1}', '3\sigma_{p_2}');
outputfig = [output_dir, '/p1_p2.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);
end

intermediatePlotter(data, msckf_index_server, output_dir);
if exist('avg_since_start','var')
    if exist('avg_trim_end','var')
        averageMsckf2VariableEstimates(original_data, msckf_index_server, ...
            [output_dir, '/avg_estimates.txt'], avg_since_start, avg_trim_end);
    else
        averageMsckf2VariableEstimates(original_data, msckf_index_server, ...
            [output_dir, '/avg_estimates.txt'], avg_since_start);
    end
else
    averageMsckf2VariableEstimates(original_data, msckf_index_server, ...
            [output_dir, '/avg_estimates.txt']);
end

end

