function plotSwiftVioResult(vio_csv, export_fig_path, ...
    gt_file, output_dir, avg_since_start, avg_trim_end, ...
    misalignment_dim, extrinsic_dim, project_intrinsic_dim, ...
    distort_intrinsic_dim, fix_extrinsic, fix_intrinsic, td_dim, tr_dim)
addpath('cgraumann-umeyama');
if nargin < 1
    disp(['Usage:plotSwiftVioResult swift_vio.csv ...']);
    return;
end
if ~exist('fix_extrinsic', 'var')
    fix_extrinsic = 0;
end
if ~exist('fix_intrinsic', 'var')
    fix_intrinsic = 0;
end
if ~exist('td_dim', 'var')
    td_dim = 1;
end
if ~exist('tr_dim', 'var')
    tr_dim = 1;
end

close all;
if ~exist('export_fig_path','var')
    export_fig_path = input('path of export_fig:', 's');
end
if isempty(export_fig_path)
    export_fig_path = '/media/jhuai/Seagate/jhuai/tools/export_fig/';
end

addpath(export_fig_path);

if ~exist('output_dir','var')
    output_dir = input('output_dir, if empty, set to dir of the csv:', 's');
end
if isempty(output_dir)
    [output_dir, ~, ~] = fileparts(vio_csv);
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
indexServer = SwiftVioConstants(misalignment_dim, extrinsic_dim, ...
    project_intrinsic_dim, distort_intrinsic_dim, fix_extrinsic, ...
    fix_intrinsic, td_dim, tr_dim);

fontsize = 18;
data = readmatrix(vio_csv, 'NumHeaderLines', 1);
original_data = data;

sec_to_nanos = 1e9;
startTime = data(1, 1);
endTime = data(end, 1);

if ~exist('gt_file','var')
    % ground truth file must have the same number of rows as output data
    gt_file = input('Ground truth csv:', 's');
end

if (gt_file)
    gt_index.r = 3:5;
    gt_index.q = 6:9;
    gt_index.v = 10:12;
    applyUmeyama = 0;

    gt = readmatrix(gt_file, 'NumHeaderLines', 1);
    index= find(abs(gt(:,1) - startTime)<5e-2);
    gt = gt(index:end,:);
    if(endTime>gt(end,1))
        endTime = gt(end,1);
        index= find(abs(endTime - data(:,1))<2.5e-2, 1);
        data= data(1:index,:);
    else
        index= find(abs(gt(:,1) - endTime)<5e-2);
        gt= gt(1:index,:);
    end

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

    gt(:,1) = gt(:,1) - startTime;
    if data(1, 1) > sec_to_nanos
        data(:,1) = (data(:,1) - startTime) / sec_to_nanos;
    else
        data(:,1) = data(:,1) - startTime;
    end
    
    if applyUmeyama == 1
        % umeyama transform to gt
        src = data(100:end, indexServer.r);
        dst = gt(assocIndex(100:end), gt_index.r);
        [R_res, t_res] = umeyama(src',dst');
        data(:,indexServer.r) = (R_res*data(:, indexServer.r)'+ ...
            repmat(t_res, 1, size(data,1)))';

        q_res= quaternion(R_res, 'rotmat', 'point');    
        for i=1:size(data,1)
            res4 = compact(q_res * quaternion(...
                [data(i,indexServer.q(4)), ...
                data(i, indexServer.q(1:3))]));
            data(i,indexServer.q(1:3)) = res4(2:4);
            data(i,indexServer.q(4)) = res4(1);
        end
        data(:,indexServer.v) = (R_res*data(:, indexServer.v)')';
    end

    data_diff = data;
    data_diff(:, indexServer.r) = ...
        data_diff(:, indexServer.r) - gt(assocIndex, gt_index.r);
    alpha= zeros(size(data,1),3);
    for i=1:size(data,1)
        qs2w= gt(assocIndex(i), [gt_index.q(4), gt_index.q(1:3)]);
        qs2w_hat = data(i, [indexServer.q(4), indexServer.q(1:3)]);
        alpha(i,:)= unskew(rotmat(quaternion(qs2w), 'point') * ...
            rotmat(quaternion(qs2w_hat), 'point')'-eye(3))';
    end
    data_diff(:, indexServer.q(1:3)) = alpha;
    if size(gt, 2) >= gt_index.v(3)
        data_diff(:, indexServer.v) = ...
            data_diff(:, indexServer.v) - gt(assocIndex, gt_index.v);
    end
else
    gt = [];
    if data(1, 1) > sec_to_nanos
        data(:,1) = (data(:,1) - startTime) / sec_to_nanos;
    else
        data(:,1) = data(:,1) - startTime;
    end
    data_diff = [];
end

figure;
plot3(data(:, indexServer.r(1)), data(:, indexServer.r(2)), ...
    data(:, indexServer.r(3)), '-b'); hold on;
plot3(data(1, indexServer.r(1)), data(1, indexServer.r(2)), ...
    data(1, indexServer.r(3)), '-or');
plot3(data(end, indexServer.r(1)), data(end, indexServer.r(2)), ...
    data(end, indexServer.r(3)), '-sr');
legend_list = {'swift_vio', 'start', 'finish'};

if(~isempty(gt))
    plot3(gt(:, gt_index.r(1)), gt(:, gt_index.r(2)), gt(:, gt_index.r(3)), '-r');
    legend_list{end+1} = 'gt';
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
    distance = totalDistance(gt(assocIndex, gt_index.r));
    errorPosition = data(:,indexServer.r)- gt(assocIndex, gt_index.r);
    absError = sqrt(sum(errorPosition.^2,2));
    [maxError, idx]= max(absError)
    maxError/distance
    rmse = sqrt(sum(sum(errorPosition.^2,2))/size(data,1))
    rmse/distance

    figure;
    drawMeanAndStdBound(data_diff, indexServer.r, indexServer.r_std, 1, 1);
    ylabel('$\delta \mathbf{t}_{WB}$ (m)', 'Interpreter', 'Latex');
    saveas(gcf,[output_dir, '\Error p_WB'],'epsc');

    figure;
    drawMeanAndStdBound(data_diff, indexServer.q(1:3), ...
        indexServer.q_std, 180/pi, 1);
    ylabel('$\delta \mathbf{\theta}_{WB}{} (^{\circ})$', 'Interpreter', 'Latex');
    saveas(gcf,[output_dir, '\Error R_WB'],'epsc');

    figure;
    if size(gt, 2) >= 11
        drawMeanAndStdBound(data_diff, indexServer.v, ...
            indexServer.v_std, 1, 1);
        ylabel('$\delta \mathbf{v}_{WB} (m/s)$', 'Interpreter', 'Latex');
        saveas(gcf,[output_dir, '\Error v_WB'],'epsc');
    end
end

figure;
plot(data(:,1), data(:, indexServer.q(1)), '-r');
hold on;
plot(data(:,1), data(:, indexServer.q(2)), '-g');
plot(data(:,1), data(:, indexServer.q(3)), '-b');

if(~isempty(gt))
    plot(gt(:,1), gt(:,gt_index.q(1)), '--r');
    plot(gt(:,1), gt(:,gt_index.q(2)), '--g');
    plot(gt(:,1), gt(:,gt_index.q(3)), '--b');
    legend('qx', 'qy', 'qz', 'gt qx', 'gt qy', 'gt qz');
else
    legend('qx', 'qy', 'qz');
end
xlabel('time[sec]');
ylabel('q xyz[1]');
grid on;
outputfig = [output_dir, '/qxyz_GB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

if indexServer.v_std(1) > size(data, 2)
    return;
end

figure;
drawMeanAndStdBound(data, indexServer.v, indexServer.v_std);
if(~isempty(gt))
    plot(gt(:,1), gt(:,gt_index.v(1)), '-.r');
    plot(gt(:,1), gt(:,gt_index.v(2)), '-.g');
    plot(gt(:,1), gt(:,gt_index.v(3)), '-.b');
    legend('vx', 'vy', 'vz', 'std x', 'std y', 'std z', '-std x', '-std y', '-std z', 'gt vx', 'gt vy', 'gt vz');
end
ylabel('v_{GB}[m/s]');
outputfig = [output_dir, '/v_GB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
drawMeanAndStdBound(data, indexServer.b_g, indexServer.b_g_std, 180/pi);
ylabel(['b_g[' char(176) '/s]']);
outputfig = [output_dir, '/b_g.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
drawMeanAndStdBound(data, indexServer.b_a, indexServer.b_a_std, 1.0);

ylabel('b_a[m/s^2]');
outputfig = [output_dir, '/b_a.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

if ~isempty(indexServer.p_BC_std)
figure;

drawMeanAndStdBound(data, indexServer.p_BC, indexServer.p_BC_std, 100.0);

ylabel('p_{BC}[cm]');
outputfig = [output_dir, '/p_BC.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);
end

nominal_intrinsics = data(1, indexServer.fxy_cxy);
disp('The nominal fxy cxy is set to ');
disp(nominal_intrinsics);

if ~isempty(indexServer.fxy_cxy_std)
figure;
data(:, indexServer.fxy_cxy) = data(:, indexServer.fxy_cxy) - ...
    repmat(nominal_intrinsics, size(data, 1), 1);
drawMeanAndStdBound(data, indexServer.fxy_cxy, ...
    indexServer.fxy_cxy_std);
legend('f_x','f_y','c_x','c_y','3\sigma_f_x', '3\sigma_f_y','3\sigma_c_x','3\sigma_c_y');
ylabel('deviation from nominal values ($f_x$, $f_y$), ($c_x$, $c_y$)[px]', 'Interpreter', 'Latex');
outputfig = [output_dir, '/fxy_cxy.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);
end

if ~isempty(indexServer.k1_k2_std)
figure;
drawMeanAndStdBound(data, indexServer.k1_k2, indexServer.k1_k2_std);
ylabel('k_1 and k_2[1]');
legend('k_1','k_2', '3\sigma_{k_1}', '3\sigma_{k_2}');
outputfig = [output_dir, '/k1_k2.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);
end

if ~isempty(indexServer.p1_p2_std)
figure;
p1p2Scale = 1e2;
drawMeanAndStdBound(data, indexServer.p1_p2, indexServer.p1_p2_std, p1p2Scale);
ylabel('p_1 and p_2[0.01]');
legend('p_1','p_2', '3\sigma_{p_1}', '3\sigma_{p_2}');
outputfig = [output_dir, '/p1_p2.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);
end

intermediatePlotter(data, indexServer, output_dir);
if exist('avg_since_start','var')
    if exist('avg_trim_end','var')
        averageSwiftVioVariableEstimates(original_data, indexServer, ...
            [output_dir, '/avg_estimates.txt'], avg_since_start, avg_trim_end);
    else
        averageSwiftVioVariableEstimates(original_data, indexServer, ...
            [output_dir, '/avg_estimates.txt'], avg_since_start);
    end
else
    averageSwiftVioVariableEstimates(original_data, indexServer, ...
            [output_dir, '/avg_estimates.txt']);
end

end

