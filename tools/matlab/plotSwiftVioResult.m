function plotSwiftVioResult(vio_file, options, gt_file, outputPath)
if nargin < 2
    disp('Usage:plotSwiftVioResult swift_vio.csv options [gt_file] [outputPath]');
    return;
end
if nargin < 3
    gt_file = '';
end
if nargin < 4
    [outputPath, ~, ~] = fileparts(vio_file);
    disp(['outputPath is set to ', outputPath]);
end

if options.export_fig_path
    addpath(options.export_fig_path);
end

addpath('plotters');

close all;

indexServer = SwiftVioConstants(options.misalignment_dim, ...
    options.extrinsic_dim, options.project_intrinsic_dim, ...
    options.distort_intrinsic_dim, options.fix_extrinsic, ...
    options.fix_intrinsic, options.td_dim, options.tr_dim);

data = readmatrix(vio_file, 'NumHeaderLines', 1);
original_data = data;

sec_to_nanos = 1e9;
startTime = data(1, 1);
endTime = data(end, 1);

gt_index.r = 3:5;
gt_index.q = 6:9;
gt_index.v = 10:12;
gt_index.b_g = 13:15;
gt_index.b_a = 16:18;
applyUmeyama = 0;

if gt_file
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
    if size(gt, 2) >= gt_index.b_g(3)
        data_diff(:, indexServer.b_g) = ...
            data_diff(:, indexServer.b_g) - gt(assocIndex, gt_index.b_g);
    end
    if size(gt, 2) >= gt_index.b_a(3)
        data_diff(:, indexServer.b_a) = ...
            data_diff(:, indexServer.b_a) - gt(assocIndex, gt_index.b_a);
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
outputfig = [outputPath, '/p_GB.eps'];
if exist(outputfig, 'file')==2
    delete(outputfig);
end
export_fig(outputfig);

if(~isempty(gt))
    % compute totla distance, max error, rmse
    distance = totalDistance(gt(assocIndex, gt_index.r));
    errorPosition = data(:,indexServer.r)- gt(assocIndex, gt_index.r);
    absError = sqrt(sum(errorPosition.^2,2));
    [maxError, idx]= max(absError);
    rmse = sqrt(sum(sum(errorPosition.^2,2))/size(data,1));
    fprintf('max error over distance %f, rmse over distance %f.\n', ...
        maxError/distance, rmse/distance);
    
    figure;
    drawMeanAndStdBound(data_diff, indexServer.r, indexServer.r_std, 1, 1);
    ylabel('$\delta \mathbf{t}_{WB}$ (m)', 'Interpreter', 'Latex');
    saveas(gcf,[outputPath, '\Error p_WB'],'epsc');
    
    figure;
    drawMeanAndStdBound(data_diff, indexServer.q(1:3), ...
        indexServer.q_std, 180/pi, 1);
    ylabel('$\delta \mathbf{\theta}_{WB}{} (^{\circ})$', 'Interpreter', 'Latex');
    saveas(gcf,[outputPath, '\Error R_WB'],'epsc');
    
    figure;
    if size(gt, 2) >= 2 + 7 + 3
        drawMeanAndStdBound(data_diff, indexServer.v, ...
            indexServer.v_std, 1, 1);
        ylabel('$\delta \mathbf{v}_{WB} (m/s)$', 'Interpreter', 'Latex');
        saveas(gcf,[outputPath, '\Error v_WB'],'epsc');
    end
    figure;
    if size(gt, 2) >= 2 + 7 + 3 + 3
        drawMeanAndStdBound(data_diff, indexServer.b_g, ...
            indexServer.b_g_std, 180/pi, 1);
        ylabel('$\delta \mathbf{b}_{g} (^\circ/s)$', 'Interpreter', 'Latex');
        saveas(gcf,[outputPath, '\Error b_g'],'epsc');
    end
    figure;
    if size(gt, 2) >= 2 + 7 + 3 + 6
        drawMeanAndStdBound(data_diff, indexServer.b_a, ...
            indexServer.b_a_std, 1, 1);
        ylabel('$\delta \mathbf{b}_{a} (m/s)$', 'Interpreter', 'Latex');
        saveas(gcf,[outputPath, '\Error b_a'],'epsc');
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
outputfig = [outputPath, '/qxyz_GB.eps'];
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
    legend('vx', 'vy', 'vz', 'std x', 'std y', 'std z', '-std x', ...
        '-std y', '-std z', 'gt vx', 'gt vy', 'gt vz');
end
ylabel('v_{GB}[m/s]');
outputfig = [outputPath, '/v_GB.eps'];
if exist(outputfig, 'file')==2
    delete(outputfig);
end
export_fig(outputfig);

plotImuBiases(data, indexServer, outputPath);

if ~isempty(indexServer.p_BC_std)
    figure;
    drawMeanAndStdBound(data, indexServer.p_BC, indexServer.p_BC_std, 100.0);
    ylabel('p_{BC}[cm]');
    outputfig = [outputPath, '/p_BC.eps'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
end

plotCameraIntrinsics(data, indexServer, options.trueCameraIntrinsics, outputPath);

plotTemporalParameters(data, indexServer, outputPath);

plotImuIntrinsicParameters(data, indexServer, outputPath);

averageSwiftVioVariableEstimates(original_data, indexServer, ...
    [outputPath, '/avg_estimates.txt'], options.avg_since_start, ...
    options.avg_trim_end);

end

function plotImuBiases(data, indexServer, outputPath)
figure;
drawMeanAndStdBound(data, indexServer.b_g, indexServer.b_g_std, 180/pi);
ylabel(['b_g[' char(176) '/s]']);
outputfig = [outputPath, '/b_g.eps'];
if exist(outputfig, 'file')==2
    delete(outputfig);
end
export_fig(outputfig);

figure;
drawMeanAndStdBound(data, indexServer.b_a, indexServer.b_a_std, 1.0);
ylabel('b_a[m/s^2]');
outputfig = [outputPath, '/b_a.eps'];
if exist(outputfig, 'file')==2
    delete(outputfig);
end
export_fig(outputfig);
end

function plotCameraIntrinsics(data, indexServer, trueIntrinsics, outputPath)
if ~isempty(indexServer.fxy_cxy_std) && indexServer.fxy_cxy_std(end) < size(data, 2)
    figure;
    data(:, indexServer.fxy_cxy) = data(:, indexServer.fxy_cxy) - ...
        repmat(trueIntrinsics, size(data, 1), 1);
    drawMeanAndStdBound(data, indexServer.fxy_cxy, ...
        indexServer.fxy_cxy_std);
    legend('f_x','f_y','c_x','c_y','3\sigma_f_x','3\sigma_f_y',...
        '3\sigma_c_x','3\sigma_c_y');
    ylabel(['deviation from nominal values ($f_x$, $f_y$), ', ...
        '($c_x$, $c_y$)[px]'], 'Interpreter', 'Latex');
    outputfig = [outputPath, '/fxy_cxy.eps'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
end

if ~isempty(indexServer.k1_k2_std) && indexServer.k1_k2_std(end) < size(data, 2)
    figure;
    drawMeanAndStdBound(data, indexServer.k1_k2, indexServer.k1_k2_std);
    ylabel('k_1 and k_2[1]');
    legend('k_1','k_2', '3\sigma_{k_1}', '3\sigma_{k_2}');
    outputfig = [outputPath, '/k1_k2.eps'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
end

if ~isempty(indexServer.p1_p2_std) && indexServer.p1_p2_std(end) < size(data, 2)
    figure;
    p1p2Scale = 1e2;
    drawMeanAndStdBound(data, indexServer.p1_p2, indexServer.p1_p2_std, p1p2Scale);
    ylabel('p_1 and p_2[0.01]');
    legend('p_1','p_2', '3\sigma_{p_1}', '3\sigma_{p_2}');
    outputfig = [outputPath, '/p1_p2.eps'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
end
end

function plotImuIntrinsicParameters(data, indexServer, outputPath)
% It will modify data by subtracting reference values from data
if ~isempty(indexServer.T_g_std) && indexServer.T_g_std(end) < size(data, 2)
    figure;
    data(:, indexServer.T_g_diag) = data(:, indexServer.T_g_diag) ...
        - ones(size(data, 1), 3);
    drawMeanAndStdBound(data, indexServer.T_g, ...
        indexServer.T_g_std);
    ylabel('$\mathbf{T}_g$[1]' , 'Interpreter', 'Latex');
    outputfig = [outputPath, '/T_g.eps'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
    
    figure;
    drawMeanAndStdBound(data, indexServer.T_s, ...
        indexServer.T_s_std);
    ylabel('$\mathbf{T}_s$[1]' , 'Interpreter', 'Latex');
    outputfig = [outputPath, '/T_s.eps'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
    
    figure;
    data(:, indexServer.T_a_diag) = data(:, indexServer.T_a_diag) ...
        - ones(size(data, 1), 3);
    drawMeanAndStdBound(data, indexServer.T_a, ...
        indexServer.T_a_std);
    ylabel('$\mathbf{T}_a$[1]' , 'Interpreter', 'Latex');
    outputfig = [outputPath, '/T_a.eps'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
end
end

function plotTemporalParameters(data, indexServer, outputPath)
if ~isempty(indexServer.td_std) && indexServer.td_std < size(data, 2)
    figure;
    drawMeanAndStdBound(data, indexServer.td, ...
        indexServer.td_std, 1000);
    legend('t_d', '3\sigma_t_d');
    ylabel('$t_d$[ms]', 'Interpreter', 'Latex');
    outputfig = [outputPath, '/t_d.eps'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
end

if ~isempty(indexServer.tr_std) && indexServer.tr_std < size(data, 2)
    figure;
    drawMeanAndStdBound(data, indexServer.tr, ...
        indexServer.tr_std, 1000);
    legend('t_r', '3\sigma_{t_r}');
    ylabel('$t_r$[ms]', 'Interpreter', 'Latex');
    outputfig = [outputPath, '/t_r.eps'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
end
end
