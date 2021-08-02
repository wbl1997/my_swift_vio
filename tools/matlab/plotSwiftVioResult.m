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

data = readmatrix(vio_file, 'NumHeaderLines', 1);
original_data = data;

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
    data(:,1) = data(:,1) - startTime;
    
    if applyUmeyama == 1
        % umeyama transform to gt
        src = data(100:end, options.r);
        dst = gt(assocIndex(100:end), gt_index.r);
        [R_res, t_res] = umeyama(src',dst');
        data(:,options.r) = (R_res*data(:, options.r)'+ ...
            repmat(t_res, 1, size(data,1)))';
        
        q_res= quaternion(R_res, 'rotmat', 'point');
        for i=1:size(data,1)
            res4 = compact(q_res * quaternion(...
                [data(i,options.q(4)), ...
                data(i, options.q(1:3))]));
            data(i,options.q(1:3)) = res4(2:4);
            data(i,options.q(4)) = res4(1);
        end
        data(:,options.v) = (R_res*data(:, options.v)')';
    end
    
    data_diff = data;
    data_diff(:, options.r) = ...
        data_diff(:, options.r) - gt(assocIndex, gt_index.r);
    alpha= zeros(size(data,1),3);
    for i=1:size(data,1)
        qs2w= gt(assocIndex(i), [gt_index.q(4), gt_index.q(1:3)]);
        qs2w_hat = data(i, [options.q(4), options.q(1:3)]);
        alpha(i,:)= unskew(rotmat(quaternion(qs2w), 'point') * ...
            rotmat(quaternion(qs2w_hat), 'point')'-eye(3))';
    end
    data_diff(:, options.q(1:3)) = alpha;
    if size(gt, 2) >= gt_index.v(3)
        data_diff(:, options.v) = ...
            data_diff(:, options.v) - gt(assocIndex, gt_index.v);
    end
else
    gt = [];
    data(:,1) = data(:,1) - startTime;
    data_diff = [];
end

figure;
plot3(data(:, options.r(1)), data(:, options.r(2)), ...
    data(:, options.r(3)), '-b'); hold on;
plot3(data(1, options.r(1)), data(1, options.r(2)), ...
    data(1, options.r(3)), '-or');
plot3(data(end, options.r(1)), data(end, options.r(2)), ...
    data(end, options.r(3)), '-sr');
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
outputfig = [outputPath, '/p_GB.pdf'];
if exist(outputfig, 'file')==2
    delete(outputfig);
end
export_fig(outputfig);

if(~isempty(gt))
    % compute total distance, max error, rmse
    distance = totalDistance(gt(assocIndex, gt_index.r));
    errorPosition = data(:,options.r)- gt(assocIndex, gt_index.r);
    absError = sqrt(sum(errorPosition.^2,2));
    [maxError, idx]= max(absError);
    rmse = sqrt(sum(sum(errorPosition.^2,2))/size(data,1));
    fprintf('max error over distance %f, rmse over distance %f.\n', ...
        maxError/distance, rmse/distance);
    
    figure;
    drawMeanAndStdBound(data_diff, options.r, options.r_std, 1, 1);
    xlim(options.xlim);
    ylabel('$\delta \mathbf{p}_{WB}$ (m)', 'Interpreter', 'Latex');
    saveas(gcf,[outputPath, '\Error p_WB.pdf']);
    
    figure;
    drawMeanAndStdBound(data_diff, options.q(1:3), ...
        options.q_std, 180/pi, 1);
    xlim(options.xlim);
    ylabel('$\delta \mathbf{\theta}_{WB}{} (^{\circ})$', 'Interpreter', 'Latex');
    saveas(gcf,[outputPath, '\Error R_WB.pdf']);
    
    figure;
    if size(gt, 2) >= 2 + 7 + 3
        drawMeanAndStdBound(data_diff, options.v, ...
            options.v_std, 1, 1);
        xlim(options.xlim);
        ylabel('$\delta \mathbf{v}_{WB} (m/s)$', 'Interpreter', 'Latex');
        saveas(gcf,[outputPath, '\Error v_WB.pdf']);
    end
end

figure;
plot(data(:,1), data(:, options.q(1)), '-r');
hold on;
plot(data(:,1), data(:, options.q(2)), '-g');
plot(data(:,1), data(:, options.q(3)), '-b');

if(~isempty(gt))
    plot(gt(:,1), gt(:,gt_index.q(1)), '--r');
    plot(gt(:,1), gt(:,gt_index.q(2)), '--g');
    plot(gt(:,1), gt(:,gt_index.q(3)), '--b');
    legend('qx', 'qy', 'qz', 'gt qx', 'gt qy', 'gt qz');
else
    legend('qx', 'qy', 'qz');
end
xlim(options.xlim);
xlabel('time[sec]');
ylabel('q xyz[1]');
grid on;
outputfig = [outputPath, '/qxyz_GB.pdf'];
if exist(outputfig, 'file')==2
    delete(outputfig);
end
export_fig(outputfig);

if options.v_std(1) > size(data, 2)
    figure;
    plot(data(:,1), data(:, options.v(1)), '-r');
    hold on;
    plot(data(:,1), data(:, options.v(2)), '-g');
    plot(data(:,1), data(:, options.v(3)), '-b');
    ylabel('v_{GB}[m/s]');
else
    figure;
    drawMeanAndStdBound(data, options.v, options.v_std);
    if(~isempty(gt))
        plot(gt(:,1), gt(:,gt_index.v(1)), '-.r');
        plot(gt(:,1), gt(:,gt_index.v(2)), '-.g');
        plot(gt(:,1), gt(:,gt_index.v(3)), '-.b');
        legend('vx', 'vy', 'vz', 'std x', 'std y', 'std z', '-std x', ...
            '-std y', '-std z', 'gt vx', 'gt vy', 'gt vz');
    end
    xlim(options.xlim);
    ylabel('v_{GB}[m/s]');
    
    outputfig = [outputPath, '/v_GB.pdf'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
end

plotImuBiases(data, options, outputPath);

if ~isempty(options.p_camera)
    figure;
    drawMeanAndStdBound(data, options.p_camera, options.p_camera_std, 100.0);
    xlim(options.xlim);
    ylabel('p_{camera}[cm]');
    outputfig = [outputPath, '/p_camera.pdf'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
end

plotCameraIntrinsics(data, options, outputPath);

plotTemporalParameters(data, options, outputPath);

plotImuIntrinsicParameters(data, options, outputPath);

averageSwiftVioVariableEstimates(original_data, options, ...
    [outputPath, '/avg_estimates.txt'], options.avg_since_start, ...
    options.avg_trim_end);

end

function plotImuBiases(data, options, outputPath)
if options.b_g_std(end) > size(data, 2)
    figure;
    plot(data(:,1), data(:, options.b_g(1)) * 180 / pi, '-r');
    hold on;
    plot(data(:,1), data(:, options.b_g(2)) * 180 / pi, '-g');
    plot(data(:,1), data(:, options.b_g(3)) * 180 / pi, '-b');
    xlim(options.xlim);
    ylabel('b_g [deg/s]');
    
    figure;
    plot(data(:,1), data(:, options.b_a(1)), '-r');
    hold on;
    plot(data(:,1), data(:, options.b_a(2)), '-g');
    plot(data(:,1), data(:, options.b_a(3)), '-b');
    xlim(options.xlim);
    ylabel('b_a[m/s^2]');
    
else
    figure;
    drawMeanAndStdBound(data, options.b_g, options.b_g_std, 180/pi);
    xlim(options.xlim);
    ylabel(['b_g[' char(176) '/s]']);
    outputfig = [outputPath, '/b_g.pdf'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
    
    figure;
    drawMeanAndStdBound(data, options.b_a, options.b_a_std, 1.0);
    xlim(options.xlim);
    ylabel('b_a[m/s^2]');
    outputfig = [outputPath, '/b_a.pdf'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
end
end

function plotCameraIntrinsics(data, options, outputPath)
if ~isempty(options.fxy_cxy) && options.fxy_cxy_std(end) < size(data, 2)
    figure;
    data(:, options.fxy_cxy) = data(:, options.fxy_cxy) - ...
        repmat(options.trueCamProjectionIntrinsics, size(data, 1), 1);
    drawMeanAndStdBound(data, options.fxy_cxy, ...
        options.fxy_cxy_std);
    xlim(options.camIntrinsicXlim);
    legend('f_x','f_y','c_x','c_y','3\sigma_f_x','3\sigma_f_y',...
        '3\sigma_c_x','3\sigma_c_y');
    ylabel(['deviation from nominal values ($f_x$, $f_y$), ', ...
        '($c_x$, $c_y$)[px]'], 'Interpreter', 'Latex');
    outputfig = [outputPath, '/fxy_cxy.pdf'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
end

if ~isempty(options.k1_k2) && options.k1_k2_std(end) < size(data, 2)
    figure;
    drawMeanAndStdBound(data, options.k1_k2, options.k1_k2_std);
    xlim(options.camIntrinsicXlim);
    ylabel('k_1 and k_2[1]');
    legend('k_1','k_2', '3\sigma_{k_1}', '3\sigma_{k_2}');
    outputfig = [outputPath, '/k1_k2.pdf'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
end

if ~isempty(options.p1_p2) && options.p1_p2_std(end) < size(data, 2)
    figure;
    p1p2Scale = 1e2;
    drawMeanAndStdBound(data, options.p1_p2, options.p1_p2_std, p1p2Scale);
    xlim(options.camIntrinsicXlim);
    ylabel('p_1 and p_2[0.01]');
    legend('p_1','p_2', '3\sigma_{p_1}', '3\sigma_{p_2}');
    outputfig = [outputPath, '/p1_p2.pdf'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
end
end

function plotImuIntrinsicParameters(data, options, outputPath)
% It will modify data by subtracting reference values from data
if ~isempty(options.T_g) && options.T_g_std(end) < size(data, 2)
    figure;
    data(:, options.T_g_diag) = data(:, options.T_g_diag) ...
        - ones(size(data, 1), 3);
    drawMeanAndStdBound(data, options.T_g, ...
        options.T_g_std);
    xlim(options.xlim);
    ylabel('$\mathbf{T}_g$[1]' , 'Interpreter', 'Latex');
    outputfig = [outputPath, '/T_g.pdf'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
    
    figure;
    drawMeanAndStdBound(data, options.T_s, ...
        options.T_s_std);
    xlim(options.xlim);
    ylabel('$\mathbf{T}_s$[1]' , 'Interpreter', 'Latex');
    outputfig = [outputPath, '/T_s.pdf'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
    
    figure;
    data(:, options.T_a_diag) = data(:, options.T_a_diag) ...
        - ones(size(data, 1), 3);
    drawMeanAndStdBound(data, options.T_a, ...
        options.T_a_std);
    xlim(options.xlim);
    ylabel('$\mathbf{T}_a$[1]' , 'Interpreter', 'Latex');
    outputfig = [outputPath, '/T_a.pdf'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
end
end

function plotTemporalParameters(data, options, outputPath)
if ~isempty(options.td) && options.td_std < size(data, 2)
    figure;
    drawMeanAndStdBound(data, options.td, ...
        options.td_std, 1000);
    xlim(options.xlim);
    legend('t_d', '3\sigma_t_d');
    ylabel('$t_d$[ms]', 'Interpreter', 'Latex');
    outputfig = [outputPath, '/t_d.pdf'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
end

if ~isempty(options.tr) && options.tr_std <= size(data, 2)
    figure;
    drawMeanAndStdBound(data, options.tr, ...
        options.tr_std, 1000);
    xlim(options.xlim);
    legend('t_r', '3\sigma_{t_r}');
    ylabel('$t_r$[ms]', 'Interpreter', 'Latex');
    outputfig = [outputPath, '/t_r.pdf'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
end
end
