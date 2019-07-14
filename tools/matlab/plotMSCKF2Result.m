function plotMSCKF2Result()
close all;
export_fig_path = '/media/jhuai/Seagate/jhuai/export_fig/';
addpath(export_fig_path);

filename = input('msckf2_csv:', 's');
output_dir = input('output_dir:', 's');
nominal_fx = input('nominal_fx:');
nominal_fy = input('nominal_fy:');
nominal_cx = input('nominal_cx:');
nominal_cy = input('nominal_cy:');
nominal_intrinsics = [nominal_fx, nominal_fy, nominal_cx, nominal_cy];

fontsize = 18;
data = dlmread(filename, ',', 1, 0);
original_data = data;
startTime = data(1, 1);
endTime = data(end, 1);

sec_to_nanos = 1e9;
% ground truth file must have the same number of rows as output data
gt_file = input('Ground truth csv:', 's');
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
    data(:,1) = (data(:,1) - startTime) / sec_to_nanos;
end

figure;
plot3(data(:, Msckf2Constants.r(1)), data(:, Msckf2Constants.r(2)), ...
    data(:, Msckf2Constants.r(3)), '-b'); hold on;
plot3(data(1, Msckf2Constants.r(1)), data(1, Msckf2Constants.r(2)), ...
    data(1, Msckf2Constants.r(3)), '-or');
plot3(data(end, Msckf2Constants.r(1)), data(end, Msckf2Constants.r(2)), ...
    data(end, Msckf2Constants.r(3)), '-sr');
legend_list = {'msckf2', 'start', 'finish'};

if(~isempty(gt))
    plot3(gt(:,2), gt(:,3), gt(:,4), '-r');
    legend_list{end+1} = 'gt';
end

% eg., 'Seagate/temp/parkinglot/opt_states.txt';
cmp_data_file = input('okvis_classic:', 's'); 
if (cmp_data_file)
    cmp_data = dlmread(cmp_data_file, ' ', 3, 0);
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
plot(data(:,1), data(:, Msckf2Constants.q(1)), '-r');
hold on;
plot(data(:,1), data(:, Msckf2Constants.q(2)), '-g');
plot(data(:,1), data(:, Msckf2Constants.q(3)), '-b');

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
draw_ekf_triplet_with_std(data, Msckf2Constants.v, Msckf2Constants.v_std);
ylabel('v_{GB}[m/s]');
outputfig = [output_dir, '/v_GB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
draw_ekf_triplet_with_std(data, Msckf2Constants.b_g, Msckf2Constants.b_g_std, 180/pi);
ylabel(['b_g[' char(176) '/s]']);
outputfig = [output_dir, '/b_g.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
draw_ekf_triplet_with_std(data, Msckf2Constants.b_a, Msckf2Constants.b_a_std, 1.0);
ylabel('b_a[m/s^2]');
outputfig = [output_dir, '/b_a.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
draw_ekf_triplet_with_std(data, Msckf2Constants.p_BC, Msckf2Constants.p_BC_std, 100.0);
ylabel('p_{BC}[cm]');
outputfig = [output_dir, '/p_BC.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
data(:, Msckf2Constants.fxy_cxy) = data(:, Msckf2Constants.fxy_cxy) - ...
    repmat(nominal_intrinsics, size(data, 1), 1);
draw_ekf_triplet_with_std(data, Msckf2Constants.fxy_cxy, ...
    Msckf2Constants.fxy_cxy_std);
legend('f_x','f_y','c_x','c_y','3\sigma_f_x', '3\sigma_f_y','3\sigma_c_x','3\sigma_c_y');
ylabel('deviation from nominal values ($f_x$, $f_y$), ($c_x$, $c_y$)[px]', 'Interpreter', 'Latex');
outputfig = [output_dir, '/fxy_cxy.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
draw_ekf_triplet_with_std(data, Msckf2Constants.k1_k2, Msckf2Constants.k1_k2_std);
ylabel('k_1 and k_2[1]');
legend('k_1','k_2', '3\sigma_{k_1}', '3\sigma_{k_2}');
outputfig = [output_dir, '/k1_k2.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
p1p2Scale = 1e2;
draw_ekf_triplet_with_std(data, Msckf2Constants.p1_p2, Msckf2Constants.p1_p2_std, p1p2Scale);
ylabel('p_1 and p_2[0.01]');
legend('p_1','p_2', '3\sigma_{p_1}', '3\sigma_{p_2}');
outputfig = [output_dir, '/p1_p2.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

intermediatePlotter(data, output_dir);

averageMsckf2VariableEstimates(original_data);
end

