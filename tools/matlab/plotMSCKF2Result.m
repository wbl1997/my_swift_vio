%usage: matlab
close all;

addpath('cgraumann-umeyama')
    addpath('/media/jhuai/Seagate/jhuai/huai_work/ekfmonoslam/instk');
addpath('/media/jhuai/Seagate/jhuai/huai_work/ekfmonoslam/voicebox');

filename = input('msckf2_csv:', 's');
output_dir = input('output_dir:', 's');
% filename =
    'F:\jhuai\huai work\dissertation\figure_data\msckf2_estimator_output_noniter_float.csv';
% filename =
    'F:\west_campus_parking_lot\calibration\Greg\output\msckf2_estimator_output_greg_fixedTgTsTa.csv';

nominal_intrinsics =[1554.622, 1554.622, 960, 540]/2;
% nominal_intrinsics = [ 458.65, 457.30, 367.22, 248.38 ];

% p_bc = R_bc * -t_cb;
p_bc = -[ 0, -1, 0; - 1, 0, 0; 0, 0, -1 ] * [0.0652; - 0.0207; - 0.0081];

fontsize= 18;

data = dlmread(filename, ',', 1, 0);
startTime = data(1,1);
endTime = data(end,1); %2.188e13-4.5e9;
index= find(abs(endTime - data(:,1))<2.5e7, 1);
if(isempty(index))
    index= size(data,1);
end
data= data(1:index,:);

% output the average estimated values for the calibration paramters

period= 5; %sec
startIndex = find(abs(endTime -period - data(:,1))<2.5e7, 1);
if(isempty(startIndex))
    startIndex=1;
end

estimate_average= mean(data(startIndex:end, :), 1);
fprintf('bg ba\n');
fprintf('%.4f\n', estimate_average(13:18));
fprintf('pbinc\n');
fprintf('%.3f\n', estimate_average(46:48)*100);
fprintf('fx fy cx cy\n');
fprintf('%.3f\n', estimate_average(49:52)*2);
fprintf('k1 k2\n');
fprintf('%.3f\n', estimate_average(53:54));
fprintf('p1 p2\n');
fprintf('%.6f\n', estimate_average(55:56));
fprintf('tr [ms]\n');
fprintf('%.3f\n', estimate_average(58)*1e3);

% ground truth file must have the same number of rows as output data gt = [];
if(~isempty(gt))
    gt = csvread('G:\state_groundtruth_estimate0\data_copy.csv');
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
    data(:,1)= (data(:,1)- startTime)/1e9;
end

figNumber = 0;
figNumber = figNumber +1;
figure(figNumber);
plot3(data(:,3), data(:,4), data(:,5), '-b'); hold on;
if(~isempty(gt))
    plot3(gt(:,2), gt(:,3), gt(:,4), '-r');
end
plot3(data(1,3),data(1,4),data(1,5), '-sr');
plot3(data(end,3),data(end,4),data(end,5), '-sb');
if(~isempty(gt))
    legend('msckf2','gt','start','finish');
else
    legend('msckf2','start','finish');
end
title ('p_b^g', 'FontSize', fontsize);
xlabel ('x [m]', 'FontSize', fontsize);
ylabel ('y [m]', 'FontSize', fontsize);
zlabel ('z [m]', 'FontSize', fontsize);
axis equal;
grid on;
set(gca,'FontSize',fontsize);

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

figNumber = figNumber +1;
figure(figNumber);
plot(data(:,1), data(:,6), '-r');
hold on;
plot(data(:,1), data(:,7), '-g');
plot(data(:,1), data(:,8), '-b');

if(~isempty(gt))
    plot(gt(:,1), gt(:,6), '--r');
    plot(gt(:,1), gt(:,7), '--g');
    plot(gt(:,1), gt(:,8), '--b');
end
xlabel('time [sec]', 'FontSize', fontsize);
legend('qx', 'qy', 'qz', 'gt qx','gt qy','gt qz');
grid on;
set(gca,'FontSize',fontsize);

figNumber = figNumber +1;
figure(figNumber);
plot(data(:,1), data(:, 10), '-r'); hold on;
plot(data(:,1), data(:, 11), '-g');
plot(data(:,1), data(:, 12), '-b');
if(~isempty(gt))
    plot(gt(:,1), gt(:, 9), '--r');
    plot(gt(:,1), gt(:, 10), '--g');
    plot(gt(:,1), gt(:, 11), '--b');
end
legend('x', 'y', 'z','gt x','gt y','gt z');
xlabel('time [sec]');
ylabel('m/sec');
title('v_b^g')
grid on;
set(gca,'FontSize',fontsize);

figNumber = figNumber +1;
figure(figNumber);
pid180 = 180/pi;
plot(data(:,1), data(:,13)*pid180, '-r'); hold on;
plot(data(:,1), data(:,14)*pid180, '-g');
plot(data(:,1), data(:,15)*pid180, '-b');

plot(data(:,1), (3*data(:,68)+ data(:, 13))*pid180, '--r');
plot(data(:,1), (3*data(:,69)+ data(:, 14))*pid180, '--g');
plot(data(:,1), (3*data(:,70)+ data(:, 15))*pid180, '--b');
if(~isempty(gt))
    plot(gt(:,1), gt(:,12)*pid180, '-k'); hold on;
    plot(gt(:,1), gt(:,13)*pid180, '-c');
    plot(gt(:,1), gt(:,14)*pid180, '-m');
end
plot(data(:,1), (-3*data(:,68)+ data(:, 13))*pid180, '--r');
plot(data(:,1), (-3*data(:,69)+ data(:, 14))*pid180, '--g');
plot(data(:,1), (-3*data(:,70)+ data(:, 15))*pid180, '--b');

xlabel('time [sec]', 'FontSize', fontsize);
s = sprintf('%c/sec', char(176));
ylabel(s, 'FontSize', fontsize);
title('b_g', 'FontSize', fontsize);
legend('x', 'y', 'z', '3\sigma_x','3\sigma_y','3\sigma_z', 'gt x','gt y','gt z');
set(gca,'FontSize',fontsize);
grid on;

figNumber = figNumber +1;
figure(figNumber);
plot(data(:,1), data(:, 16), '-r'); hold on;
plot(data(:,1), data(:, 17), '-g');
plot(data(:,1), data(:, 18), '-b');

plot(data(:,1), 3*data(:,71)+ data(:, 16), '--r');
plot(data(:,1), 3*data(:,72)+ data(:, 17), '--g');
plot(data(:,1), 3*data(:,73)+ data(:, 18), '--b');
if(~isempty(gt))
    plot(gt(:,1), gt(:, 15), '-k'); hold on;
    plot(gt(:,1), gt(:, 16), '-c');
    plot(gt(:,1), gt(:, 17), '-m');
end
plot(data(:,1), -3*data(:,71)+ data(:, 16), '--r');
plot(data(:,1), -3*data(:,72)+ data(:, 17), '--g');
plot(data(:,1), -3*data(:,73)+ data(:, 18), '--b');

xlabel('time [sec]', 'FontSize', fontsize);
ylabel('m/sec^2', 'FontSize', fontsize);
title('b_a', 'FontSize', fontsize);
legend('x', 'y', 'z','3\sigma_x','3\sigma_y','3\sigma_z', 'gt x','gt y','gt z');
set(gca,'FontSize',fontsize);
grid on;

figNumber = figNumber +1;
figure(figNumber);
ruler=100;
plot(data(:,1), data(:, 46)*ruler, '-r'); hold on;
plot(data(:,1), data(:, 47)*ruler, '-g');
plot(data(:,1), data(:, 48)*ruler, '-b');

plot(data(:,1), (3*data(:,101)+ data(:, 46))*ruler, '--r');
plot(data(:,1), (3*data(:,102)+ data(:, 47))*ruler, '--g');
plot(data(:,1), (3*data(:,103)+ data(:, 48))*ruler, '--b');

if(~isempty(gt) && ~isempty(p_bc))
    plot(gt(:,1), ones(size(gt,1),1)*p_bc(1)*ruler, '-k'); hold on;
    plot(gt(:,1), ones(size(gt,1),1)*p_bc(2)*ruler, '-c');
    plot(gt(:,1), ones(size(gt,1),1)*p_bc(3)*ruler, '-m');
end
plot(data(:,1), (-3*data(:,101)+ data(:, 46))*ruler, '--r');
plot(data(:,1), (-3*data(:,102)+ data(:, 47))*ruler, '--g');
plot(data(:,1), (-3*data(:,103)+ data(:, 48))*ruler, '--b');

xlabel('time [sec]', 'FontSize', fontsize);
ylabel('cm', 'FontSize', fontsize);
title('p_c^b', 'FontSize', fontsize);
legend('x', 'y', 'z','3\sigma_x','3\sigma_y','3\sigma_z', 'gt x','gt y','gt z');
set(gca,'FontSize',fontsize);
grid on;
saveas(gcf,[output_dir, '\Error p_CB'],'epsc');


figNumber = figNumber +1;
figure(figNumber);

plot(data(:,1), data(:, 49)-nominal_intrinsics(1), '-r'); hold on;
plot(data(:,1), data(:, 50)-nominal_intrinsics(2), '-g');
plot(data(:,1), data(:, 51)-nominal_intrinsics(3), '-b');
plot(data(:,1), data(:, 52)-nominal_intrinsics(4), '-k');

% photoscan_calib_result= [1551.91 1551.91 934.853 529.925]/2;
% tile_data= repmat(photoscan_calib_result, size(data,1), 1);
% plot(data(:,1), tile_data(:,1)-nominal_intrinsics(1), '.r'); hold on;
% plot(data(:,1), tile_data(:,2)-nominal_intrinsics(2), '.g');
% plot(data(:,1), tile_data(:,3)-nominal_intrinsics(3), '.b');
% plot(data(:,1), tile_data(:,4)-nominal_intrinsics(4), '.k');

plot(data(:,1), data(:, 49)-nominal_intrinsics(1)+3*data(:, 104), '--r'); hold on;
plot(data(:,1), data(:, 50)-nominal_intrinsics(2)+3*data(:, 105), '--g');
plot(data(:,1), data(:, 51)-nominal_intrinsics(3)+3*data(:, 106), '--b');
plot(data(:,1), data(:, 52)-nominal_intrinsics(4)+3*data(:, 107), '--k');

plot(data(:,1), data(:, 49)-nominal_intrinsics(1)-3*data(:, 104), '--r'); hold on;
plot(data(:,1), data(:, 50)-nominal_intrinsics(2)-3*data(:, 105), '--g');
plot(data(:,1), data(:, 51)-nominal_intrinsics(3)-3*data(:, 106), '--b');
plot(data(:,1), data(:, 52)-nominal_intrinsics(4)-3*data(:, 107), '--k');

% legend('f_x','f_y','c_x','c_y', 'ref f_x','ref f_y','ref c_x','ref c_y','3\sigma_f_x', '3\sigma_f_y','3\sigma_c_x','3\sigma_c_y');
legend('f_x','f_y','c_x','c_y','3\sigma_f_x', '3\sigma_f_y','3\sigma_c_x','3\sigma_c_y');
grid on;
xlabel('time [sec]');
title('Error ($f_x$, $f_y$), ($c_x$, $c_y$)', 'Interpreter', 'Latex');
ylabel('pixel');
set(gca,'FontSize',fontsize);


figNumber = figNumber +1;
figure(figNumber);
p1p2Scale= 1e2;
showP1P2= true;
plot(data(:,1), data(:, 53), '-r'); hold on;
plot(data(:,1), data(:, 54), '-g');
if(showP1P2)
    plot(data(:,1), data(:, 55)*p1p2Scale, '-b');
    plot(data(:,1), data(:, 56)*p1p2Scale, '-k');
end
% photoscan_calib_result= [0.1377 -0.3436 5.2e-4 -3.5e-4];
% tile_data= repmat(photoscan_calib_result, size(data,1), 1);
% plot(data(:,1), tile_data(:,1), '.r'); hold on;
% plot(data(:,1), tile_data(:,2), '.g');
if(showP1P2)
%     plot(data(:,1), tile_data(:,3)*p1p2Scale, '.b');
%     plot(data(:,1), tile_data(:,4)*p1p2Scale, '.k');
end
plot(data(:,1), data(:, 53)+3*data(:, 108), '--r');
plot(data(:,1), data(:, 54)+3*data(:, 109), '--g');
if(showP1P2)
    plot(data(:,1), (data(:, 55)+3*data(:, 110))*p1p2Scale, '--b');
    plot(data(:,1), (data(:, 56)+3*data(:, 111))*p1p2Scale, '--k');
end
plot(data(:,1), data(:, 53)-3*data(:, 108), '--r');
plot(data(:,1), data(:, 54)-3*data(:, 109), '--g');
if(showP1P2)
    plot(data(:,1), (data(:, 55)-3*data(:, 110))*p1p2Scale, '--b');
    plot(data(:,1), (data(:, 56)-3*data(:, 111))*p1p2Scale, '--k');
    
    legend('k_1','k_2','p_1','p_2','ref k_1','ref k_2','ref p_1','ref p_2', '3\sigma_{k_1}', '3\sigma_{k_2}', '3\sigma_{p_1}', '3\sigma_{p_2}');
    
    title('($k_1$, $k_2$, $p_1$, $p_2$)', 'Interpreter', 'Latex');
else
    legend('k_1','k_2','ref k_1','ref k_2', '3\sigma_{k_1}', '3\sigma_{k_2}');
    legend('k_1','k_2', '3\sigma_{k_1}', '3\sigma_{k_2}');
    
    title('($k_1$, $k_2$)', 'Interpreter', 'Latex');
end
grid on;
xlabel('time [sec]');
set(gca,'FontSize',fontsize);

figNumber = intermediatePlotter( ...
    figNumber, data,  nominal_intrinsics'*2, output_dir, fontsize);
