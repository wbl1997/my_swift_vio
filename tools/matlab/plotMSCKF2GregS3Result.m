function plotMSCKF2GregS3Result()
% plot the msckf2 output against the orb-vo output
close all;
addpath('C:\JianzhuHuai\GPS_IMU\programs\matlab_ws\voicebox');
data = csvread('F:\west_campus_parking_lot\calibration\Greg\output\msckf2_estimator_output_greg1.csv');
data = csvread('F:\west_campus_parking_lot\calibration\Greg\output\msckf2_estimator_output_greg_floatTgTsTa.csv');


duration =143.3; % only use first duration secs

data(:,1)= data(:,1)*1e-9;

data2 = data(:, [1,3:5]);
finishIndex = find(abs(data2(:,1) - data2(1,1) -duration)< 2e-2,1);
if(isempty(finishIndex))
    finishIndex = size(data2,1);
end
data2= data2(1: finishIndex, :);

maxTime= data2(end,1);
deltaT=21736.25588; %sec
data3 = load('F:\west_campus_parking_lot\calibration\Greg\output\greg_s3_calib_orbvo1.txt');
data3 = load('F:\west_campus_parking_lot\calibration\Greg\output\greg_s3_calib_orbslam.txt');
data3(:,1) = data3(:,1) + deltaT;
data4= data3;
finishIndex = find(abs(data4(:,1)- maxTime)< 2e-2,1);
if(isempty(finishIndex))
    finishIndex = size(data4,1);
end
data4= data4(1: finishIndex, 1:4);

[data_trans, s_res, R_res]=alignTwoPositionSet(data4, data2);
data4= data_trans;

figure(1)
[val, startindex] = min(abs(data2(:,1) - data4(1,1)));

plot3(data2(startindex:end,2), data2(startindex:end,3), data2(startindex:end,4), '-r'); hold on;
plot3(data2(startindex,2), data2(startindex,3), data2(startindex,4), 'sr', 'MarkerSize', 22); hold on
plot3(data4(:,2), data4(:,3), data4(:,4), '-k'); hold on
plot3(data4(1,2), data4(1,3), data4(1,4), 'sk', 'MarkerSize', 22); hold on
% plot3(data3(1:maxIndex,4)*scale, data3(1:maxIndex,2)*scale, -data3(1:maxIndex,3)*scale, '-b');
% plot3(data3(1,4)*scale, data3(1,2)*scale, -data3(1,3)*scale, 'sb');
% plot3(data4(:,4)*scale, data4(:,2)*scale, -data4(:,3)*scale, '--m');
h_legend=legend( 'MSCKF','MSCKF Start', 'ORB-VO Scaled', 'ORB-VO Start');
set(h_legend,'FontSize',22);
xlabel('x [m]');
ylabel('y [m]');
zlabel('z [m]');
set(gca,'FontSize',22);
axis equal;
grid on;
figure(2)

startTime = data(1,1);
data(:,1) = data(:,1)- startTime;
data3(:,1) = data3(:,1)- startTime;

q_res = rotro2qr(R_res);
euler= zeros(size(data,1), 3);
euler3= zeros(size(data3,1), 3);
for i=1: size(data,1)
    euler(i,:) = rotqr2eu('zyx', [data(i, 9), data(i, 6:8)]')';
    if(data(i,9)<0)
    euler(i,:) = -data(i, 6:8);
    else
        euler(i,:) = data(i, 6:8);
    end
end
R_sc =[0 -1 0; -1 0 0; 0 0 -1];
for i=1: size(data3,1)
    euler3(i,:) = rotro2eu('zyx', R_res*rotqr2ro([data3(i, 8), data3(i, 5:7)]')*R_sc')';
    qr = rotro2qr(R_res*rotqr2ro([data3(i, 8), data3(i, 5:7)]')*R_sc');
    if(qr(1)<0)
        euler3(i,:) = -qr(2:4)';
    else
        euler3(i,:) = qr(2:4)';
    end
end
plot(data(:, 1), euler(:, 1), '-r'); hold on;
plot(data(:, 1), euler(:, 2), '-g');
plot(data(:, 1), euler(:, 3), '-b');

plot(data3(:, 1), euler3(:, 1), '.r', 'MarkerSize', 12); hold on;
plot(data3(:, 1), euler3(:, 2), '.g', 'MarkerSize', 12);
plot(data3(:, 1), euler3(:, 3), '.b', 'MarkerSize', 12);

h_legend=legend('MSCKF q_x', 'MSCKF q_y', 'MSCKF q_z', 'ORB-VO q_x', 'ORB-VO q_y', 'ORB-VO q_z');
set(h_legend,'FontSize',22);
xlabel('time [sec]');
ylabel('[1]');
set(gca,'FontSize',22);
end

function [data_trans, s_res, R_res]=alignTwoPositionSet(data, gt)
% data and gt rows x 4, each col time in sec, x,y,z
% return transformed data aligned to ground truth
% umeyama transformation

   % associate the data by timestamps, discrepancy less than 0.02sec
   
    assocIndex = zeros(size(data,1), 1);   
    for i=1:size(data,1)
        [val, index] = min(abs(gt(:,1) - data(i,1)));
        if(val < 1e-2)
            assocIndex(i,1)= index;
        end      
    end
    assoc2 = zeros(length(assocIndex), 2);
    lastIndex=1;
    for i=1:size(data,1)
        if(assocIndex(i)~=0)
            assoc2(lastIndex, 1) = i;
            assoc2(lastIndex, 2) = assocIndex(i);
            lastIndex = lastIndex+1;
        end
    end
    assocIndex = assoc2(1: lastIndex-1, :);
    % association end
 
    % umeyama transform to gt
    src = data(assocIndex(:,1), 2:4); % remove potential outliers
    dst = gt(assocIndex(:,2), 2:4);
    [s_res, R_res, t_res] = ralign(src',dst');
    data_trans= data;
    data_trans(:,2:4) = (s_res*R_res*data(:, 2:4)'+repmat(t_res, 1, size(data,1)))';
end