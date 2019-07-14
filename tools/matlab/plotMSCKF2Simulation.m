%usage: matlab

close all;
addpath('C:\JianzhuHuai\GPS_IMU\programs\matlab_ws\voicebox');
addpath('C:\JianzhuHuai\GPS_IMU\programs\matlab_ws\instk');
path = 'G:/temp';
outputPath = 'G:/temp';
filename= [path, '/sinusoidMSCKF2_590.txt'];
data = load(filename);
finishTime = 400;
data(:,1)= data(:,1)- data(1,1);
index= find(data(:,1)>finishTime, 1);
if(isempty(index))
    index= size(data,1);
end
data = data(1:index,:);

% ground truth file must have the same number of rows as output data
gt = load([path, '/sinusoidTruth.txt']);
gt = gt(1:index, :);

gtPoints =[];
gtPoints = load([path, '/sinusoidPoints.txt']);
%indices=find(abs(gtPoints(:, 4))<2.3);
%gtPoints= gtPoints(indices, :);

nees =[];
%nees = load([path, '/sinusoidNEES.txt']);

rmse =[];
%rmse = load([path, '/sinusoidRMSE.txt']);
if(~isempty(rmse))
  rmse(:,1) = rmse(:,1) - rmse(1,1);
end

imuData =[];
%imuData = load([path, '/sinusoidInertial.txt']);

alpha= zeros(size(gt,1),3);
for i=1:size(gt,1)
qs2w= gt(i, [9,6:8]);
qs2w_hat = data(i, [9,6:8]);
alpha(i,:)= unskew(rotqr2ro(qs2w')*rotqr2ro(qs2w_hat')'-eye(3))';
end

figNumber = 0;
figNumber = figNumber +1;
figure(figNumber);

%plot3(data(:,3), data(:,4), data(:,5),'-b'); 
hold on;
plot3(gt(:,3), gt(:,4), gt(:,5), '-r');
plot3(data(1,3), data(1,4), data(1,5),'sr', 'MarkerSize', 18);
plot3(data(end,3), data(end,4), data(end,5), 'sb', 'MarkerSize', 18);
%plot3(gt(1,3), gt(1,4), gt(1,5),'sg', 'MarkerSize', 18);
%plot3(gt(end,3), gt(end,4), gt(end,5), 'sb', 'MarkerSize', 18);
if(~isempty(gtPoints))
    plot3(gtPoints(:,2),gtPoints(:,3),gtPoints(:,4), '+b');
end
h_legend= legend('ground truth','start','finish');
set(h_legend,'FontSize',22);
xlabel('x [m]','FontSize',22);
ylabel('y [m]','FontSize',22);
zlabel('z [m]','FontSize',22);
set(gca,'FontSize',22);
xlim([-5.5,5.5]);
ylim([-5.5,5.5]);
zlim([-3.0,3.0]);

axis equal;
grid on;

figNumber = figNumber +1;
figure(figNumber);
plot(data(:,1), data(:,6), '-r');
hold on;
plot(data(:,1), data(:,7), '-g');
plot(data(:,1), data(:,8), '-b');
plot(data(:,1), gt(:,6), '--r');
plot(data(:,1), gt(:,7), '--g');
plot(data(:,1), gt(:,8), '--b');
legend('qx', 'qy', 'qz', 'gt qx','gt qy','gt qz');
grid on;
xlabel('time [sec]');

figNumber = figNumber +1;
figure(figNumber);
plot(data(:,1), data(:, 10), '-r'); hold on;
plot(data(:,1), data(:, 11), '-g');
plot(data(:,1), data(:, 12), '-b');
plot(data(:,1), gt(:, 10), '--r');
plot(data(:,1), gt(:, 11), '--g');
plot(data(:,1), gt(:, 12), '--b');
legend('v_x', 'v_y', 'v_z','gt v_x','gt v_y','gt v_z');
grid on;
xlabel('time [sec]');
ylabel('velocity [m/s]');
set(gca,'FontSize',18);

figNumber = figNumber +1;
figure(figNumber);
pid180= 180/pi;
plot(data(:,1), data(:,13)*pid180, '-r'); hold on;
plot(data(:,1), data(:,14)*pid180, '-g');
plot(data(:,1), data(:,15)*pid180, '-b');
plot(data(:,1), 3*data(:,68)*pid180, '--r');
plot(data(:,1), 3*data(:,69)*pid180, '--g');
plot(data(:,1), 3*data(:,70)*pid180, '--b');
plot(data(:,1), -3*data(:,68)*pid180, '--r');
plot(data(:,1), -3*data(:,69)*pid180, '--g');
plot(data(:,1), -3*data(:,70)*pid180, '--b');
h_legend=legend('x', 'y', 'z', '3\sigma_x','3\sigma_y','3\sigma_z');
set(h_legend,'FontSize',18);
grid on;
xlabel('time [sec]', 'FontSize', 18);
title('$\mathbf{b}_g$', 'Interpreter', 'Latex');
s = sprintf('%c/sec', char(176));
ylabel(s,'FontSize', 18);
set(gca,'FontSize',18);
saveas(gcf,[outputPath, '\Error bg'],'epsc');

figNumber = figNumber +1;
figure(figNumber);
plot(data(:,1), data(:, 16), '-r'); hold on;
plot(data(:,1), data(:, 17), '-g');
plot(data(:,1), data(:, 18), '-b');
plot(data(:,1), 3*data(:,71), '--r');
plot(data(:,1), 3*data(:,72), '--g');
plot(data(:,1), 3*data(:,73), '--b');
plot(data(:,1), -3*data(:,71), '--r');
plot(data(:,1), -3*data(:,72), '--g');
plot(data(:,1), -3*data(:,73), '--b');
h_legend=legend('x', 'y', 'z', '3\sigma_x','3\sigma_y','3\sigma_z');
set(h_legend,'FontSize',18);
grid on;
xlabel('time [sec]', 'FontSize', 18);
title('$\mathbf{b}_a$', 'Interpreter', 'Latex');
ylabel('m/sec^2', 'FontSize', 18);
set(gca,'FontSize',18);
saveas(gcf,[outputPath, '\Error ba'],'epsc');

intermediatePlotter(data);

figNumber = figNumber + 5;
figure(figNumber);
plot(data(:,1), gt(:,3)-data(:, 3), '-r'); hold on;
plot(data(:,1), gt(:,4)-data(:, 4), '-g');
plot(data(:,1), gt(:,5)-data(:, 5), '-b');

plot(data(:,1), 3*data(:, 59), '--r');
plot(data(:,1), 3*data(:, 60), '--g');
plot(data(:,1), 3*data(:, 61), '--b');
plot(data(:,1), -3*data(:, 59), '--r');
plot(data(:,1), -3*data(:, 60), '--g');
plot(data(:,1), -3*data(:, 61), '--b');
h_legend=legend('x','y','z');
set(h_legend,'FontSize',18);
grid on;
xlabel('time [sec]', 'FontSize', 18);
title('$\mathbf{p}_b^g$', 'Interpreter', 'Latex');
ylabel('m', 'FontSize', 18);
saveas(gcf,[outputPath, '\Error p_GB'],'epsc');

figNumber = figNumber +1;
figure(figNumber);

plot(data(:,1), alpha(:,1)*pid180, '-r'); hold on;
plot(data(:,1), alpha(:,2)*pid180, '-g');
plot(data(:,1), alpha(:,3)*pid180, '-b');

plot(data(:,1), 3*data(:, 62)*pid180, '--r');
plot(data(:,1), 3*data(:, 63)*pid180, '--g');
plot(data(:,1), 3*data(:, 64)*pid180, '--b');
plot(data(:,1), -3*data(:, 62)*pid180, '--r');
plot(data(:,1), -3*data(:, 63)*pid180, '--g');
plot(data(:,1), -3*data(:, 64)*pid180, '--b');
legend('x','y','z');
grid on;
xlabel('time [sec]', 'FontSize', 18);
title('$\delta\mathbf{\theta}$', 'Interpreter', 'Latex');
s = sprintf('%c', char(176));
ylabel(s, 'FontSize', 18);
saveas(gcf,[outputPath, '\Error R_GB'],'epsc');

figNumber = figNumber +1;
figure(figNumber);
plot(data(:,1), gt(:,10)-data(:, 10), '-r'); hold on;
plot(data(:,1), gt(:,11)-data(:, 11), '-g');
plot(data(:,1), gt(:,12)-data(:, 12), '-b');

plot(data(:,1), 3*data(:, 65), '--r');
plot(data(:,1), 3*data(:, 66), '--g');
plot(data(:,1), 3*data(:, 67), '--b');
plot(data(:,1), -3*data(:, 65), '--r');
plot(data(:,1), -3*data(:, 66), '--g');
plot(data(:,1), -3*data(:, 67), '--b');
legend('x','y','z');
grid on;
xlabel('time [sec]', 'FontSize', 18);
title('$\mathbf{v}_b^g$', 'Interpreter', 'Latex');
ylabel('m/sec', 'FontSize', 18);
saveas(gcf,[outputPath, '\Error v_GB'],'epsc');

if(~isempty(nees))
figNumber = figNumber +1;
figure(figNumber);
nees(:,1) =nees(:,1) - nees(1,1);
plot(nees(:,1), nees(:,2), '-r'); hold on;
plot(nees(:,1), nees(:,3), '-g');
plot(nees(:,1), nees(:,4), '-b');
plot(nees(:,1), ones(size(nees(:,1),1),1)*3, '-r'); hold on;
plot(nees(:,1), ones(size(nees(:,1),1),1)*1, '-g');
plot(nees(:,1), ones(size(nees(:,1),1),1)*4, '-b');
h_legend= legend('position', 'attitude', 'pose');
set(h_legend,'FontSize',22);
xlabel('time [sec]', 'FontSize', 22);
title('nees');
set(gca,'FontSize',22);
saveas(gcf,[outputPath, '\NEES_GB'],'epsc');
end

if(~isempty(imuData))
figNumber = figNumber +1;
figure(figNumber);

plot(imuData(:,1), imuData(:,2), '--r'); hold on;
plot(imuData(:,1), imuData(:,3), '--g');
plot(imuData(:,1), imuData(:,4), '--b');
legend('\omega_x', '\omega_y', '\omega_z')
xlabel('time [sec]');
title('gyroscope data');
ylabel('rad/sec')

figNumber = figNumber +1;
figure(figNumber);
plot(imuData(:,1), imuData(:,8), '-r'); hold on;
plot(imuData(:,1), imuData(:,9), '-g');
plot(imuData(:,1), imuData(:,10), '-b');
legend('\omega_x', '\omega_y', '\omega_z')
xlabel('time [sec]');
title('noisy gyro data');
ylabel('rad/sec')

figNumber = figNumber +1;
figure(figNumber);
plot(imuData(:,1), imuData(:,5), '--r'); hold on;
plot(imuData(:,1), imuData(:,6), '--g');
plot(imuData(:,1), imuData(:,7), '--b');
legend('a_x', 'a_y', 'a_z')
xlabel('time [sec]');
title('accelerometer data');
ylabel('m/sec^2');

figNumber = figNumber +1;
figure(figNumber);
plot(imuData(:,1), imuData(:,11), '-r'); hold on;
plot(imuData(:,1), imuData(:,12), '-g');
plot(imuData(:,1), imuData(:,13), '-b');
legend('a_x', 'a_y', 'a_z');
xlabel('time [sec]');
title('noisy accelerometer data');
ylabel('m/sec^2');
end

if(~isempty(rmse)) 
    
figNumber = figNumber +1;
figure(figNumber);
plot(rmse(:,1), rmse(:,2), '-r'); hold on;
plot(rmse(:,1), rmse(:,3), '-g');
plot(rmse(:,1), rmse(:,4), '-b');
h_legend=legend('x', 'y', 'z');
set(h_legend,'FontSize',22);
xlabel('time [sec]', 'FontSize', 22);
title('$\mathbf{p}_b^g$', 'Interpreter', 'Latex');
ylabel('m', 'FontSize', 22);
set(gca,'FontSize',22);
saveas(gcf,[outputPath, '\RMSE_p_GB'],'epsc');


figNumber = figNumber +1;
figure(figNumber);
plot(rmse(:,1), rmse(:,5)*pid180, '-r'); hold on;
plot(rmse(:,1), rmse(:,6)*pid180, '-g');
plot(rmse(:,1), rmse(:,7)*pid180, '-b');
h_legend=legend('roll', 'pitch', 'yaw');
set(h_legend,'FontSize',22);
xlabel('time [sec]', 'FontSize', 22);
title('$\delta\mathbf{\theta}$', 'Interpreter', 'Latex');
s = sprintf('%c', char(176));
ylabel(s, 'FontSize', 22);
set(gca,'FontSize',22);
saveas(gcf,[outputPath, '\RMSE_q_GB'],'epsc');

figNumber = figNumber +1;
figure(figNumber);
plot(rmse(:,1), rmse(:,8), '-r'); hold on;
plot(rmse(:,1), rmse(:,9), '-g');
plot(rmse(:,1), rmse(:,10), '-b');
h_legend=legend('x', 'y', 'z');
set(h_legend,'FontSize',22);
xlabel('time [sec]', 'FontSize', 22);
title('$\mathbf{v}_b^g$', 'Interpreter', 'Latex');
ylabel('m/sec', 'FontSize', 22);
set(gca,'FontSize',22);
saveas(gcf,[outputPath, '\RMSE_v_GB'],'epsc');

figNumber = figNumber +1;
figure(figNumber);
plot(rmse(:,1), rmse(:,11)*pid180, '-r'); hold on;
plot(rmse(:,1), rmse(:,12)*pid180, '-g');
plot(rmse(:,1), rmse(:,13)*pid180, '-b');
h_legend = legend('x', 'y','z');
set(h_legend,'FontSize',22);
xlabel('time [sec]', 'FontSize',22);
title('$\mathbf{b}_{g}$', 'Interpreter', 'Latex');
s = sprintf('%c/sec', char(176));
ylabel(s, 'FontSize',22);
set(gca,'FontSize',22);
saveas(gcf,[outputPath, '\RMSE bg'],'epsc');

figNumber = figNumber +1;
figure(figNumber);
plot(rmse(:,1), rmse(:,14), '-r'); hold on;
plot(rmse(:,1), rmse(:,15), '-g');
plot(rmse(:,1), rmse(:,16), '-b');
h_legend= legend('x', 'y', 'z');
set(h_legend,'FontSize',22);
xlabel('time [sec]', 'FontSize',22);
title('$\mathbf{b}_a$', 'Interpreter', 'Latex');
ylabel('m/sec^2', 'FontSize',22);
set(gca,'FontSize',22);
saveas(gcf,[outputPath, '\RMSE ba'],'epsc');

figNumber = figNumber +1;
figure(figNumber);
plot(rmse(:,1), rmse(:,17), '-r'); hold on;
plot(rmse(:,1), rmse(:,18), '-g');
plot(rmse(:,1), rmse(:,19), '-b');
plot(rmse(:,1), rmse(:,20), '-k');
plot(rmse(:,1), rmse(:,21), '.k');
plot(rmse(:,1), rmse(:,22), '.b');
plot(rmse(:,1), rmse(:,23), '-c');
plot(rmse(:,1), rmse(:,24), '-m');
plot(rmse(:,1), rmse(:,25), '-y');
h_legend=legend('1', '2', '3', ...
    '4', '5', '6', '7', ...
    '8', '9');
set(h_legend,'FontSize',22);
xlabel('time [sec]', 'FontSize',22);
title('$\mathbf{T}_g$', 'Interpreter', 'Latex');
set(gca,'FontSize',22);
saveas(gcf,[outputPath, '\RMSE Tg'],'epsc');

figNumber = figNumber +1;
figure(figNumber);
plot(rmse(:,1), rmse(:,26), '-r'); hold on;
plot(rmse(:,1), rmse(:,27), '-g');
plot(rmse(:,1), rmse(:,28), '-b');
plot(rmse(:,1), rmse(:,29), '-k');
plot(rmse(:,1), rmse(:,30), '.k');
plot(rmse(:,1), rmse(:,31), '.b');
plot(rmse(:,1), rmse(:,32), '-c');
plot(rmse(:,1), rmse(:,33), '-m');
plot(rmse(:,1), rmse(:,34), '-y');
h_legend=legend('1', '2', '3', '4',...
    '5', '6', '7', '8', '9');
set(h_legend,'FontSize',22);
xlabel('time [sec]','FontSize',22);
title('$\mathbf{T}_s$', 'Interpreter', 'Latex');
set(gca,'FontSize',22);
saveas(gcf,[outputPath, '\RMSE Ts'],'epsc');

figNumber = figNumber +1;
% figure(figNumber);
plot(rmse(:,1), rmse(:,35), '-r'); hold on;
plot(rmse(:,1), rmse(:,36), '-g');
plot(rmse(:,1), rmse(:,37), '-b');
plot(rmse(:,1), rmse(:,38), '-k');
plot(rmse(:,1), rmse(:,39), '.k');
plot(rmse(:,1), rmse(:,40), '.b');
plot(rmse(:,1), rmse(:,41), '-c');
plot(rmse(:,1), rmse(:,42), '-m');
plot(rmse(:,1), rmse(:,43), '-y');
h_legend=legend('1', '2', '3', '4',...
    '5', '6', '7', '8', '9');
set(h_legend,'FontSize',22);
xlabel('time [sec]', 'FontSize',22);
title('$\mathbf{T}_a$', 'Interpreter', 'Latex');
set(gca,'FontSize',22);
saveas(gcf,[outputPath, '\RMSE Ta'],'epsc');

figNumber = figNumber +1;
figure(figNumber);
ruler= 100;
plot(rmse(:,1), rmse(:,44)*ruler, '-r'); hold on;
plot(rmse(:,1), rmse(:,45)*ruler, '-g');
plot(rmse(:,1), rmse(:,46)*ruler, '-b');
h_legend=legend('x', 'y', 'z');
set(h_legend,'FontSize',22);
xlabel('time [sec]', 'FontSize',22);
title('$\mathbf{p}_b^c$', 'Interpreter', 'Latex');
ylabel('cm', 'FontSize',22);
set(gca,'FontSize',22);
saveas(gcf,[outputPath, '\RMSE p_CB'],'epsc');

figNumber = figNumber +1;
figure(figNumber);
plot(rmse(:,1), rmse(:,47), '-r'); hold on;
plot(rmse(:,1), rmse(:,48), '-g');
plot(rmse(:,1), rmse(:,49), '-b');
plot(rmse(:,1), rmse(:,50), '-k');
h_legend=legend('f_x', 'f_y', 'c_x', 'c_y');
set(h_legend,'FontSize',22);
xlabel('time [sec]', 'FontSize',22);
ylabel('pixel', 'FontSize',22);
title('($f_x$, $f_y$), ($c_x$, $c_y$)', 'Interpreter', 'Latex');
set(gca,'FontSize',22);
saveas(gcf,[outputPath, '\RMSE intrinsics'],'epsc');

figNumber = figNumber +1;
figure(figNumber);
plot(rmse(:,1), rmse(:,51), '-r'); hold on;
plot(rmse(:,1), rmse(:,52), '-g');
plot(rmse(:,1), rmse(:,53), '-b');
plot(rmse(:,1), rmse(:,54), '-k');
h_legend=legend('k_1', 'k_2', 'p_1', 'p_2');
set(h_legend,'FontSize',22);
xlabel('time [sec]', 'FontSize',22);
title('($k_1$, $k_2$, $p_1$, $p_2$)', 'Interpreter', 'Latex');
set(gca,'FontSize',22);
saveas(gcf,[outputPath, '\RMSE distortion'],'epsc');

figNumber = figNumber +1;
figure(figNumber);
clock= 1e3;
plot(rmse(:,1), rmse(:,55)*clock, '-r'); hold on;
plot(rmse(:,1), rmse(:,56)*clock, '-g');
h_legend=legend('t_d', 't_r');
set(h_legend,'FontSize',22);
xlabel('time [sec]', 'FontSize',22);
ylabel('msec', 'FontSize',22);
title('$t_d$, $t_r$', 'Interpreter', 'Latex');
set(gca,'FontSize',22);
saveas(gcf,[outputPath, '\RMSE td tr'],'epsc');
end
