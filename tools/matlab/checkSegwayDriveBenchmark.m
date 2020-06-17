% By inspecting the reference trajectories.
% We found that T_Cp0_C is wrong in orientation.
% The below two files are from
% http://drive.segwayrobotics.com/dataset/download.
hokuyo = '/B2_lidar_2018-11-22_14-30-03_T_Wl_C.txt';
encoder = '/B2_lidar_2018-11-22_14-30-03_T_Cp0_C.txt';
hokuyo = '/B2_lidar_2018-09-21_14-46-46__B2_T_Wl_C.txt';
encoder = '/B2_lidar_2018-09-21_14-46-46__B2_T_Cp0_C.txt';

close all;
hdata = readmatrix(hokuyo);
edata = readmatrix(encoder);
drawTrajectoryWithCoordinateFrames({hdata, edata}, {'hokuyo-T-Wl-C', 'wheel-T-Cp0-C'}, 2:4, 5:8);
