function plotVinsMonoResult(vins_result_dir, ...
    export_fig_path, output_dir, avg_since_start, avg_trim_end)
close all;
if ~exist('export_fig_path', 'var') || isempty(export_fig_path)
  export_fig_path = '/media/jhuai/Seagate/jhuai/tools/export_fig/';
end
addpath(export_fig_path);
if ~exist('output_dir','var')
    output_dir = input('output_dir of plots, if empty, set to dir of the csv:', 's');
end
if isempty(output_dir)
    output_dir = vins_result_dir;
    disp(['plot_output_dir is set to ', output_dir]);
end

vinsmono_calib_csv = [vins_result_dir, '/vins_result_ex.csv'];
vinsmono_csv = [vins_result_dir, '/vins_result_no_loop.csv'];
nanos_to_sec = 1e-9;
rad_to_deg = 180/pi;
vinsmono_ex_data = readmatrix(vinsmono_calib_csv, 'NumHeaderLines', 0);
assert(size(vinsmono_ex_data, 2) == 9);

vinsmono_data = readmatrix(vinsmono_csv, 'ExpectedNumVariables', 17, 'NumHeaderLines', 0);
assert(size(vinsmono_data, 2) == 17);

% align timestamps
ref_time = min(vinsmono_ex_data(1, 1), vinsmono_data(1, 1));
vinsmono_ex_data(:, 1) = (vinsmono_ex_data(:, 1) - ref_time) * nanos_to_sec;
vinsmono_data(:, 1) = (vinsmono_data(:, 1) - ref_time) * nanos_to_sec;

data=vinsmono_data;
figure;
plot3(data(:, VinsmonoConstants.r(1)), data(:, VinsmonoConstants.r(2)), ...
    data(:, VinsmonoConstants.r(3)), '-b'); hold on;
plot3(data(1, VinsmonoConstants.r(1)), data(1, VinsmonoConstants.r(2)), ...
    data(1, VinsmonoConstants.r(3)), '-or');
plot3(data(end, VinsmonoConstants.r(1)), data(end, VinsmonoConstants.r(2)), ...
    data(end, VinsmonoConstants.r(3)), '-sr');
legend_list = {'vins-mono', 'start', 'finish'};

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


figure;
plot(data(:,1), data(:, VinsmonoConstants.q(1)), '-r');
hold on;
plot(data(:,1), data(:, VinsmonoConstants.q(2)), '-g');
plot(data(:,1), data(:, VinsmonoConstants.q(3)), '-b');

xlabel('time[sec]');
ylabel('quaternion[1]');
legend('qx', 'qy', 'qz');
grid on;
outputfig = [output_dir, '/qxyz_GB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
plot(data(:,1), data(:, VinsmonoConstants.v(1)), '-r');
hold on;
plot(data(:,1), data(:, VinsmonoConstants.v(2)), '-g');
plot(data(:,1), data(:, VinsmonoConstants.v(3)), '-b');
xlabel('time[sec]');
ylabel('v_{GB}[m/s]');
legend('x', 'y', 'z');
grid on;
outputfig = [output_dir, '/v_GB.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
plot(data(:,1), data(:, VinsmonoConstants.b_g(1)), '-r');
hold on;
plot(data(:,1), data(:, VinsmonoConstants.b_g(2)), '-g');
plot(data(:,1), data(:, VinsmonoConstants.b_g(3)), '-b');
xlabel('time[sec]');
ylabel(['b_g[' char(176) '/s]']);
legend('x', 'y', 'z');
grid on;
outputfig = [output_dir, '/b_g.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
plot(data(:,1), data(:, VinsmonoConstants.b_a(1)), '-r');
hold on;
plot(data(:,1), data(:, VinsmonoConstants.b_a(2)), '-g');
plot(data(:,1), data(:, VinsmonoConstants.b_a(3)), '-b');
xlabel('time[sec]');
ylabel('b_a[m/s^2]');
legend('x', 'y', 'z');
grid on;
outputfig = [output_dir, '/b_a.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

data=vinsmono_ex_data;
figure;
plot(data(:,1), data(:, VinsmonoConstants.p_BC(1)), '-r');
hold on;
plot(data(:,1), data(:, VinsmonoConstants.p_BC(2)), '-g');
plot(data(:,1), data(:, VinsmonoConstants.p_BC(3)), '-b');
xlabel('time[sec]');
ylabel('p_{BC}');
legend('x', 'y', 'z');
grid on;
outputfig = [output_dir, '/p_BC.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
plot(data(:,1), data(:, VinsmonoConstants.q_BC(1)), '-r');
hold on;
plot(data(:,1), data(:, VinsmonoConstants.q_BC(2)), '-g');
plot(data(:,1), data(:, VinsmonoConstants.q_BC(3)), '-b');

xlabel('time[sec]');
ylabel('quaternion[1]');
legend('qx', 'qy', 'qz');
grid on;
outputfig = [output_dir, '/q_BC.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
plot(data(:,1), data(:, VinsmonoConstants.td(1))*1000, '-r');
hold on;
xlabel('time[sec]');
ylabel('t_d[millisec]');
grid on;
outputfig = [output_dir, '/td.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

if ~exist('avg_since_start','var')
    avg_since_start = input(['To compute average of estimated parameters,', ...
        ' specify num of secs since start:']);
end
if ~exist('avg_trim_end','var')
    avg_trim_end = input(['To compute average of estimated parameters,', ...
        ' specify num of secs to trim from the end:']);
end

startTime = data(1, 1);
endTime = data(end, 1);
sticky_time_range = 1; % sec
endIndex = find(abs(endTime - avg_trim_end - data(:,1)) <...
    sticky_time_range, 1);
if(isempty(endIndex))
    endIndex = size(data, 1);
end

startIndex = find(abs(startTime + avg_since_start - data(:,1)) ...
    < sticky_time_range, 1);
if(isempty(startIndex))
    disp(['unable to locate start index at time since start ', num2str(startTime)]);
    return;
end

log_file = [output_dir, '/avg_estimates.txt'];
fileID = fopen(log_file, 'w');
estimate_average = mean(data(startIndex:endIndex, :), 1);
fprintf(fileID, ['average over [', ...
    num2str(data(startIndex, 1)), ...  
    ',', num2str(data(endIndex, 1)), ...
    '] secs, i.e., (', num2str(avg_since_start), ...
    ' s since start and ', num2str(avg_trim_end), ...
    ' s back from the end).\n' ]);
fprintf(fileID, 'p_{BC}[cm]:');
fprintf(fileID, '%.3f ', estimate_average(VinsmonoConstants.p_BC)*100);
fprintf(fileID, ['\nt_d[millisec]:']);
fprintf(fileID, '%.3f ', estimate_average(VinsmonoConstants.td) * 1000);

estimate_average = mean(vinsmono_data(startIndex:endIndex, :), 1);
fprintf(fileID, ['\nb_g[' char(176) '/s]:']);
fprintf(fileID, '%.4f ', estimate_average(VinsmonoConstants.b_g) * 180/pi);
fprintf(fileID, '\nb_a[m/s^2]:');
fprintf(fileID, '%.4f ', estimate_average(VinsmonoConstants.b_a));

if fileID ~= 1
    fclose(fileID);
    fprintf('The average estimates are saved at \n%s.\n', log_file);
end
