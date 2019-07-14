function averageMsckf2VariableEstimates(msckf2_data)
% The msckf2 result has a format described in Msckf2Constants.m

startTime = msckf2_data(1, 1);
endTime = msckf2_data(end, 1);
sec_to_nanos = 1e9;
sticky_time_range = 2.5e7; % nanosec
avg_since_start = input(['To compute average of estimated parameters,', ...
    ' specify num of secs since start:']);
avg_trim_end = input(['To compute average of estimated parameters,', ...
    ' specify num of secs to trim from the end:']);

endIndex = find(abs(endTime - avg_trim_end*sec_to_nanos - msckf2_data(:,1)) <...
    sticky_time_range, 1);
if(isempty(endIndex))
    endIndex = size(msckf2_data, 1);
end

startIndex = find(abs(startTime + avg_since_start*sec_to_nanos - msckf2_data(:,1)) ...
    < sticky_time_range, 1);
if(isempty(startIndex))
    return;
end

estimate_average = mean(msckf2_data(startIndex:endIndex, :), 1);
fprintf(['average over [', ...
    num2str(msckf2_data(startIndex, 1) / sec_to_nanos), ...  
    ',', num2str(msckf2_data(endIndex, 1) / sec_to_nanos), ...
    '] secs.\n' ]);
fprintf(['b_g[' char(176) '/s]:']);
fprintf('%.4f,', estimate_average(Msckf2Constants.b_g) * 180/pi);
fprintf('\nb_a[m/s^2]:');
fprintf('%.4f,', estimate_average(Msckf2Constants.b_a));
fprintf('\np_{BC}[cm]:');
fprintf('%.3f,', estimate_average(Msckf2Constants.p_BC)*100);
fprintf('\nfx fy cx cy[px]:');
fprintf('%.3f,', estimate_average(Msckf2Constants.fxy_cxy));
fprintf('\nk1 k2[1]:');
fprintf('%.3f,', estimate_average(Msckf2Constants.k1_k2));
fprintf('\np1 p2[1]:');
fprintf('%.6f,', estimate_average(Msckf2Constants.p1_p2));
fprintf('\ntd[ms]:');
fprintf('%.3f\n', estimate_average(Msckf2Constants.td)*1e3);
fprintf('tr[ms]:');
fprintf('%.3f\n', estimate_average(Msckf2Constants.tr)*1e3);
end