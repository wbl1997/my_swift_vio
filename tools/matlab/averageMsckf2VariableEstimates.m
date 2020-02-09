function averageMsckf2VariableEstimates(msckf2_data, msckf_index_server, ...
    log_file, avg_since_start, avg_trim_end)
if ~exist('log_file','var')
    fileID = 1;
else
    fileID = fopen(log_file, 'w');    
end
startTime = msckf2_data(1, 1);
endTime = msckf2_data(end, 1);
sec_to_nanos = 1e9;
sticky_time_range = 1e9; % nanosec
if ~exist('avg_since_start','var')
    avg_since_start = input(['To compute average of estimated parameters,', ...
        ' specify num of secs since start:']);
end
if ~exist('avg_trim_end','var')
    avg_trim_end = input(['To compute average of estimated parameters,', ...
        ' specify num of secs to trim from the end:']);
end

endIndex = find(abs(endTime - avg_trim_end*sec_to_nanos - msckf2_data(:,1)) <...
    sticky_time_range, 1);
if(isempty(endIndex))
    endIndex = size(msckf2_data, 1);
end

startIndex = find(abs(startTime + avg_since_start*sec_to_nanos - msckf2_data(:,1)) ...
    < sticky_time_range, 1);
if(isempty(startIndex))
    disp(['unable to locate start index at time since start ', num2str(startTime)]);
    return;
end

estimate_average = mean(msckf2_data(startIndex:endIndex, :), 1);
fprintf(fileID, ['average over [', ...
    num2str(msckf2_data(startIndex, 1) / sec_to_nanos), ...  
    ',', num2str(msckf2_data(endIndex, 1) / sec_to_nanos), ...
    '] secs, i.e., (', num2str(avg_since_start), ...
    ' s since start and ', num2str(avg_trim_end), ...
    ' s back from the end).\n' ]);
fprintf(fileID, ['b_g[' char(176) '/s]:']);
fprintf(fileID, '%.4f ', estimate_average(msckf_index_server.b_g) * 180/pi);
fprintf(fileID, '+/- ');
fprintf(fileID, '%.5f ', estimate_average(msckf_index_server.b_g_std) * 180/pi);
fprintf(fileID, '\nb_a[m/s^2]:');
fprintf(fileID, '%.4f ', estimate_average(msckf_index_server.b_a));

fprintf(fileID, '+/- ');
fprintf(fileID, '%.5f ', estimate_average(msckf_index_server.b_a_std));
fprintf(fileID, '\np_{BC}[cm]:');
fprintf(fileID, '%.3f ', estimate_average(msckf_index_server.p_BC(1:3))*100);
if size(msckf_index_server.p_BC, 2) > 3
fprintf(fileID, '%.3f ', estimate_average(msckf_index_server.p_BC(4:end)));
end
if ~isempty(msckf_index_server.p_BC_std)
fprintf(fileID, '+/- ');
fprintf(fileID, '%.4f ', estimate_average(msckf_index_server.p_BC_std)*100);
end

fprintf(fileID, '\nfx fy cx cy[px]:');
fprintf(fileID, '%.3f ', estimate_average(msckf_index_server.fxy_cxy));
if ~isempty(msckf_index_server.fxy_cxy_std)
fprintf(fileID, '+/- ');
fprintf(fileID, '%.4f ', estimate_average(msckf_index_server.fxy_cxy_std));
end
fprintf(fileID, '\nk1 k2[1]:');
fprintf(fileID, '%.3f ', estimate_average(msckf_index_server.k1_k2));
if ~isempty(msckf_index_server.k1_k2_std)
fprintf(fileID, '+/- ');
fprintf(fileID, '%.4f ', estimate_average(msckf_index_server.k1_k2_std));
end
fprintf(fileID, '\np1 p2[1]:');
fprintf(fileID, '%.6f ', estimate_average(msckf_index_server.p1_p2));
if ~isempty(msckf_index_server.p1_p2_std)
fprintf(fileID, '+/- ');
fprintf(fileID, '%.6f ', estimate_average(msckf_index_server.p1_p2_std));
end
fprintf(fileID, '\ntd[ms]:');
fprintf(fileID, '%.3f ', estimate_average(msckf_index_server.td)*1e3);
fprintf(fileID, '+/- ');
fprintf(fileID, '%.3f\n', estimate_average(msckf_index_server.td_std)*1e3);
fprintf(fileID, 'tr[ms]:');
fprintf(fileID, '%.3f ', estimate_average(msckf_index_server.tr)*1e3);
fprintf(fileID, '+/- ');
fprintf(fileID, '%.3f\n', estimate_average(msckf_index_server.tr_std)*1e3);
if fileID ~= 1
    fclose(fileID);
    fprintf('The average estimates are saved at \n%s.\n', log_file);
end
end
