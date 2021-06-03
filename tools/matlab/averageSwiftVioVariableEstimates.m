function averageSwiftVioVariableEstimates(vioData, indexServer, ...
    logFile, meanSince, meanExceptEnd)
if ~exist('logFile','var')
    fileID = 1;
else
    fileID = fopen(logFile, 'w');    
end
startTime = vioData(1, 1);
endTime = vioData(end, 1);
sec_to_nanos = 1e9;
sticky_time_range = 1e9; % nanosec
if ~exist('meanSince','var')
    meanSince = input(['To compute average of estimated parameters,', ...
        ' specify num of secs since start:']);
end
if ~exist('meanExceptEnd','var')
    meanExceptEnd = input(['To compute average of estimated parameters,', ...
        ' specify num of secs to trim from the end:']);
end

endIndex = find(abs(endTime - meanExceptEnd*sec_to_nanos - vioData(:,1)) <...
    sticky_time_range, 1);
if(isempty(endIndex))
    endIndex = size(vioData, 1);
end

startIndex = find(abs(startTime + meanSince*sec_to_nanos - vioData(:,1)) ...
    < sticky_time_range, 1);
if(isempty(startIndex))
    disp(['Unable to locate start index at time since start ', num2str(startTime)]);
    disp('Start from the first entry instead.');
    startIndex = 1;
end

estimate_average = mean(vioData(startIndex:endIndex, :), 1);
fprintf(fileID, ['average over [', ...
    num2str(vioData(startIndex, 1) / sec_to_nanos), ...  
    ',', num2str(vioData(endIndex, 1) / sec_to_nanos), ...
    '] secs, i.e., (', num2str(meanSince), ...
    ' s since start and ', num2str(meanExceptEnd), ...
    ' s back from the end).\n' ]);
fprintf(fileID, ['b_g[' char(176) '/s]:']);
fprintf(fileID, '%.4f ', estimate_average(indexServer.b_g) * 180/pi);
fprintf(fileID, '+/- ');
fprintf(fileID, '%.5f ', estimate_average(indexServer.b_g_std) * 180/pi);
fprintf(fileID, '\nb_a[m/s^2]:');
fprintf(fileID, '%.4f ', estimate_average(indexServer.b_a));

fprintf(fileID, '+/- ');
fprintf(fileID, '%.5f ', estimate_average(indexServer.b_a_std));

if isempty(indexServer.p_camera)
    return;
end

fprintf(fileID, '\np_{camera}[cm]:');
fprintf(fileID, '%.3f ', estimate_average(indexServer.p_camera(1:3))*100);
if size(indexServer.p_camera, 2) > 3
fprintf(fileID, '%.3f ', estimate_average(indexServer.p_camera(4:end)));
end
if ~isempty(indexServer.p_camera_std)
fprintf(fileID, '+/- ');
fprintf(fileID, '%.4f ', estimate_average(indexServer.p_camera_std)*100);
end

fprintf(fileID, '\nfx fy cx cy[px]:');
fprintf(fileID, '%.3f ', estimate_average(indexServer.fxy_cxy));
if ~isempty(indexServer.fxy_cxy_std)
fprintf(fileID, '+/- ');
fprintf(fileID, '%.4f ', estimate_average(indexServer.fxy_cxy_std));
end
fprintf(fileID, '\nk1 k2[1]:');
fprintf(fileID, '%.3f ', estimate_average(indexServer.k1_k2));
if ~isempty(indexServer.k1_k2_std)
fprintf(fileID, '+/- ');
fprintf(fileID, '%.4f ', estimate_average(indexServer.k1_k2_std));
end
fprintf(fileID, '\np1 p2[1]:');
fprintf(fileID, '%.6f ', estimate_average(indexServer.p1_p2));
if ~isempty(indexServer.p1_p2_std)
fprintf(fileID, '+/- ');
fprintf(fileID, '%.6f ', estimate_average(indexServer.p1_p2_std));
end
fprintf(fileID, '\ntd[ms]:');
fprintf(fileID, '%.3f ', estimate_average(indexServer.td)*1e3);
fprintf(fileID, '+/- ');
fprintf(fileID, '%.3f\n', estimate_average(indexServer.td_std)*1e3);
fprintf(fileID, 'tr[ms]:');
fprintf(fileID, '%.3f ', estimate_average(indexServer.tr)*1e3);
fprintf(fileID, '+/- ');
fprintf(fileID, '%.3f\n', estimate_average(indexServer.tr_std)*1e3);
if fileID ~= 1
    fclose(fileID);
    fprintf('The average estimates are saved at \n%s.\n', logFile);
end
end
