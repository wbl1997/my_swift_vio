function intermediatePlotter(data, indexServer, outputPath)
% It will modify data by subtracting reference values from data
if (nargin <= 1) 
    outputPath = '';
end

figure;
data(:, indexServer.T_g_diag) = data(:, indexServer.T_g_diag) ...
    - ones(size(data, 1), 3);
drawMeanAndStdBound(data, indexServer.T_g, ...
    indexServer.T_g_std);
ylabel('$\mathbf{T}_g$[1]' , 'Interpreter', 'Latex');
outputfig = [outputPath, '/T_g.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
drawMeanAndStdBound(data, indexServer.T_s, ...
    indexServer.T_s_std);
ylabel('$\mathbf{T}_s$[1]' , 'Interpreter', 'Latex');
outputfig = [outputPath, '/T_s.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
data(:, indexServer.T_a_diag) = data(:, indexServer.T_a_diag) ...
    - ones(size(data, 1), 3);
drawMeanAndStdBound(data, indexServer.T_a, ...
    indexServer.T_a_std);
ylabel('$\mathbf{T}_a$[1]' , 'Interpreter', 'Latex');
outputfig = [outputPath, '/T_a.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
drawMeanAndStdBound(data, indexServer.td, ...
    indexServer.td_std, 1000);
legend('t_d', '3\sigma_t_d');
ylabel('$t_d$[ms]', 'Interpreter', 'Latex');
outputfig = [outputPath, '/t_d.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
drawMeanAndStdBound(data, indexServer.tr, ...
    indexServer.tr_std, 1000);
legend('t_r', '3\sigma_{t_r}');
ylabel('$t_r$[ms]', 'Interpreter', 'Latex');
outputfig = [outputPath, '/t_r.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);
end
