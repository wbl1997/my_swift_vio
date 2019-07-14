function intermediatePlotter(data, outputPath)
% It will modify data by subtracting reference values from data
if (nargin <= 1) 
    outputPath = '';
end

figure;
data(:, Msckf2Constants.T_g_diag) = data(:, Msckf2Constants.T_g_diag) ...
    - ones(size(data, 1), 3);
draw_ekf_triplet_with_std(data, Msckf2Constants.T_g, ...
    Msckf2Constants.T_g_std);
ylabel('$\mathbf{T}_g$[1]' , 'Interpreter', 'Latex');
outputfig = [outputPath, '/T_g.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
draw_ekf_triplet_with_std(data, Msckf2Constants.T_s, ...
    Msckf2Constants.T_s_std);
ylabel('$\mathbf{T}_s$[1]' , 'Interpreter', 'Latex');
outputfig = [outputPath, '/T_s.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
data(:, Msckf2Constants.T_a_diag) = data(:, Msckf2Constants.T_a_diag) ...
    - ones(size(data, 1), 3);
draw_ekf_triplet_with_std(data, Msckf2Constants.T_a, ...
    Msckf2Constants.T_a_std);
ylabel('$\mathbf{T}_a$[1]' , 'Interpreter', 'Latex');
outputfig = [outputPath, '/T_a.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
draw_ekf_single_with_std(data, Msckf2Constants.td, ...
    Msckf2Constants.td_std, 1000);
legend('t_d', '3\sigma_t_d');
ylabel('$t_d$[ms]', 'Interpreter', 'Latex');
outputfig = [outputPath, '/t_d.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);

figure;
draw_ekf_single_with_std(data, Msckf2Constants.tr, ...
    Msckf2Constants.tr_std, 1000);
legend('t_r', '3\sigma_{t_r}');
ylabel('$t_r$[ms]', 'Interpreter', 'Latex');
outputfig = [outputPath, '/t_r.eps'];
if exist(outputfig, 'file')==2
  delete(outputfig);
end
export_fig(outputfig);
end
