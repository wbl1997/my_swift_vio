function annotate_line_fitting_iros2020(est_file)
% the below location parameters are suitable only with
% /hailun_house/halfway_pan_init/2020_02_23_12_22_56/rel_msckf_epi_3.0/msckf_estimates.csv
% which is used in IROS 2020 paper.

if ~contains(est_file, ['/hailun_house/halfway_pan_init/', ...
        '2020_02_23_12_22_56/rel_msckf_epi_3.0/msckf_estimates.csv'])
    return;
end
x = [0.45 0.5];
y = [0.55 0.49];
ta = annotation('textarrow',x,y,'interpreter','latex','String','$\hat{L}$');
ta.HeadWidth = 5;
dim = [.71 .53 .2 .1];
annotation('textbox',dim,'interpreter','latex', 'String', ...
    '$\Delta_o$', 'FitBoxToText','on', 'EdgeColor', 'none');
dim = [.38 .375 .2 .1];
annotation('textbox',dim,'interpreter','latex', 'String', ...
    '$\Delta$', 'FitBoxToText','on', 'EdgeColor', 'none');
dim = [.42 .42 .2 .1];
annotation('textbox',dim,'interpreter','latex', 'String', ...
    '$\Delta$', 'FitBoxToText','on', 'EdgeColor', 'none');
end