function draw_all_rmse_iros2020(output_dir, index)
% draw the rmse curves for all estimators
% output_dir the dir of the simulation log dir
% index which index of RMSE to draw, 2,3,4, x, y, z
% 5,6,7, roll, pitch, yaw
% -1, sqrt(x^2 + y^2 + z^2)
% -2, sqrt(roll^2 + pitch^2 + yaw^2)
% to generate the IROS202O figures
% draw_rmse_iros2020('/media/jhuai/Seagate/jhuai/RSS-2020/results/msckf_simulation', -1)
% draw_rmse_iros2020('/media/jhuai/Seagate/jhuai/RSS-2020/results/msckf_simulation', -2)

background_color='None'; % set to None for paper.
motion_list = {'WavyCircle', 'Squircle', 'Dot', 'Motionless'};
estimator_list = {'OKVIS_Proj_Euc_F', 'MSCKF_Proj_Idp_F', 'MSCKF_Epi_Proj_Idp_F', 'TFVIO_Epi_Euc_F'};
label_list={'OKVIS', 'MSCKF', 'Epi-MSCKF', 'TFVIO'};
radtodeg = 180 / pi;

close all;
start_time = 20;
for motion_id = 1:length(motion_list)
    rmse_list = cell(length(estimator_list), 1);
    figure;
    for estimator_id = 1:length(estimator_list)
        rmse_list{estimator_id} = [output_dir, '/', ...
            estimator_list{estimator_id}, '_',  motion_list{motion_id}, '_RMSE.txt'];
        rmse = readmatrix(rmse_list{estimator_id}, 'NumHeaderLines', 1);
        if (index > 0)
            if (index >= 5) 
                 rmse_interest = rmse(:, index) * radtodeg;
                  else 
            rmse_interest = rmse(:, index);
            
            end
        else
            switch (index)
                case -1
                    rmse_interest = sqrt(sum(rmse(:, 2:4).^2, 2));
                case -2
                    rmse_interest = sqrt(sum(rmse(:, 5:7).^2, 2)) * radtodeg;
                otherwise
                    disp(['Unsupported index ', num2str(index)]);
            end
        end
        plot(rmse(:, 1) - start_time, rmse_interest); hold on;
        if max(rmse_interest > 100)
            ylim([0, 100]);
        end
    end
    xlabel('time (sec)');
    if (index >= 5 || index == -2)
        ylabel('RMSE_{yaw} (^{\circ})');
    else
        ylabel('RMSE_{xyz} (m)');
    end
    titlestr = [motion_list{motion_id} '_rmse_' num2str(index)];
    grid on;
    legend(label_list, 'Location', 'northwest');
    set(gcf, 'Color', background_color);
    outputfig = [output_dir, '/', titlestr, '.eps'];
    if exist(outputfig, 'file')==2
        delete(outputfig);
    end
    export_fig(outputfig);
end
end

