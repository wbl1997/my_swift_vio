function drawRpgTrajEvalResults(rootfolder, method, displacementTol)
% draw the trajectories estimated by a method and check the quality by 
% a threshold on final translation.

% e.g.,
% rootfolder='tumvi_derby/raw_swiftvio1/vio';
% method='KSWF_fix_all';
% displacementTol = 20;

close all;
numfigures=0;
success = 0;
missing = 0;
trials = 5;
sessions = {'room1', 'room2', 'room3', 'room4', 'room5', 'room6'};
numsessions = length(sessions);


for r = 1:numsessions
    session = sessions{r};
    folder=[rootfolder, '/laptop/', method, '/laptop_', method, '_', session];
    gtdata = readmatrix([folder, '/stamped_groundtruth.txt'], 'NumHeaderLines', 1);
    ratio = 1.0;
    for index = 0:trials - 1
        files = {
            [folder, '/stamped_traj_estimate', num2str(index), '.txt'],
            [folder, '/stamped_groundtruth.txt']};
        if isfile(files{1})
            data = readmatrix(files{1}, 'NumHeaderLines', 1);
            figure;
            count = floor(size(gtdata, 1) * ratio);
            plot3(gtdata(1:count, 2), gtdata(1:count, 3), gtdata(1:count, 4), 'g');
            hold on; axis equal;
            xlabel('x'); ylabel('y'); zlabel('z');
            count = floor(size(data, 1) * ratio);
            plot3(data(1:count, 2), data(1:count, 3), data(1:count, 4), 'r');
            hold off;
            
            distance = sqrt(data(count, 3:5) * data(count, 3:5)');
            if distance < displacementTol
                success = success + 1;
            end
            numfigures = numfigures + 1;
        else
            missing = missing + 1;
            warning('missing %s\n', files{1});
        end        
    end
end
fprintf('Success %d, aborted %d, out of %d trials.\n', success, missing, trials * numsessions);
end
