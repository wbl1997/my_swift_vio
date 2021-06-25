function options = SwiftVioOptions(isfilter, numCameras, stdCamIdx)
% set variable and standard deviation indices in the swift_vio output.
% isfilter: is the output from a filter or a smoother?
% numCameras: how many cameras are in the setup?
% stdCamIdx: to show which camera (1-based index), as we only support
% drawing uncertainties for one camera.
%
% Note this method assumes full calibration, and does not support reduced
% camera intrinsic models.

if nargin < 3
    stdCamIdx = 1;
end
if nargin < 2
    numCameras = 1;
end
projectionIntrinsicDim = 3;
distortionIntrinsicDim = 4;
if isfilter
    if numCameras == 1
        variableDimList = [3, 4, 3, 3, 3, 9, 9, 9, 3, projectionIntrinsicDim, distortionIntrinsicDim, 2];
    else
        variableDimList = [3, 4, 3, 3, 3, 9, 9, 9, 3, projectionIntrinsicDim, ...
            distortionIntrinsicDim, 2, 3, 4, projectionIntrinsicDim, distortionIntrinsicDim, 2];
    end
else
    variableDimList = [3, 4, 3, 3, 3];
end

options.export_fig_path = '';
options.avg_since_start = 10;
options.avg_trim_end = 10;
options.trueCameraIntrinsics = zeros(1, projectionIntrinsicDim);


padding = 2;
options.r = padding + (1:3);
options.q = padding + (4:7);
options.v = padding + (8:10);
options.b_g = padding + (11:13);
options.b_a = padding + (14:16);

options.std_start_index = padding + sum(variableDimList);
options.r_std = options.std_start_index + (1:3);
options.q_std = options.std_start_index + (4:6);
options.v_std = options.std_start_index + (7:9);
options.b_g_std = options.std_start_index + (10:12);
options.b_a_std = options.std_start_index + (13:15);

switch isfilter
    case 1
        options.T_g = 19:27;
        options.T_g_diag = [19, 23, 27];
        options.T_s = 28:36;
        options.T_s_diag = [28, 32, 36];
        options.T_a = 37:45;
        options.T_a_diag = [37, 41, 45];

        options.std_Tg_start_index = options.std_start_index + 15;
        options.T_g_std = options.std_Tg_start_index + (1:9);
        options.T_s_std = options.std_Tg_start_index + (10:18);
        options.T_a_std = options.std_Tg_start_index + (19:27);
        
        % parameters for the first camera.
            options.p_camera = options.T_a_diag(end) + (1:3); % a camera's extrinsic parameters.
            options.fxy_cxy = options.p_camera(end) + (1:projectionIntrinsicDim);
            options.k1_k2 = options.fxy_cxy(end) + (1:2);
            if distortionIntrinsicDim <= 2
                options.p1_p2 = options.k1_k2(end);
            else
                options.p1_p2 = options.k1_k2(end) + (1:distortionIntrinsicDim - 2);
            end
            options.td = options.p1_p2(end) + 1;
            options.tr = options.td + 1;
            options.std_camera_start_index = options.std_start_index + 15 + 27;
            options.p_camera_std = options.std_camera_start_index + (1:3);
            options.fxy_cxy_std = options.p_camera_std(end) + (1:projectionIntrinsicDim);
            options.k1_k2_std = options.fxy_cxy_std(end) + (1:2);
            if distortionIntrinsicDim > 2
                options.p1_p2_std = options.k1_k2_std(end) + (1:distortionIntrinsicDim-2);
            else
                options.p1_p2_std = options.k1_k2_std(end);
            end
            options.td_std = options.p1_p2_std(end) + 1;
            options.tr_std = options.td_std(end) + 1;
            % parameters for the second camera.
        if stdCamIdx == 2
            options.p_camera = options.tr + (1:3);
            options.q_XC = options.p_camera(end) + (1:4);
            options.fxy_cxy = options.q_XC(end) + (1:projectionIntrinsicDim);
            options.k1_k2 = options.fxy_cxy(end) + (1:2);
            if distortionIntrinsicDim > 2
            options.p1_p2 = options.k1_k2(end) + (1:distortionIntrinsicDim-2);
            else
                options.p1_p2 = options.k1_k2(end);
            end
            options.td = options.p1_p2(end) + 1;
            options.tr = options.td + 1;
            options.std_camera_start_index = options.tr_std + 3 + 3 + projectionIntrinsicDim + distortionIntrinsicDim + 2;
            options.p_camera_std = options.std_camera_start_index + (1:3);
            options.q_XC_std = options.p_camera_std(end) + (1:3);
            options.fxy_cxy_std = options.q_XC_std(end) + (1:projectionIntrinsicDim);
            options.k1_k2_std = options.fxy_cxy_std(end) + (1:2);
            if distortionIntrinsicDim > 2
                options.p1_p2_std = options.k1_k2_std(end) + (1:distortionIntrinsicDim - 2);
            else
                options.p1_p2_std = options.k1_k2_std(end);
            end
            options.td_std = options.p1_p2_std(end) + 1;
            options.tr_std = options.td_std + 1;
        end
        
    case 0
        options.T_g = [];
        options.T_g_diag = [];
        options.T_s = [];
        options.T_s_diag = [];
        options.T_a = [];
        options.T_a_diag = [];
        
        options.p_camera = [];
        options.fxy_cxy = [];
        options.k1_k2 = [];
        options.p1_p2 = [];
        options.td = [];
        options.tr = [];
        
end
end
