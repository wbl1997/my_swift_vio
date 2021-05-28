function options = SwiftVioOptions(isfilter)
switch isfilter
    case 1
        options.export_fig_path = '';
        options.avg_since_start = 10;
        options.avg_trim_end = 10;
        options.misalignment_dim = 27;
        options.extrinsic_dim = 3;
        options.project_intrinsic_dim = 4;
        options.distort_intrinsic_dim = 4;
        options.fix_extrinsic = 0;
        options.fix_intrinsic = 0;
        options.td_dim = 1;
        options.tr_dim = 1;
        options.trueCameraIntrinsics = [0, 0, 0, 0];
    case 0
        options.export_fig_path = '';
        options.avg_since_start = 10;
        options.avg_trim_end = 10;
        options.misalignment_dim = 0;
        options.extrinsic_dim = 0;
        options.project_intrinsic_dim = 0;
        options.distort_intrinsic_dim = 0;
        options.fix_extrinsic = 1;
        options.fix_intrinsic = 1;
        options.td_dim = 0;
        options.tr_dim = 0;
        options.trueCameraIntrinsics = [0, 0, 0, 0];
end
end
