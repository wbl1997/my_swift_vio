#!/usr/bin/env bash
/usr/local/MATLAB/R2019b/bin/matlab -nodesktop -nosplash -r \
  "cd('$SWIFT_VIO_WS/src/swift_vio/tools/matlab'); \
    vio_csv = '$VIO_OUTPUT_DIR/swift_vio.csv'; \
    export_fig_path = '/media/jhuai/Seagate/jhuai/tools/export_fig/'; \
    voicebox_path = '/media/jhuai/Seagate/jhuai/tools/voicebox'; \
    output_dir = '$VIO_OUTPUT_DIR'; \
    cmp_data_file = ''; \
    gt_file = ''; \
    avg_since_start = 10; \
    avg_trim_end = 10; \
    misalignment_dim = 27; \
    extrinsic_dim = 7; \
    project_intrinsic_dim = 4; \
    distort_intrinsic_dim = 4; \
    fix_extrinsic = 1; \
    fix_intrinsic = 1; \
    try \
      plotSwiftVioResult(vio_csv, export_fig_path, voicebox_path, output_dir, \
        cmp_data_file, gt_file, avg_since_start, avg_trim_end, \
        misalignment_dim, extrinsic_dim, project_intrinsic_dim, \
        distort_intrinsic_dim, fix_extrinsic, fix_intrinsic); \
    catch e; \
      msgText = getReport(e); \
      disp(['Error occurs: ' msgText]); \
    end; \
    quit";