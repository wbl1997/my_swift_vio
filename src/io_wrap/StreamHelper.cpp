#include "io_wrap/StreamHelper.hpp"
#include "msckf/CameraRig.hpp"
#include "msckf/ExtrinsicModels.hpp"
#include "msckf/ImuModels.hpp"
#include "msckf/ProjParamOptModels.hpp"

DEFINE_string(datafile_separator, ",",
              "the separator used for a ASCII output file");

namespace okvis {

void StreamHelper::composeHeaderLine(const std::string &imu_model,
                                     const std::string &cam0_proj_opt_rep,
                                     const std::string &cam0_extrinsic_opt_rep,
                                     const std::string &cam0_distortion_rep,
                                     DUMP_RESULT_OPTION result_option,
                                     std::string *header_line, bool include_frameid) {
  composeHeaderLine(imu_model, ProjectionOptNameToId(cam0_proj_opt_rep),
                    ExtrinsicModelNameToId(cam0_extrinsic_opt_rep),
                    cam0_distortion_rep, result_option, header_line, include_frameid);
}

void StreamHelper::composeHeaderLine(const std::string &imu_model,
                                     const int &cam0_proj_opt_mode,
                                     const int &cam0_extrinsic_opt_mode,
                                     const std::string &cam0_distortion_rep,
                                     DUMP_RESULT_OPTION result_option,
                                     std::string *header_line,
                                     bool include_frameid) {
  std::stringstream stream;
  stream << "%timestamp" << FLAGS_datafile_separator
         << (include_frameid ? "frameIdInSource" + FLAGS_datafile_separator : "")
         << "p_WS_W_x" << FLAGS_datafile_separator
         << "p_WS_W_y" << FLAGS_datafile_separator << "p_WS_W_z"
         << FLAGS_datafile_separator << "q_WS_x" << FLAGS_datafile_separator
         << "q_WS_y" << FLAGS_datafile_separator << "q_WS_z"
         << FLAGS_datafile_separator << "q_WS_w" << FLAGS_datafile_separator
         << "v_WS_W_x" << FLAGS_datafile_separator << "v_WS_W_y"
         << FLAGS_datafile_separator << "v_WS_W_z";
  std::string imu_param_format;
  ImuModelToFormatString(ImuModelNameToId(imu_model), FLAGS_datafile_separator,
                         &imu_param_format);
  std::string cam_proj_intrinsic_format;
  ProjectionOptToParamsInfo(cam0_proj_opt_mode, FLAGS_datafile_separator,
                            &cam_proj_intrinsic_format);
  std::string cam_extrinsic_format;
  ExtrinsicModelToParamsInfo(cam0_extrinsic_opt_mode, FLAGS_datafile_separator,
                             &cam_extrinsic_format);
  std::string cam_distortion_format;
  okvis::cameras::DistortionNameToParamsInfo(
      cam0_distortion_rep, FLAGS_datafile_separator, &cam_distortion_format);

  if (result_option == FULL_STATE_WITH_ALL_CALIBRATION) {
    stream << FLAGS_datafile_separator << imu_param_format;
    stream << FLAGS_datafile_separator << cam_extrinsic_format;
    stream << (cam_extrinsic_format.empty() ? "" : FLAGS_datafile_separator)
           << cam_proj_intrinsic_format;
    stream << (cam_proj_intrinsic_format.empty() ? "" : FLAGS_datafile_separator)
           << cam_distortion_format;
    stream << FLAGS_datafile_separator << "td[s]" << FLAGS_datafile_separator
           << "tr[s]";
  } else if (result_option == FULL_STATE_WITH_EXTRINSICS) {
    stream << FLAGS_datafile_separator << imu_param_format;
    stream << FLAGS_datafile_separator << cam_extrinsic_format;
  } else {
    std::string imu_param_format;
    ImuModelToFormatString(ImuModelNameToId("BG_BA"), FLAGS_datafile_separator,
                           &imu_param_format);
    stream << FLAGS_datafile_separator << imu_param_format;
  }
  *header_line = stream.str();
}

}  // namespace okvis
