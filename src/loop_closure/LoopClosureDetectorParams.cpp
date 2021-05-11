/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   LoopClosureDetectorParams.cpp
 * @brief
 * @author Antoni Rosinol
 */

#include "loop_closure/LoopClosureDetectorParams.h"
#include "io_wrap/YamlParser.h"

namespace swift_vio {
LoopClosureDetectorParams::LoopClosureDetectorParams(
    LoopClosureMethodType lc_method_id,
    bool use_nss,
    float alpha,
    int min_temporal_matches,
    int dist_local,
    int max_db_results,
    float min_nss_factor,
    int min_matches_per_group,
    int max_intragroup_gap,
    int max_distance_between_groups,
    int max_distance_between_queries,

    int min_correspondences,

    int max_ransac_iterations_stereo,
    double ransac_threshold_stereo,
    int relative_pose_opt_iterations,
    double ransac_inlier_threshold_stereo,
    double relative_pose_info_damper,

    double lowe_ratio,
    int matcher_type,

    int nfeatures,
    float scale_factor,
    int nlevels,
    int edge_threshold,
    int first_level,
    int WTA_K,
    int score_type,
    int patch_sze,
    int fast_threshold,

    bool pgo_uniform_weight,
    double pgo_rot_threshold,
    double pgo_trans_threshold)
    : LoopClosureParameters("Loop Closure Parameters"),
      loop_closure_method_(lc_method_id),
      use_nss_(use_nss),
      alpha_(alpha),
      min_temporal_matches_(min_temporal_matches),
      dist_local_(dist_local),
      max_db_results_(max_db_results),
      min_nss_factor_(min_nss_factor),
      min_matches_per_group_(min_matches_per_group),
      max_intragroup_gap_(max_intragroup_gap),
      max_distance_between_groups_(max_distance_between_groups),
      max_distance_between_queries_(max_distance_between_queries),

      min_correspondences_(min_correspondences),

      max_ransac_iterations_stereo_(max_ransac_iterations_stereo),
      ransac_threshold_stereo_(ransac_threshold_stereo),
      relative_pose_opt_iterations_(relative_pose_opt_iterations),
      ransac_inlier_threshold_stereo_(ransac_inlier_threshold_stereo),
      relative_pose_info_damper_(relative_pose_info_damper),

      lowe_ratio_(lowe_ratio),
      matcher_type_(matcher_type),

      nfeatures_(nfeatures),
      scale_factor_(scale_factor),
      nlevels_(nlevels),
      edge_threshold_(edge_threshold),
      first_level_(first_level),
      WTA_K_(WTA_K),
      score_type_(score_type),
      patch_sze_(patch_sze),
      fast_threshold_(fast_threshold),

      pgo_uniform_weight_(pgo_uniform_weight),
      pgo_rot_threshold_(pgo_rot_threshold),
      pgo_trans_threshold_(pgo_trans_threshold) {
  // Trivial sanity checks:
  CHECK(alpha_ > 0);
  CHECK(nfeatures_ >= 100);  // TODO(marcus): add more checks, change this one
}

bool LoopClosureDetectorParams::parseYAML(const std::string& filepath) {
  YamlParser yaml_parser(filepath);

  int lc_method_id;
  yaml_parser.getYamlParam("loop_closure_method", &lc_method_id);
  switch (lc_method_id) {
    case 1:
      loop_closure_method_ = LoopClosureMethodType::OrbBoW;
      break;
    case 0:
    default:
      loop_closure_method_ = LoopClosureMethodType::Mock;
      break;
  }

  yaml_parser.getYamlParam("use_nss", &use_nss_);
  yaml_parser.getYamlParam("alpha", &alpha_);
  yaml_parser.getYamlParam("min_temporal_matches", &min_temporal_matches_);
  yaml_parser.getYamlParam("dist_local", &dist_local_);
  yaml_parser.getYamlParam("max_db_results", &max_db_results_);
  yaml_parser.getYamlParam("min_nss_factor", &min_nss_factor_);
  yaml_parser.getYamlParam("min_matches_per_group", &min_matches_per_group_);
  yaml_parser.getYamlParam("max_intragroup_gap", &max_intragroup_gap_);
  yaml_parser.getYamlParam("max_distance_between_groups",
                           &max_distance_between_groups_);
  yaml_parser.getYamlParam("max_distance_between_queries",
                           &max_distance_between_queries_);

  yaml_parser.getYamlParam("min_correspondences", &min_correspondences_);

  yaml_parser.getYamlParam("max_ransac_iterations_stereo",
                           &max_ransac_iterations_stereo_);
  yaml_parser.getYamlParam("ransac_threshold_stereo",
                           &ransac_threshold_stereo_);
  yaml_parser.getYamlParam("relative_pose_opt_iterations",
                           &relative_pose_opt_iterations_);
  yaml_parser.getYamlParam("ransac_inlier_threshold_stereo",
                           &ransac_inlier_threshold_stereo_);
  yaml_parser.getYamlParam("relative_pose_info_damper",
                           &relative_pose_info_damper_);
  yaml_parser.getYamlParam("lowe_ratio", &lowe_ratio_);
  yaml_parser.getYamlParam("matcher_type", &matcher_type_);
  yaml_parser.getYamlParam("nfeatures", &nfeatures_);
  yaml_parser.getYamlParam("scale_factor", &scale_factor_);
  yaml_parser.getYamlParam("nlevels", &nlevels_);
  yaml_parser.getYamlParam("edge_threshold", &edge_threshold_);
  yaml_parser.getYamlParam("first_level", &first_level_);
  yaml_parser.getYamlParam("WTA_K", &WTA_K_);

  int score_type_id;
  yaml_parser.getYamlParam("score_type_id", &score_type_id);
  switch (score_type_id) {
    case 0:
      score_type_ = cv::ORB::HARRIS_SCORE;
      break;
    // TODO(marcus): add the rest of the options here
    default:
      throw std::runtime_error("LCDparams parseYAML: wrong score_type_id");
      break;
  }
  yaml_parser.getYamlParam("patch_sze", &patch_sze_);
  yaml_parser.getYamlParam("fast_threshold", &fast_threshold_);
  yaml_parser.getYamlParam("pgo_uniform_weight", &pgo_uniform_weight_);
  yaml_parser.getYamlParam("pgo_rot_threshold", &pgo_rot_threshold_);
  yaml_parser.getYamlParam("pgo_trans_threshold", &pgo_trans_threshold_);

  return true;
}

void LoopClosureDetectorParams::print() const {
  // TODO(marcus): print all params
  LOG(INFO)
      << "$$$$$$$$$$$$$$$$$$$$$ LCD PARAMETERS $$$$$$$$$$$$$$$$$$$$$\n"
      << "loop closure method: " << (int)loop_closure_method_ << '\n'
      << "use_nss_: " << use_nss_ << '\n'
      << "alpha_: " << alpha_ << '\n'
      << "min_temporal_matches_: " << min_temporal_matches_ << '\n'
      << "dist_local_: " << dist_local_ << '\n'
      << "max_db_results_: " << max_db_results_ << '\n'
      << "max_db_results_: " << max_db_results_ << '\n'
      << "min_nss_factor_: " << min_nss_factor_ << '\n'
      << "min_matches_per_group_: " << min_matches_per_group_ << '\n'
      << "max_intragroup_gap_: " << max_intragroup_gap_ << '\n'
      << "max_distance_between_groups_: " << max_distance_between_groups_
      << '\n'
      << "max_distance_between_queries_: " << max_distance_between_queries_
      << '\n'

      << "min_correspondences_: " << min_correspondences_ << '\n'

      << "max_ransac_iterations_stereo_: " << max_ransac_iterations_stereo_
      << '\n'
      << "ransac_threshold_stereo_: " << ransac_threshold_stereo_ << '\n'
      << "relative_pose_opt_iterations_: " << relative_pose_opt_iterations_ << '\n'
      << "ransac_inlier_threshold_stereo_: "
      << ransac_inlier_threshold_stereo_ << '\n'
      << "relative_pose_info_damper_:" << relative_pose_info_damper_ << '\n'

      << "lowe_ratio_: " << lowe_ratio_ << '\n'
      << "matcher_type_:" << static_cast<unsigned int>(matcher_type_) << '\n'

      << "nfeatures_: " << nfeatures_ << '\n'
      << "scale_factor_: " << scale_factor_ << '\n'
      << "nlevels_: " << nlevels_ << '\n'
      << "edge_threshold_: " << edge_threshold_ << '\n'
      << "first_level_: " << first_level_ << '\n'
      << "WTA_K_: " << WTA_K_ << '\n'
      << "score_type_: " << score_type_ << '\n'
      << "patch_sze_: " << patch_sze_ << '\n'
      << "fast_threshold_: " << fast_threshold_ << '\n'

      << "pgo_uniform_weight_: " << pgo_uniform_weight_ << '\n'
      << "pgo_rot_threshold_: " << pgo_rot_threshold_ << '\n'
      << "pgo_trans_threshold_: " << pgo_trans_threshold_;
}
}  // namespace swift_vio
