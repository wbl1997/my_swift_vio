/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   VioBackEndParams.h
 * @brief  Class parsing the parameters for the VIO's Backend from a YAML file.
 * @author Antoni Rosinol, Luca Carlone
 */

#pragma once

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include <glog/logging.h>

#include "okvis/PipelineParams.h"
#include "okvis/class_macros.hpp"
#include "io_wrap/YamlParser.h"

namespace okvis {

/** \struct Backend Output Params
 * \brief Params controlling what the backend outputs.
 */
struct BackendOutputParams {
 public:
  BackendOutputParams(
      const bool& output_map_lmk_ids_to_3d_points_in_time_horizon,
      const int& min_num_obs_for_lmks_in_time_horizon,
      const bool& output_lmk_id_to_lmk_type_map)
      : output_map_lmk_ids_to_3d_points_in_time_horizon_(
            output_map_lmk_ids_to_3d_points_in_time_horizon),
        min_num_obs_for_lmks_in_time_horizon_(
            min_num_obs_for_lmks_in_time_horizon),
        output_lmk_id_to_lmk_type_map_(output_lmk_id_to_lmk_type_map) {}
  ~BackendOutputParams() = default;

 public:
  //! Whether to output the map from lmk ids to actual lmk 3D positions for
  //! those landmarks that are in the time-horizon of the backend optimization.
  bool output_map_lmk_ids_to_3d_points_in_time_horizon_ = false;
  //! Minimum number of observations for a landmark to be included in the
  //! output of the map from landmark ids to actual landmark 3D positions.
  int min_num_obs_for_lmks_in_time_horizon_ = 4u;
  //! Whether to output as well the type of lmk id (smart, projection, etc).
  //! This is typically used for visualization, to display lmks with different
  //! colors depending on their type.
  bool output_lmk_id_to_lmk_type_map_ = false;
};

enum class BackendModality {
  //! Only use structureless factors.
  STRUCTURELESS = 0,
  //! Only use projection factors
  PROJECTION = 1,
};

class BackendParams : public PipelineParams {
 public:
  POINTER_TYPEDEFS(BackendParams);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BackendParams();
  virtual ~BackendParams() = default;

 public:
  virtual bool equals(const BackendParams& vp2, double tol = 1e-8) const;
  void print() const override;
  bool parseYAML(const std::string& filepath) override;

 protected:
  bool equals(const PipelineParams& obj) const override {
    const auto& rhs = static_cast<const BackendParams&>(obj);
    return equals(rhs, 1e-8);
  }
  bool parseYAMLVioBackEndParams(const YamlParser& yaml_parser);
  bool equalsVioBackEndParams(const BackendParams& vp2,
                              double tol = 1e-8) const;
  void printVioBackEndParams() const;

 public:
  BackendModality backendModality_ = BackendModality::STRUCTURELESS;

  //! Smart factor params
  double smartNoiseSigma_ = 1.5;
  double rankTolerance_ = 1.0;
  //! max distance to triangulate point in meters
  double landmarkDistanceThreshold_ = 20.0;
  //! max acceptable reprojection error // before tuning: 3
  double outlierRejection_ = 8.0;
  double retriangulationThreshold_ = 1.0e-3;

  bool addBetweenStereoFactors_ = true;

  // Inverse of variance
  double betweenRotationPrecision_ = 0.0;
  double betweenTranslationPrecision_ = 1 / (0.1 * 0.1);

  //! iSAM params
  double relinearizeThreshold_ = 1.0e-2;
  double relinearizeSkip_ = 1.0;

  int numOptimize_ = 1;
  double wildfire_threshold_ = 0.001;
  bool useDogLeg_ = false;

  //! Batch smoother params
  double initialLambda_ = 0.0;  // same as GN
  double lowerBoundLambda_ = 0.0;  // same as GN
  double upperBoundLambda_ = 0.0;  // same as GN

  //! No Motion params
  double zeroVelocitySigma_ = 1.0e-3;
  double noMotionPositionSigma_ = 1.0e-3;
  double noMotionRotationSigma_ = 1.0e-4;
  double constantVelSigma_ = 1.0e-2;
};


}  // namespace okvis
