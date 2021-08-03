/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   VioBackEndParams.cpp
 * @brief  Class parsing the parameters for the VIO's Backend from a YAML file.
 * @author Antoni Rosinol, Luca Carlone
 */

#include "gtsam/VioBackEndParams.h"

namespace swift_vio {
BackendParams::BackendParams() : PipelineParams("Backend Parameters") {
  // Trivial sanity checks.
  CHECK_GE(numOptimize_, 0);
}

bool BackendParams::equals(const BackendParams& vp2, double tol) const {
  return equalsVioBackEndParams(vp2, tol);
}

void BackendParams::print() const { printVioBackEndParams(); }

bool BackendParams::parseYAML(const std::string& filepath) {
  YamlParser yaml_parser(filepath);
  return parseYAMLVioBackEndParams(yaml_parser);
}

bool BackendParams::parseYAMLVioBackEndParams(
    const YamlParser& yaml_parser) {
  // VISION PARAMS
  int backend_modality = 0;
  yaml_parser.getYamlParam("backend_modality", &backend_modality);
  backendModality_ = static_cast<BackendModality>(backend_modality);

  yaml_parser.getYamlParam("smartNoiseSigma", &smartNoiseSigma_);
  yaml_parser.getYamlParam("rankTolerance", &rankTolerance_);
  yaml_parser.getYamlParam("landmarkDistanceThreshold",
                           &landmarkDistanceThreshold_);
  yaml_parser.getYamlParam("outlierRejection", &outlierRejection_);
  yaml_parser.getYamlParam("retriangulationThreshold",
                           &retriangulationThreshold_);

  yaml_parser.getYamlParam("addBetweenStereoFactors",
                           &addBetweenStereoFactors_);
  yaml_parser.getYamlParam("betweenRotationPrecision",
                           &betweenRotationPrecision_);
  yaml_parser.getYamlParam("betweenTranslationPrecision",
                           &betweenTranslationPrecision_);

  // OPTIMIZATION PARAMS
  yaml_parser.getYamlParam("relinearizeThreshold", &relinearizeThreshold_);
  yaml_parser.getYamlParam("relinearizeSkip", &relinearizeSkip_);
  yaml_parser.getYamlParam("zeroVelocitySigma", &zeroVelocitySigma_);
  yaml_parser.getYamlParam("noMotionPositionSigma", &noMotionPositionSigma_);
  yaml_parser.getYamlParam("noMotionRotationSigma", &noMotionRotationSigma_);
  yaml_parser.getYamlParam("constantVelSigma", &constantVelSigma_);
  yaml_parser.getYamlParam("numOptimize", &numOptimize_);
  yaml_parser.getYamlParam("wildfire_threshold", &wildfire_threshold_);
  yaml_parser.getYamlParam("useDogLeg", &useDogLeg_);

  yaml_parser.getYamlParam("initialLambda", &initialLambda_);
  yaml_parser.getYamlParam("lowerBoundLambda", &lowerBoundLambda_);
  yaml_parser.getYamlParam("upperBoundLambda", &upperBoundLambda_);
  return true;
}

bool BackendParams::equalsVioBackEndParams(const BackendParams& vp2,
                                              double tol) const {
  return
      // VISION PARAMS
      (backendModality_ == vp2.backendModality_) &&
      (fabs(smartNoiseSigma_ - vp2.smartNoiseSigma_) <= tol) &&
      (fabs(rankTolerance_ - vp2.rankTolerance_) <= tol) &&
      (fabs(landmarkDistanceThreshold_ - vp2.landmarkDistanceThreshold_) <=
       tol) &&
      (fabs(outlierRejection_ - vp2.outlierRejection_) <= tol) &&
      (fabs(retriangulationThreshold_ - vp2.retriangulationThreshold_) <=
       tol) &&
      (addBetweenStereoFactors_ == vp2.addBetweenStereoFactors_) &&
      (fabs(betweenRotationPrecision_ - vp2.betweenRotationPrecision_) <=
       tol) &&
      (fabs(betweenTranslationPrecision_ - vp2.betweenTranslationPrecision_) <=
       tol) &&
      // OPTIMIZATION PARAMS
      (fabs(relinearizeThreshold_ - vp2.relinearizeThreshold_) <= tol) &&
      (relinearizeSkip_ == vp2.relinearizeSkip_) &&
      (fabs(zeroVelocitySigma_ - vp2.zeroVelocitySigma_) <= tol) &&
      (fabs(noMotionPositionSigma_ - vp2.noMotionPositionSigma_) <= tol) &&
      (fabs(noMotionRotationSigma_ - vp2.noMotionRotationSigma_) <= tol) &&
      (fabs(constantVelSigma_ - vp2.constantVelSigma_) <= tol) &&
      (numOptimize_ == vp2.numOptimize_) &&
      (wildfire_threshold_ == vp2.wildfire_threshold_) &&
      (useDogLeg_ == vp2.useDogLeg_) &&
      (initialLambda_ == vp2.initialLambda_) &&
      (lowerBoundLambda_ == vp2.lowerBoundLambda_) &&
      (upperBoundLambda_ == vp2.upperBoundLambda_);
}

void BackendParams::printVioBackEndParams() const {
  LOG(INFO) << "$$$$$$$$$$$$$$$$$$$$$ VIO PARAMETERS $$$$$$$$$$$$$$$$$$$$$\n"
            << "** VISION parameters **\n"
            << "rankTolerance_: " << rankTolerance_ << '\n'
            << "landmarkDistanceThreshold_: " << landmarkDistanceThreshold_
            << '\n'
            << "outlierRejection_: " << outlierRejection_ << '\n'
            << "retriangulationThreshold_: " << retriangulationThreshold_ << '\n'
            << "addBetweenStereoFactors_: " << addBetweenStereoFactors_ << '\n'
            << "betweenRotationPrecision_: " << betweenRotationPrecision_
            << '\n'
            << "betweenTranslationPrecision_: " << betweenTranslationPrecision_
            << '\n'

            << "** OPTIMIZATION parameters **\n"
            << "relinearizeThreshold_: " << relinearizeThreshold_ << '\n'
            << "relinearizeSkip_: " << relinearizeSkip_ << '\n'
            << "zeroVelocitySigma_: " << zeroVelocitySigma_ << '\n'
            << "noMotionPositionSigma_: " << noMotionPositionSigma_ << '\n'
            << "noMotionRotationSigma_: " << noMotionRotationSigma_ << '\n'
            << "constantVelSigma_: " << constantVelSigma_ << '\n'
            << "numOptimize_: " << numOptimize_ << '\n'
            << "wildfire_threshold_: " << wildfire_threshold_ << '\n'
            << "useDogLeg_: " << useDogLeg_ << '\n'
            << "initialLambda_: " << initialLambda_ << '\n'
            << "lowerBoundLambda_: " << lowerBoundLambda_ << '\n'
            << "upperBoundLambda_: " << upperBoundLambda_;
}
}  // namespace swift_vio
