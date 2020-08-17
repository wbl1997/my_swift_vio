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

namespace okvis {

BackendParams::BackendParams() : PipelineParams("Backend Parameters") {
  // Trivial sanity checks.
  CHECK_GE(horizon_, 0);
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
  // INITIALIZATION
  yaml_parser.getYamlParam("autoInitialize", &autoInitialize_);
  yaml_parser.getYamlParam("roundOnAutoInitialize", &roundOnAutoInitialize_);
  yaml_parser.getYamlParam("initialPositionSigma", &initialPositionSigma_);
  yaml_parser.getYamlParam("initialRollPitchSigma", &initialRollPitchSigma_);
  yaml_parser.getYamlParam("initialYawSigma", &initialYawSigma_);
  yaml_parser.getYamlParam("initialVelocitySigma", &initialVelocitySigma_);
  yaml_parser.getYamlParam("initialAccBiasSigma", &initialAccBiasSigma_);
  yaml_parser.getYamlParam("initialGyroBiasSigma", &initialGyroBiasSigma_);

  // VISION PARAMS
  yaml_parser.getYamlParam("smartNoiseSigma", &smartNoiseSigma_);
  yaml_parser.getYamlParam("rankTolerance", &rankTolerance_);
  yaml_parser.getYamlParam("landmarkDistanceThreshold",
                           &landmarkDistanceThreshold_);
  yaml_parser.getYamlParam("outlierRejection", &outlierRejection_);
  yaml_parser.getYamlParam("retriangulationThreshold",
                           &retriangulationThreshold_);
  yaml_parser.getYamlParam("raySigmaScalar",
                           &raySigmaScalar_);
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
  yaml_parser.getYamlParam("horizon", &horizon_);
  yaml_parser.getYamlParam("wildfire_threshold", &wildfire_threshold_);
  yaml_parser.getYamlParam("useDogLeg", &useDogLeg_);

  return true;
}

bool BackendParams::equalsVioBackEndParams(const BackendParams& vp2,
                                              double tol) const {
  return
      // INITIALIZATION
      (autoInitialize_ == vp2.autoInitialize_) &&
      (roundOnAutoInitialize_ == vp2.roundOnAutoInitialize_) &&
      (fabs(initialPositionSigma_ - vp2.initialPositionSigma_) <= tol) &&
      (fabs(initialRollPitchSigma_ - vp2.initialRollPitchSigma_) <= tol) &&
      (fabs(initialYawSigma_ - vp2.initialYawSigma_) <= tol) &&
      (fabs(initialVelocitySigma_ - vp2.initialVelocitySigma_) <= tol) &&
      (fabs(initialAccBiasSigma_ - vp2.initialAccBiasSigma_) <= tol) &&
      (fabs(initialGyroBiasSigma_ - vp2.initialGyroBiasSigma_) <= tol) &&
      // VISION PARAMS
      (fabs(smartNoiseSigma_ - vp2.smartNoiseSigma_) <= tol) &&
      (fabs(rankTolerance_ - vp2.rankTolerance_) <= tol) &&
      (fabs(landmarkDistanceThreshold_ - vp2.landmarkDistanceThreshold_) <=
       tol) &&
      (fabs(outlierRejection_ - vp2.outlierRejection_) <= tol) &&
      (fabs(retriangulationThreshold_ - vp2.retriangulationThreshold_) <=
       tol) &&
      (fabs(raySigmaScalar_ - vp2.raySigmaScalar_) <= tol) &&
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
      (numOptimize_ == vp2.numOptimize_) && (horizon_ == vp2.horizon_) &&
      (wildfire_threshold_ == vp2.wildfire_threshold_) &&
      (useDogLeg_ == vp2.useDogLeg_);
}

void BackendParams::printVioBackEndParams() const {
  LOG(INFO) << "$$$$$$$$$$$$$$$$$$$$$ VIO PARAMETERS $$$$$$$$$$$$$$$$$$$$$\n"
            << "** INITIALIZATION parameters **\n"
            << "autoInitialize_: " << autoInitialize_ << '\n'
            << "initial_ground_truth_state_: ";
  LOG(INFO) << "roundOnAutoInitialize_: " << roundOnAutoInitialize_ << '\n'
            << "initialPositionSigma: " << initialPositionSigma_ << '\n'
            << "initialRollPitchSigma: " << initialRollPitchSigma_ << '\n'
            << "initialYawSigma: " << initialYawSigma_ << '\n'
            << "initialVelocitySigma: " << initialVelocitySigma_ << '\n'
            << "initialAccBiasSigma: " << initialAccBiasSigma_ << '\n'
            << "initialGyroBiasSigma: " << initialGyroBiasSigma_ << '\n'

            << "** VISION parameters **\n"
            << "rankTolerance_: " << rankTolerance_ << '\n'
            << "landmarkDistanceThreshold_: " << landmarkDistanceThreshold_
            << '\n'
            << "outlierRejection_: " << outlierRejection_ << '\n'
            << "retriangulationThreshold_: " << retriangulationThreshold_
            << '\n' << "raySigmaScalar_: " << raySigmaScalar_ << '\n'
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
            << "horizon_: " << horizon_ << '\n'
            << "wildfire_threshold_: " << wildfire_threshold_ << '\n'
            << "useDogLeg_: " << useDogLeg_;
}

}  // namespace okvis
