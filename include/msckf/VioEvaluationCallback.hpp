#ifndef INCLUDE_MSCKF_VIO_EVALUATION_CALLBACK_HPP_
#define INCLUDE_MSCKF_VIO_EVALUATION_CALLBACK_HPP_

#include <vector>
#include <msckf/CameraRig.hpp>
#include <msckf/ImuRig.hpp>
#include <okvis/ceres/ParameterBlock.hpp>

#include <glog/logging.h>

namespace msckf {
class VioEvaluationCallback : public ::ceres::EvaluationCallback
{
public:
  VioEvaluationCallback();
  virtual ~VioEvaluationCallback();

  void addCameraExtrinsicParam(
      std::shared_ptr<okvis::ceres::ParameterBlock> extrinsicParamBlockPtr, int i) {
    CHECK_EQ(i, cameraExtrinsicParams_.size());
    cameraExtrinsicParams_.push_back(extrinsicParamBlockPtr);
  }

  void addCameraProjectionIntrinsicParam(
      std::shared_ptr<okvis::ceres::ParameterBlock>
                 projIntrinsicParamBlockPtr, int i) {
    CHECK_EQ(i, cameraProjectionIntrinsicParams_.size());
    cameraProjectionIntrinsicParams_.push_back(projIntrinsicParamBlockPtr);
  }

  void addCameraDistortionParam(
      std::shared_ptr<okvis::ceres::ParameterBlock> distortionParamBlockPtr, int i) {
    CHECK_EQ(i, cameraDistortionParams_.size());
    cameraDistortionParams_.push_back(distortionParamBlockPtr);
  }

  void addCameraTimeOffsetParam(std::shared_ptr<okvis::ceres::ParameterBlock>
                                    cameraTimeOffsetParamBlockPtr, int i) {
    CHECK_EQ(i, cameraTimeOffsetParams_.size());
    cameraTimeOffsetParams_.push_back(cameraTimeOffsetParamBlockPtr);
  }

  void addCameraReadoutTimeParam(std::shared_ptr<okvis::ceres::ParameterBlock>
                                     cameraReadoutTimeParamBlockPtr, int i) {
    CHECK_EQ(i, cameraReadoutTimeParams_.size());
    cameraReadoutTimeParams_.push_back(cameraReadoutTimeParamBlockPtr);
  }

  void addSpeedAndBiasParam(
      std::shared_ptr<okvis::ceres::ParameterBlock> speedAndBiasParamBlockPtr) {
    imuSpeedBiasParams_ = speedAndBiasParamBlockPtr;
  }

  void addImuAugmentedParams(
      std::vector<std::shared_ptr<okvis::ceres::ParameterBlock>>
                 imuAugmentedParamBlockPtrs) {
    imuAugmentedParams_ = imuAugmentedParamBlockPtrs;
  }

  void PrepareForEvaluation(bool evaluate_jacobians,
                            bool new_evaluation_point) final;

 private:
  std::vector<std::shared_ptr<okvis::ceres::ParameterBlock>>
      cameraExtrinsicParams_;
  std::vector<std::shared_ptr<okvis::ceres::ParameterBlock>>
      cameraProjectionIntrinsicParams_;
  std::vector<std::shared_ptr<okvis::ceres::ParameterBlock>>
      cameraDistortionParams_;
  std::vector<std::shared_ptr<okvis::ceres::ParameterBlock>>
      cameraTimeOffsetParams_;
  std::vector<std::shared_ptr<okvis::ceres::ParameterBlock>>
      cameraReadoutTimeParams_;

  std::shared_ptr<okvis::ceres::ParameterBlock> imuSpeedBiasParams_;
  std::vector<std::shared_ptr<okvis::ceres::ParameterBlock>>
      imuAugmentedParams_;

  std::shared_ptr<okvis::cameras::CameraRig> cameraRigPtr_;
  std::shared_ptr<okvis::ImuRig> imuRigPtr_;
};
} // namespace msckf
#endif // INCLUDE_MSCKF_VIO_EVALUATION_CALLBACK_HPP_
