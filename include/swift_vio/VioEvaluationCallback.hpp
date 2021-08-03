#ifndef INCLUDE_SWIFT_VIO_VIO_EVALUATION_CALLBACK_HPP_
#define INCLUDE_SWIFT_VIO_VIO_EVALUATION_CALLBACK_HPP_

#include <vector>
#include <swift_vio/CameraRig.hpp>
#include <swift_vio/imu/ImuRig.hpp>
#include <swift_vio/PointSharedData.hpp>
#include <okvis/ceres/ParameterBlock.hpp>

namespace swift_vio {
class VioEvaluationCallback : public ::ceres::EvaluationCallback
{
public:
  VioEvaluationCallback();
  virtual ~VioEvaluationCallback();
  void PrepareForEvaluation(bool evaluate_jacobians,
                            bool new_evaluation_point) final;

 private:
  std::unordered_map<uint64_t, std::shared_ptr<swift_vio::PointSharedData>> pointId2PointData_Map_;
};
} // namespace swift_vio
#endif // INCLUDE_SWIFT_VIO_VIO_EVALUATION_CALLBACK_HPP_
