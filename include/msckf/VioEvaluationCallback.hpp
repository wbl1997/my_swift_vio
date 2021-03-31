#ifndef INCLUDE_MSCKF_VIO_EVALUATION_CALLBACK_HPP_
#define INCLUDE_MSCKF_VIO_EVALUATION_CALLBACK_HPP_

#include <vector>
#include <msckf/CameraRig.hpp>
#include <msckf/imu/ImuRig.hpp>
#include <msckf/PointSharedData.hpp>
#include <okvis/ceres/ParameterBlock.hpp>

namespace msckf {
class VioEvaluationCallback : public ::ceres::EvaluationCallback
{
public:
  VioEvaluationCallback();
  virtual ~VioEvaluationCallback();
  void PrepareForEvaluation(bool evaluate_jacobians,
                            bool new_evaluation_point) final;

 private:
  std::unordered_map<uint64_t, std::shared_ptr<msckf::PointSharedData>> pointId2PointData_Map_;
};
} // namespace msckf
#endif // INCLUDE_MSCKF_VIO_EVALUATION_CALLBACK_HPP_
