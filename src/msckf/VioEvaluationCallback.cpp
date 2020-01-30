#include "msckf/VioEvaluationCallback.hpp"

namespace msckf {
VioEvaluationCallback::VioEvaluationCallback()
{

}

VioEvaluationCallback::~VioEvaluationCallback() {

}

void VioEvaluationCallback::PrepareForEvaluation(
    bool /*evaluate_jacobians*/, bool new_evaluation_point) {
  if (new_evaluation_point) {
    // updateSensorRigs which includes the shared data of T_BCi, cameraGeometry

      // TODO(jhuai): do this prior to Evaluate() or EvaluateWithMinimalJacobians() in a EvaluationCallback().
    //  Eigen::VectorXd intrinsics(GEOMETRY_TYPE::NumIntrinsics);
    //  Eigen::Map<const Eigen::Matrix<double, PROJ_INTRINSIC_MODEL::kNumParams, 1>>
    //      projIntrinsics(parameters[3]);
    //  PROJ_INTRINSIC_MODEL::localToGlobal(projIntrinsics, &intrinsics);

    //  Eigen::Map<const Eigen::Matrix<double, kDistortionDim, 1>>
    //      distortionIntrinsics(parameters[4]);
    //  intrinsics.tail<kDistortionDim>() = distortionIntrinsics;
    //  cameraGeometryBase_->setIntrinsics(intrinsics);
  }

}
} // namespace msckf
