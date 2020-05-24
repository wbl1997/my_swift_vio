#include "msckf/PointLandmarkSimulationRS.hpp"

#include <random>

#include <glog/logging.h>

#include <Eigen/Core>

#include <msckf/MultipleTransformPointJacobian.hpp>
#include <msckf/TransformMultiplyJacobian.hpp>

void PointLandmarkSimulationRS::projectLandmarksToNFrame(
    const std::vector<Eigen::Vector4d,
                      Eigen::aligned_allocator<Eigen::Vector4d>>&
        homogeneousPoints,
    std::shared_ptr<const simul::CircularSinusoidalTrajectory> simulatedTrajectory,
    okvis::Time trueCentralRowEpoch,
    std::shared_ptr<const okvis::cameras::NCameraSystem> cameraSystemRef,
    std::shared_ptr<okvis::MultiFrame> framesInOut,
    std::vector<std::vector<size_t>>* frameLandmarkIndices,
    std::vector<std::vector<int>>* keypointIndices,
    const double* imageNoiseMag) {
  size_t numFrames = framesInOut->numFrames();
  std::vector<std::vector<cv::KeyPoint>> frame_keypoints;

  okvis::kinematics::Transformation T_WS_ref =
      simulatedTrajectory->computeGlobalPose(trueCentralRowEpoch);
  // project landmarks onto frames of framesInOut
  for (size_t i = 0; i < numFrames; ++i) {
    std::vector<size_t> lmk_indices;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<int> frameKeypointIndices(homogeneousPoints.size(), -1);

    for (size_t j = 0; j < homogeneousPoints.size(); ++j) {
      Eigen::Vector2d projection;
      double tr = cameraSystemRef->cameraGeometry(i)->readoutTime();
      double height = cameraSystemRef->cameraGeometry(i)->imageHeight();
      okvis::cameras::CameraBase::ProjectionStatus status;
      if (tr > 1e-6) {
        double dt = 1e6; // f(t) / f(t')
        double relativeFeatureTime = 0; // feature epoch relative to central row, i.e., frame timestamp.
        const double tol = 1e-5;
        int numIter = 0;
        const int maxIter = 5;
        while (numIter < maxIter && std::fabs(dt) > tol) {
            okvis::Time featureTime = trueCentralRowEpoch + okvis::Duration(relativeFeatureTime);
            okvis::kinematics::Transformation T_WBt =
                simulatedTrajectory->computeGlobalPose(featureTime);
            Eigen::Vector4d point_C = cameraSystemRef->T_SC(i)->inverse() *
                                      T_WBt.inverse() * homogeneousPoints[j];
            Eigen::Matrix<double, 2, 4> pointJacobian;
            status = cameraSystemRef->cameraGeometry(i)->projectHomogeneous(
                point_C, &projection, &pointJacobian);
            if (status != okvis::cameras::CameraBase::ProjectionStatus::Successful) {
                break;
            }
            double f_of_t = projection[1] - (relativeFeatureTime / tr + 0.5) * height;

            // compute Jacobians required by Newton Raphson method.
            AlignedVector<okvis::kinematics::Transformation> transformList{*cameraSystemRef->T_SC(i), T_WBt};
            std::vector<int> exponentList{-1, -1};
            okvis::MultipleTransformPointJacobian mtpj(transformList, exponentList, homogeneousPoints[j]);
            Eigen::Matrix<double, 4, 6> dpC_dT_WB = mtpj.dp_dT(0);
            okvis::kinematics::Transformation T_identity;
            Eigen::Vector3d v_WB = simulatedTrajectory->computeGlobalLinearVelocity(featureTime);
            Eigen::Vector3d omega_W =  simulatedTrajectory->computeGlobalAngularRate(featureTime);
            Eigen::Vector3d omega_B;
            omega_B.noalias() = T_WBt.C().transpose() * omega_W;
            msckf::TransformMultiplyJacobian tmj(T_WBt, T_identity, v_WB, omega_B);
            Eigen::Matrix<double, 6, 1> dT_WB_dt;
            dT_WB_dt.head<3>() = tmj.dp_dt();
            dT_WB_dt.tail<3>() = tmj.dtheta_dt();

            double fprime_of_t = pointJacobian.row(1) * dpC_dT_WB * dT_WB_dt - height / tr;
            dt = f_of_t / fprime_of_t;
//            LOG(INFO) << "Rs projection iter " << numIter << " dt " << dt;
            relativeFeatureTime -= dt;
            ++numIter;
        }
      } else {
        Eigen::Vector4d point_C = cameraSystemRef->T_SC(i)->inverse() *
                                  T_WS_ref.inverse() * homogeneousPoints[j];

        status = cameraSystemRef->cameraGeometry(i)->projectHomogeneous(
            point_C, &projection);
      }
      if (status == okvis::cameras::CameraBase::ProjectionStatus::Successful) {
        Eigen::Vector2d measurement(projection);
        if (imageNoiseMag) {
          std::random_device rd{};
          std::mt19937 gen{rd()};
          std::normal_distribution<> d{0, *imageNoiseMag};
          measurement[0] += d(gen);
          measurement[1] += d(gen);
        }
        frameKeypointIndices[j] = keypoints.size();
        keypoints.emplace_back(measurement[0], measurement[1], 8.0);
        lmk_indices.emplace_back(j);
      }
    }
    frameLandmarkIndices->emplace_back(lmk_indices);
    frame_keypoints.emplace_back(keypoints);
    framesInOut->resetKeypoints(i, keypoints);
    keypointIndices->emplace_back(frameKeypointIndices);
  }
}
