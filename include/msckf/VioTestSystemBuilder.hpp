#ifndef INCLUDE_MSCKF_VIO_TEST_SYSTEM_BUILDER_HPP_
#define INCLUDE_MSCKF_VIO_TEST_SYSTEM_BUILDER_HPP_

#include <msckf/ImuSimulator.h>
#include <msckf/MSCKF2.hpp>
#include <msckf/SimulationFrontend.hpp>
#include <msckf/CameraSystemCreator.hpp>

namespace simul {
class VioTestSystemBuilder {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VioTestSystemBuilder() {
  }
  virtual ~VioTestSystemBuilder() {}

  void createVioSystem(const okvis::TestSetting& testSetting,
                       SimulatedTrajectoryType trajectoryId,
                       std::string projOptModelName, std::string extrinsicModelName,
                       double td, double tr,
                       std::string imuLogFile, std::string pointFile);

public:
  std::shared_ptr<okvis::Estimator> mutableEstimator() {
    return estimator;
  }

  std::shared_ptr<okvis::SimulationFrontend> mutableFrontend() {
    return frontend;
  }

  std::shared_ptr<const simul::CircularSinusoidalTrajectory> sinusoidalTrajectory() {
    return circularSinusoidalTrajectory;
  }

  std::string distortionType() const {
    return distortionType_;
  }

  std::string imuModelType() const {
    return imuModelType_;
  }

  std::vector<okvis::Time> sampleTimes() const {
    return times_;
  }

  okvis::ImuMeasurementDeque trueBiases() const {
    return trueBiases_;
  }

  std::vector<okvis::kinematics::Transformation> ref_T_WS_list() const {
    return ref_T_WS_list_;
  }

  okvis::ImuMeasurementDeque imuMeasurements() const {
    return imuMeasurements_;
  }

  std::shared_ptr<const okvis::cameras::NCameraSystem> trueCameraSystem() const {
    return trueCameraSystem_;
  }

  const okvis::InitialNavState& initialNavState() const {
    return initialNavState_;
  }

private:
  std::shared_ptr<okvis::Estimator> estimator;
  std::shared_ptr<::ceres::EvaluationCallback> evaluationCallback_;
  std::shared_ptr<okvis::SimulationFrontend> frontend;
  std::shared_ptr<simul::CircularSinusoidalTrajectory> circularSinusoidalTrajectory;
  std::string distortionType_;
  std::string imuModelType_;
  std::vector<okvis::Time> times_;
  okvis::ImuMeasurementDeque trueBiases_; // true biases used for computing RMSE
  std::vector<okvis::kinematics::Transformation> ref_T_WS_list_;
  okvis::ImuMeasurementDeque imuMeasurements_;
  std::shared_ptr<okvis::cameras::NCameraSystem> trueCameraSystem_;
  okvis::InitialNavState initialNavState_;
};

} // namespace simul
#endif // INCLUDE_MSCKF_VIO_TEST_SYSTEM_BUILDER_HPP_
