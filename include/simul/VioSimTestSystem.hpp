#ifndef INCLUDE_SWIFT_VIO_VIO_TEST_SYSTEM_BUILDER_HPP_
#define INCLUDE_SWIFT_VIO_VIO_TEST_SYSTEM_BUILDER_HPP_

#include <gtsam/VioBackEndParams.h>

#include <simul/SimulationFrontend.hpp>
#include <simul/SimDataInterface.hpp>
#include <simul/SimParameters.h>

#include <swift_vio/MSCKF.hpp>

namespace simul {
typedef std::function<void(const Eigen::VectorXd &mse,
                           const Eigen::VectorXd &desiredStdevs,
                           const std::vector<std::string> &dimensionLabels)>
    CheckMseCallback;

typedef std::function<void(const Eigen::Vector3d &nees)> CheckNeesCallback;

class VioSimTestSystem {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VioSimTestSystem(const CheckMseCallback& checkMseCallback, const CheckNeesCallback& checkNeesCallback) :
     checkMseCallback_(checkMseCallback), checkNeesCallback_(checkNeesCallback) {
  }

  virtual ~VioSimTestSystem() {}

  void createSensorSystem(const TestSetting &testSetting);

  void createEstimator(const TestSetting &testSetting);

  void run(const simul::TestSetting &testSetting,
           const std::string &outputPath);

  void setCheckNeesCallback(const CheckNeesCallback& checkNeesCallback) {
    checkNeesCallback_ = checkNeesCallback;
  }

  void setCheckMseCallback(const CheckMseCallback& checkMseCallback) {
    checkMseCallback_ = checkMseCallback;
  }

private:
  std::shared_ptr<SimDataInterface> simData_;


  std::shared_ptr<okvis::cameras::NCameraSystem> refCameraSystem_;
  std::shared_ptr<okvis::cameras::NCameraSystem> noisyCameraSystem_;
  okvis::ExtrinsicsEstimationParameters extrinsicsEstimationParameters_;

  okvis::ImuParameters imuParameters_;


  std::shared_ptr<okvis::Estimator> estimator_;
  swift_vio::InitialNavState initialNavState_;
  std::shared_ptr<SimulationFrontend> frontend_;

  std::shared_ptr<::ceres::EvaluationCallback> evaluationCallback_;
  CheckMseCallback checkMseCallback_;
  CheckNeesCallback checkNeesCallback_;
};

} // namespace simul
#endif // INCLUDE_SWIFT_VIO_VIO_TEST_SYSTEM_BUILDER_HPP_
