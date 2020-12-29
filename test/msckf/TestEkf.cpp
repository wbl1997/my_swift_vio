
#include <gtest/gtest.h>

#include <random>

#include "msckf/memory.h"
#include <msckf/BaseFilter.h>
#include <msckf/MultipleTransformPointJacobian.hpp>

#include <okvis/cameras/CameraBase.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/sophus_operators.hpp>

namespace okvis {
class SimPoseFilter : public BaseFilter {
public:
  SimPoseFilter()
      : poseParameterBlock(new okvis::ceres::PoseParameterBlock(
            okvis::kinematics::Transformation(), 1u, okvis::Time(0.0))) {}

  virtual ~SimPoseFilter() {}

  void
  cloneFilterStates(StatePointerAndEstimateList *currentStates) const override {
    currentStates->clear();
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> T_WC(
        poseParameterBlock->parameters());
    currentStates->emplace_back(poseParameterBlock, T_WC);
  }

  int computeStackedJacobianAndResidual(
      Eigen::MatrixXd *T_H, Eigen::Matrix<double, Eigen::Dynamic, 1> *r_q,
      Eigen::MatrixXd *R_q) const override {
    T_H->resize(observations.size() * 2, 6);
    r_q->resize(observations.size() * 2);
    R_q->resize(observations.size() * 2, observations.size() * 2);
    R_q->setIdentity();
    int validObservations = 0;
    okvis::kinematics::Transformation T_WC_est = poseParameterBlock->estimate();
    AlignedVector<okvis::kinematics::Transformation> transformList{T_WC_est};

    for (size_t j = 0u; j < observations.size(); ++j) {
      Eigen::Vector2d imagePoint;
      Eigen::Matrix<double, 2, 4> dz_dpCtj;
      Eigen::Vector4d pCtj = T_WC_est.inverse() * observedCorners[j];
      cameras::CameraBase::ProjectionStatus status =
          cameraSystem->cameraGeometry(0)->projectHomogeneous(pCtj, &imagePoint,
                                                              &dz_dpCtj);
      if (status != okvis::cameras::CameraBase::ProjectionStatus::Successful) {
        continue;
      }
      r_q->segment<2>(2 * validObservations) = observations[j] - imagePoint;
      okvis::MultipleTransformPointJacobian mtpj(transformList, {-1},
                                                 observedCorners[j]);
      T_H->block<2, 6>(2 * validObservations, 0) = dz_dpCtj * mtpj.dp_dT(0u);
      (*R_q)(validObservations * 2, validObservations * 2) =
          std::pow(observationStddev[j][0], 2);
      (*R_q)(validObservations * 2 + 1, validObservations * 2 + 1) =
          std::pow(observationStddev[j][1], 2);
      validObservations++;
    }
    LOG(INFO) << "Linearized observations " << validObservations;
    return validObservations;
  }

  void boxminusFromInput(
      const StatePointerAndEstimateList &refState,
      Eigen::Matrix<double, Eigen::Dynamic, 1> *deltaX) const override {
    int covDim = covariance_.rows();
    deltaX->resize(covDim, 1);
    Eigen::Matrix<double, 6, 1> delta_T_WC;
    okvis::ceres::PoseLocalParameterization::minus(
        refState.at(0).parameterBlockPtr->parameters(),
        refState.at(0).parameterEstimate.data(), delta_T_WC.data());
    deltaX->head<6>() = delta_T_WC;
  }

  void updateStates(
      const Eigen::Matrix<double, Eigen::Dynamic, 1> &deltaX) override {
    kinematics::Transformation T_WC = poseParameterBlock->estimate();
    // In effect this amounts to PoseParameterBlock::plus().
    Eigen::Vector3d deltaAlpha = deltaX.segment<3>(3);
    Eigen::Quaterniond deltaq = okvis::kinematics::expAndTheta(deltaAlpha);
    T_WC = kinematics::Transformation(T_WC.r() + deltaX.head<3>(),
                                      deltaq * T_WC.q());
    poseParameterBlock->setEstimate(T_WC);
  }

  void setPoseEstimate(const okvis::kinematics::Transformation &T_WC) {
    poseParameterBlock->setEstimate(T_WC);
  }

  void setCovariance(const Eigen::Matrix<double, 6, 6> &covPose) {
    covariance_ = covPose;
  }

  const Eigen::MatrixXd &covariance() const { return covariance_; }

  okvis::kinematics::Transformation estimate() const {
    return poseParameterBlock->estimate();
  }

  void
  setCameraSystem(std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem) {
    this->cameraSystem = cameraSystem;
  }

  void
  setObservations(const AlignedVector<Eigen::Vector4d> &observedCorners,
                  const AlignedVector<Eigen::Vector2d> &observations,
                  const AlignedVector<Eigen::Vector2d> &observationStddev) {
    this->observedCorners = observedCorners;
    this->observations = observations;
    this->observationStddev = observationStddev;
  }

  std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock;
  std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem;

  AlignedVector<Eigen::Vector4d> observedCorners;
  AlignedVector<Eigen::Vector2d> observations;
  AlignedVector<Eigen::Vector2d> observationStddev;
};
} // namespace okvis

class BaseFilterTest : public ::testing::Test {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
protected:
  BaseFilterTest() {

    corners.resize(cols * rows);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        corners[i * cols + j] = Eigen::Vector4d(i * spacing, j * spacing, 0, 1);
      }
    }

    std::shared_ptr<const okvis::kinematics::Transformation> T_SC_0(
        new okvis::kinematics::Transformation());
    std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry0(
        okvis::cameras::PinholeCamera<
            okvis::cameras::EquidistantDistortion>::createTestObject());
    cameraSystem.reset(new okvis::cameras::NCameraSystem);
    cameraSystem->addCamera(
        T_SC_0, cameraGeometry0,
        okvis::cameras::NCameraSystem::DistortionType::Equidistant);
    filter.setCameraSystem(cameraSystem);
  }

  void SetUp() override {}

  void simulateObservations(const okvis::kinematics::Transformation &T_WS) {
    T_WS_ref = T_WS;
    observations.clear();
    observations.reserve(corners.size());
    observedCorners.clear();
    observedCorners.reserve(corners.size());

    size_t camId = 0u;
    for (size_t j = 0u; j < corners.size(); ++j) {
      Eigen::Vector2d projection;
      Eigen::Vector4d point_C = cameraSystem->T_SC(camId)->inverse() *
                                T_WS_ref.inverse() * corners[j];
      okvis::cameras::CameraBase::ProjectionStatus status =
          cameraSystem->cameraGeometry(camId)->projectHomogeneous(point_C,
                                                                  &projection);
      if (status == okvis::cameras::CameraBase::ProjectionStatus::Successful) {
        observations.emplace_back(projection);
        observedCorners.emplace_back(corners[j]);
      }
    }
  }

  void setupFilter(bool addNoise) {
    T_WS_init = T_WS_ref;
    T_WS_init.oplus(Eigen::Matrix<double, 6, 1>::Random() * 0.2);

    filter.setPoseEstimate(T_WS_init);
    double sigmaTranslation = 0.2;
    double sigmaRotation = 0.2;
    Eigen::Matrix<double, 6, 1> covDiagonal;
    covDiagonal.head<3>().setConstant(sigmaTranslation);
    covDiagonal.tail<3>().setConstant(sigmaRotation);
    covInitial = covDiagonal.asDiagonal();
    filter.setCovariance(covInitial);
    if (addNoise) {
      observationStddev.clear();
      observationStddev.reserve(corners.size());
      for (size_t j = 0u; j < observations.size(); ++j) {
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<> d{0, imageNoiseStd};
        observations[j][0] += d(gen);
        observations[j][1] += d(gen);
        observationStddev.emplace_back(
            Eigen::Vector2d(imageNoiseStd, imageNoiseStd));
      }
    } else {
      for (size_t j = 0u; j < observations.size(); ++j) {
        observationStddev.emplace_back(
            Eigen::Vector2d(imageNoiseStd * 0.1, imageNoiseStd * 0.1));
      }
    }
    filter.setObservations(observedCorners, observations, observationStddev);
  }

  okvis::SimPoseFilter filter;

  okvis::kinematics::Transformation T_WS_ref;
  int cols = 6;
  int rows = 7;
  double spacing = 0.06;
  double imageNoiseStd = 1.0;
  AlignedVector<Eigen::Vector4d> corners;
  std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem;

  AlignedVector<Eigen::Vector2d> observations;
  AlignedVector<Eigen::Vector2d> observationStddev;
  AlignedVector<Eigen::Vector4d> observedCorners;

  okvis::kinematics::Transformation T_WS_init;
  Eigen::Matrix<double, 6, 6> covInitial;
};

inline void isRotationClose(const okvis::kinematics::Transformation &T_expected,
                            const okvis::kinematics::Transformation &T_actual,
                            double tol) {
  EXPECT_LT((T_expected.q().coeffs() - T_actual.q().coeffs())
                .lpNorm<Eigen::Infinity>(),
            tol)
      << "T_expected q:" << T_expected.q().coeffs().transpose()
      << "\nT_actual q:" << T_actual.q().coeffs().transpose();
}

inline void
isTranslationClose(const okvis::kinematics::Transformation &T_expected,
                   const okvis::kinematics::Transformation &T_actual,
                   double tol) {
  EXPECT_LT((T_expected.r() - T_actual.r()).lpNorm<Eigen::Infinity>(), tol)
      << "T_expected r:" << T_expected.q().coeffs().transpose()
      << "\nT_actual r:" << T_actual.q().coeffs().transpose();
}

TEST_F(BaseFilterTest, EkfUpdate) {
  okvis::kinematics::Transformation T_shift(
      Eigen::Vector3d(0.3, 0.4, 2),
      Eigen::AngleAxisd(0.01, Eigen::Vector3d::UnitX()) *
          Eigen::AngleAxisd(0.03, Eigen::Vector3d::UnitZ()));
  Eigen::Matrix3d R_WC_canon;
  R_WC_canon << 0, 1, 0, 1, 0, 0, 0, 0, -1;

  okvis::kinematics::Transformation T_WC_canon(
      Eigen::Vector3d(spacing * rows * 0.5, spacing * cols * 0.5, 0),
      Eigen::Quaterniond(R_WC_canon));
  simulateObservations(T_shift * T_WC_canon);
  setupFilter(false);
  filter.updateEkf(0, 6);
  isRotationClose(T_WS_ref, filter.estimate(), 1e-2);
  isTranslationClose(T_WS_ref, filter.estimate(), 8e-2);

//  LOG(INFO) << "Initial cov\n"
//            << covInitial << "\nFiltered cov\n"
//            << filter.covariance();
}

TEST_F(BaseFilterTest, IekfUpdate) {
  okvis::kinematics::Transformation T_shift(
      Eigen::Vector3d(0.3, 0.4, 2),
      Eigen::AngleAxisd(0.01, Eigen::Vector3d::UnitX()) *
          Eigen::AngleAxisd(0.03, Eigen::Vector3d::UnitZ()));
  Eigen::Matrix3d R_WC_canon;
  R_WC_canon << 0, 1, 0, 1, 0, 0, 0, 0, -1;

  okvis::kinematics::Transformation T_WC_canon(
      Eigen::Vector3d(spacing * rows * 0.5, spacing * cols * 0.5, 0),
      Eigen::Quaterniond(R_WC_canon));
  simulateObservations(T_shift * T_WC_canon);
  setupFilter(false);
  filter.updateIekf(0, 6);
  isRotationClose(T_WS_ref, filter.estimate(), 1e-5);
  isTranslationClose(T_WS_ref, filter.estimate(), 1e-4);
//  LOG(INFO) << "Initial cov\n"
//            << covInitial << "\nFiltered cov\n"
//            << filter.covariance();
}
