#include <gtest/gtest.h>

#include <msckf/TFVIO.hpp>
#include <msckf/MSCKF2.hpp>

#include <okvis/IdProvider.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>

#include <okvis/assert_macros.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/ceres/CameraTimeParamBlock.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>

#include <vio/Sample.h>
#include <okvis/ceres/EuclideanParamBlock.hpp>

#include "io_wrap/StreamHelper.hpp"

#include "msckf/ImuOdometry.h"
#include "msckf/ImuSimulator.h"
#include "msckf/ProjParamOptModels.hpp"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/density.hpp>
#include <boost/accumulators/statistics/stats.hpp>


DECLARE_bool(use_mahalanobis);
DECLARE_int32(estimator_algorithm);

void initCameraNoiseParams(
    okvis::ExtrinsicsEstimationParameters* cameraNoiseParams,
    double sigma_abs_position) {
  cameraNoiseParams->sigma_absolute_translation = sigma_abs_position;
  cameraNoiseParams->sigma_absolute_orientation = 0;
  cameraNoiseParams->sigma_c_relative_translation = 0;
  cameraNoiseParams->sigma_c_relative_orientation = 0;

  cameraNoiseParams->sigma_focal_length = 5;
  cameraNoiseParams->sigma_principal_point = 5;
  cameraNoiseParams->sigma_distortion << 5e-2, 1e-2, 1e-3, 1e-3,
      1e-3;  /// k1, k2, p1, p2, [k3]
  cameraNoiseParams->sigma_td = 5e-3;
  cameraNoiseParams->sigma_tr = 5e-3;
}

/**
 * @brief initImuNoiseParams
 * @param imuParameters
 * @param addPriorNoise
 * @param sigma_bg std dev of initial gyroscope bias.
 * @param sigma_ba std dev of initial accelerometer bias.
 */
void initImuNoiseParams(
    okvis::ImuParameters* imuParameters, bool addPriorNoise,
    double sigma_bg, double sigma_ba, double std_Ta_elem,
    double sigma_td) {
  imuParameters->g = 9.81;
  imuParameters->a_max = 1000.0;
  imuParameters->g_max = 1000.0;
  imuParameters->rate = 100;

  imuParameters->sigma_g_c = 1.2e-3;
  imuParameters->sigma_a_c = 8e-3;
  imuParameters->sigma_gw_c = 2e-5;
  imuParameters->sigma_aw_c = 5.5e-5;
  imuParameters->tau = 600.0;

  imuParameters->sigma_bg = sigma_bg;
  imuParameters->sigma_ba = sigma_ba;

  // std for every element in shape matrix T_g
  imuParameters->sigma_TGElement = 5e-3;
  imuParameters->sigma_TSElement = 1e-3;
  imuParameters->sigma_TAElement = std_Ta_elem;
  imuParameters->model_type = "BG_BA_TG_TS_TA";

  Eigen::Matrix<double, 9, 1> eye;
  eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;

  if (addPriorNoise) {
    imuParameters->a0[0] = vio::gauss_rand(0, imuParameters->sigma_ba);
    imuParameters->a0[1] = vio::gauss_rand(0, imuParameters->sigma_ba);
    imuParameters->a0[2] = vio::gauss_rand(0, imuParameters->sigma_ba);
    imuParameters->g0[0] = vio::gauss_rand(0, imuParameters->sigma_bg);
    imuParameters->g0[1] = vio::gauss_rand(0, imuParameters->sigma_bg);
    imuParameters->g0[2] = vio::gauss_rand(0, imuParameters->sigma_bg);

    imuParameters->Tg0 =
        eye + vio::Sample::gaussian(imuParameters->sigma_TGElement, 9);
    imuParameters->Ts0 =
        vio::Sample::gaussian(imuParameters->sigma_TSElement, 9);
    imuParameters->Ta0 =
        eye + vio::Sample::gaussian(imuParameters->sigma_TAElement, 9);
    imuParameters->td0 =
        vio::gauss_rand(0, sigma_td);
  } else {
    imuParameters->a0.setZero();
    imuParameters->g0.setZero();

    imuParameters->Tg0 = eye;
    imuParameters->Ts0.setZero();
    imuParameters->Ta0 = eye;
    imuParameters->td0 = 0;
  }
}

void create_landmark_grid(
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
        *homogeneousPoints,
    std::vector<uint64_t> *lmIds, std::string pointFile = "") {
  //        const double xyLimit = 10, zLimit = 5,
  //            xyzIncrement = 0.5, offsetNoiseMag = 0.1;
  //        const double xyLimit = 5, zLimit = 2.5,
  //            xyzIncrement = 0.25, offsetNoiseMag = 0.05;
  const double xyLimit = 5, zLimit = 2.5, xyzIncrement = 0.5,
               offsetNoiseMag = 0.05;
  // four walls
  double x(xyLimit), y(xyLimit), z(zLimit);
  for (y = -xyLimit; y <= xyLimit; y += xyzIncrement) {
    for (z = -zLimit; z <= zLimit; z += xyzIncrement) {
      homogeneousPoints->push_back(
          Eigen::Vector4d(x + vio::gauss_rand(0, offsetNoiseMag),
                          y + vio::gauss_rand(0, offsetNoiseMag),
                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
      lmIds->push_back(okvis::IdProvider::instance().newId());
    }
  }

  for (y = -xyLimit; y <= xyLimit; y += xyzIncrement) {
    for (z = -zLimit; z <= zLimit; z += xyzIncrement) {
      homogeneousPoints->push_back(
          Eigen::Vector4d(y + vio::gauss_rand(0, offsetNoiseMag),
                          x + vio::gauss_rand(0, offsetNoiseMag),
                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
      lmIds->push_back(okvis::IdProvider::instance().newId());
    }
  }

  x = -xyLimit;
  for (y = -xyLimit; y <= xyLimit; y += xyzIncrement) {
    for (z = -zLimit; z <= zLimit; z += xyzIncrement) {
      homogeneousPoints->push_back(
          Eigen::Vector4d(x + vio::gauss_rand(0, offsetNoiseMag),
                          y + vio::gauss_rand(0, offsetNoiseMag),
                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
      lmIds->push_back(okvis::IdProvider::instance().newId());
    }
  }

  for (y = -xyLimit; y <= xyLimit; y += xyzIncrement) {
    for (z = -zLimit; z <= zLimit; z += xyzIncrement) {
      homogeneousPoints->push_back(
          Eigen::Vector4d(y + vio::gauss_rand(0, offsetNoiseMag),
                          x + vio::gauss_rand(0, offsetNoiseMag),
                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
      lmIds->push_back(okvis::IdProvider::instance().newId());
    }
  }
//  // top
//  z = zLimit;
//  for (y = -xyLimit; y <= xyLimit; y += xyzIncrement) {
//    for (x = -xyLimit; x <= xyLimit; x += xyzIncrement) {
//      homogeneousPoints->push_back(
//          Eigen::Vector4d(x + vio::gauss_rand(0, offsetNoiseMag),
//                          y + vio::gauss_rand(0, offsetNoiseMag),
//                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
//      lmIds->push_back(okvis::IdProvider::instance().newId());
//    }
//  }
//  // bottom
//  z = -zLimit;
//  for (y = -xyLimit; y <= xyLimit; y += xyzIncrement) {
//    for (x = -xyLimit; x <= xyLimit; x += xyzIncrement) {
//      homogeneousPoints->push_back(
//          Eigen::Vector4d(x + vio::gauss_rand(0, offsetNoiseMag),
//                          y + vio::gauss_rand(0, offsetNoiseMag),
//                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
//      lmIds->push_back(okvis::IdProvider::instance().newId());
//    }
//  }

  // save these points into file
  if (pointFile.size()) {
    std::ofstream pointStream(pointFile, std::ofstream::out);
    pointStream << "%id, x, y, z in the world frame " << std::endl;
    auto iter = homogeneousPoints->begin();
    for (auto it = lmIds->begin(); it != lmIds->end(); ++it, ++iter)
      pointStream << *it << " " << (*iter)[0] << " " << (*iter)[1] << " "
                  << (*iter)[2] << std::endl;
    pointStream.close();
    assert(iter == homogeneousPoints->end());
  }
}

typedef boost::iterator_range<std::vector<std::pair<double, double>>::iterator>
    HistogramType;

void outputFeatureHistogram(const std::string& featureHistFile,
                            const HistogramType& hist) {
  std::ofstream featureHistStream(featureHistFile, std::ios_base::out);
  double total = 0.0;
  featureHistStream << "Histogram of number of features in images (bin "
              << "lower bound, value)" << std::endl;
  for (size_t i = 0; i < hist.size(); i++) {
    featureHistStream << hist[i].first << " " << hist[i].second << std::endl;
    total += hist[i].second;
  }
  featureHistStream.close();
  EXPECT_NEAR(total, 1.0, 1e-5)
      << "Total of densities: " << total << " should be 1.";
}

class CameraSystemCreator {
 public:
  CameraSystemCreator(const int cameraModelId,
                      const std::string projIntrinsicRep,
                      const std::string extrinsicRep)
      : cameraModelId_(cameraModelId),
        projIntrinsicRep_(projIntrinsicRep),
        extrinsicRep_(extrinsicRep) {}

  void createDummyCameraSystem(
      std::shared_ptr<okvis::cameras::CameraBase>* cameraGeometry,
      std::shared_ptr<okvis::cameras::NCameraSystem>* cameraSystem) {
    Eigen::Matrix<double, 4, 4> matT_SC0 = create_T_SC(cameraModelId_, camIdx_);
    std::shared_ptr<const okvis::kinematics::Transformation> T_SC_0(
        new okvis::kinematics::Transformation(matT_SC0));
    cameraGeometry->reset(new okvis::cameras::PinholeCamera<
                          okvis::cameras::RadialTangentialDistortion>(
        0, 0, 0, 0, 0, 0,
        okvis::cameras::RadialTangentialDistortion(0.00, 0.00, 0.000, 0.000)));
    cameraSystem->reset(new okvis::cameras::NCameraSystem);
    (*cameraSystem)
        ->addCamera(
            T_SC_0, *cameraGeometry,
            okvis::cameras::NCameraSystem::DistortionType::RadialTangential,
            projIntrinsicRep_, extrinsicRep_);
  }

  void createNominalCameraSystem(
      std::shared_ptr<okvis::cameras::CameraBase>* cameraGeometry,
      std::shared_ptr<okvis::cameras::NCameraSystem>* cameraSystem) {
    Eigen::Matrix<double, 4, 4> matT_SC0 = create_T_SC(cameraModelId_, camIdx_);
    std::shared_ptr<const okvis::kinematics::Transformation> T_SC_0(
        new okvis::kinematics::Transformation(matT_SC0));

    *cameraGeometry = createCameraGeometry(cameraModelId_);

    cameraSystem->reset(new okvis::cameras::NCameraSystem);
    (*cameraSystem)
        ->addCamera(
            T_SC_0, *cameraGeometry,
            okvis::cameras::NCameraSystem::DistortionType::RadialTangential,
            projIntrinsicRep_, extrinsicRep_);
  }

  void createNoisyCameraSystem(
      std::shared_ptr<okvis::cameras::CameraBase>* cameraGeometry,
      std::shared_ptr<okvis::cameras::NCameraSystem>* cameraSystem,
      const okvis::ExtrinsicsEstimationParameters& cameraNoiseParams) {
    Eigen::Matrix<double, 4, 1> fcNoise = vio::Sample::gaussian(1, 4);
    fcNoise.head<2>() *= cameraNoiseParams.sigma_focal_length;
    fcNoise.tail<2>() *= cameraNoiseParams.sigma_principal_point;
    Eigen::Matrix<double, 4, 1> kpNoise = vio::Sample::gaussian(1, 4);
    for (int jack = 0; jack < 4; ++jack) {
      kpNoise[jack] *= cameraNoiseParams.sigma_distortion[jack];
    }
    Eigen::Vector3d p_CBNoise;
    for (int jack = 0; jack < 3; ++jack) {
      p_CBNoise[jack] =
          vio::gauss_rand(0, cameraNoiseParams.sigma_absolute_translation);
    }

    okvis::kinematics::Transformation ref_T_SC(create_T_SC(cameraModelId_, 0));
    std::shared_ptr<const okvis::kinematics::Transformation> T_SC_noisy(
        new okvis::kinematics::Transformation(
            ref_T_SC.r() - ref_T_SC.C() * p_CBNoise, ref_T_SC.q()));
    std::shared_ptr<const okvis::cameras::CameraBase> refCameraGeometry =
        createCameraGeometry(cameraModelId_);
    Eigen::VectorXd projDistortIntrinsics;
    refCameraGeometry->getIntrinsics(projDistortIntrinsics);
    cameraGeometry->reset(new okvis::cameras::PinholeCamera<
                          okvis::cameras::RadialTangentialDistortion>(
        refCameraGeometry->imageWidth(), refCameraGeometry->imageHeight(),
        projDistortIntrinsics[0] + fcNoise[0],
        projDistortIntrinsics[1] + fcNoise[1],
        projDistortIntrinsics[2] + fcNoise[2],
        projDistortIntrinsics[3] + fcNoise[3],
        okvis::cameras::RadialTangentialDistortion(kpNoise[0], kpNoise[1],
                                                   kpNoise[2], kpNoise[3])));
    cameraSystem->reset(new okvis::cameras::NCameraSystem);
    (*cameraSystem)
        ->addCamera(
            T_SC_noisy, *cameraGeometry,
            okvis::cameras::NCameraSystem::DistortionType::RadialTangential,
            projIntrinsicRep_, extrinsicRep_);
  }

 private:
  Eigen::Matrix<double, 4, 4> create_T_SC(const int caseId,
                                          const int /*camIdx*/) {
    Eigen::Matrix<double, 4, 4> matT_SC0;
    switch (caseId) {
      case 1: // sideways motion facing inward if the device goes straight forward
        matT_SC0 << 1, 0, 0, 0,
                    0, 0, 1, 0,
                    0, -1, 0, 0,
                    0, 0, 0, 1;
        break;
      case 2: // sideways motion facing outward if the device goes straight forward
        matT_SC0 << -1, 0, 0, 0,
                    0, 0, -1, 0,
                    0, -1, 0, 0,
                    0, 0, 0, 1;
        break;
      case 0: // forward motion if the device goes straight forward
      default:
        matT_SC0 << 0, 0, 1, 0,
                   -1, 0, 0, 0,
                   0, -1, 0, 0,
                   0, 0, 0, 1;
        break;
    }
    return matT_SC0;
  }

  std::shared_ptr<okvis::cameras::CameraBase> createCameraGeometry(
      const int caseId) {
    std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry;
    switch (caseId) {
      case 1:
        cameraGeometry.reset(new okvis::cameras::PinholeCamera<
                             okvis::cameras::RadialTangentialDistortion>(
            640, 480, 350, 350, 322, 238,
            okvis::cameras::RadialTangentialDistortion(0, 0, 0, 0)));
        break;

      case 0:
      default:
        cameraGeometry.reset(new okvis::cameras::PinholeCamera<
                             okvis::cameras::RadialTangentialDistortion>(
            752, 480, 350, 360, 378, 238,
            okvis::cameras::RadialTangentialDistortion(0.00, 0.00, 0.000,
                                                       0.000)));
        break;
    }
    return cameraGeometry;
  }

  static const okvis::cameras::NCameraSystem::DistortionType distortType_ =
      okvis::cameras::NCameraSystem::DistortionType::RadialTangential;
  static const std::string distortName_;
  static const int camIdx_ = 0;

  const int cameraModelId_;
  const std::string projIntrinsicRep_;
  const std::string extrinsicRep_;
};

const std::string CameraSystemCreator::distortName_ = "RadialTangentialDistortion";

/**
 * @brief compute_errors
 * @param estimator
 * @param T_WS
 * @param v_WS_true
 * @param ref_measurement
 * @param ref_camera_geometry
 * @param normalizedError nees in position, nees in orientation, nees in pose
 * @param rmsError rmse in xyz, \alpha, v_WS, bg, ba, Tg, Ts, Ta, p_CB,
 *  (fx, fy), (cx, cy), k1, k2, p1, p2, td, tr
 */
void compute_errors(
    const okvis::HybridFilter* estimator,
    const okvis::kinematics::Transformation& T_WS,
    const Eigen::Vector3d& v_WS_true,
    const okvis::ImuSensorReadings& ref_measurement,
    const std::shared_ptr<const okvis::cameras::CameraBase> ref_camera_geometry,
    const int projOptModelId,
    Eigen::Vector3d* normalizedError, Eigen::VectorXd* rmsError) {
  int projOptModelDim = okvis::ProjectionOptGetMinimalDim(projOptModelId);
  rmsError->resize(51 + projOptModelDim);

  okvis::kinematics::Transformation T_WS_est;
  uint64_t currFrameId = estimator->currentFrameId();
  estimator->get_T_WS(currFrameId, T_WS_est);
  Eigen::Vector3d delta = T_WS.r() - T_WS_est.r();
  Eigen::Vector3d alpha = vio::unskew3d(T_WS.C() * T_WS_est.C().transpose() -
                                        Eigen::Matrix3d::Identity());
  Eigen::Matrix<double, 6, 1> deltaPose;
  deltaPose << delta, alpha;

  (*normalizedError)[0] =
      delta.transpose() *
      estimator->covariance_.topLeftCorner<3, 3>().inverse() * delta;
  (*normalizedError)[1] = alpha.transpose() *
                          estimator->covariance_.block<3, 3>(3, 3).inverse() *
                          alpha;
  Eigen::Matrix<double, 6, 1> tempPoseError =
      estimator->covariance_.topLeftCorner<6, 6>().ldlt().solve(deltaPose);
  (*normalizedError)[2] = deltaPose.transpose() * tempPoseError;

  Eigen::Matrix<double, 9, 1> eye;
  eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  int index = 0;
  rmsError->head<3>() = delta.cwiseAbs2();
  index += 3;
  rmsError->segment<3>(index) = alpha.cwiseAbs2();
  index += 3;
  okvis::SpeedAndBias speedAndBias_est;
  estimator->getSpeedAndBias(currFrameId, 0, speedAndBias_est);
  Eigen::Vector3d deltaV = speedAndBias_est.head<3>() - v_WS_true;
  rmsError->segment<3>(index) = deltaV.cwiseAbs2();
  index += 3;
  rmsError->segment<3>(index) =
      (speedAndBias_est.segment<3>(3) - ref_measurement.gyroscopes).cwiseAbs2();
  index += 3;
  rmsError->segment<3>(index) =
      (speedAndBias_est.tail<3>() - ref_measurement.accelerometers).cwiseAbs2();
  index += 3;

  Eigen::Matrix<double, 9, 1> Tg_est;
  estimator->getSensorStateEstimateAs<okvis::ceres::ShapeMatrixParamBlock>(
      currFrameId, 0, okvis::HybridFilter::SensorStates::Imu,
      okvis::HybridFilter::ImuSensorStates::TG, Tg_est);

  rmsError->segment<9>(index) = (Tg_est - eye).cwiseAbs2();
  index += 9;
  Eigen::Matrix<double, 9, 1> Ts_est;
  estimator->getSensorStateEstimateAs<okvis::ceres::ShapeMatrixParamBlock>(
      currFrameId, 0, okvis::HybridFilter::SensorStates::Imu,
      okvis::HybridFilter::ImuSensorStates::TS, Ts_est);
  rmsError->segment<9>(index) = Ts_est.cwiseAbs2();
  index += 9;
  Eigen::Matrix<double, 9, 1> Ta_est;
  estimator->getSensorStateEstimateAs<okvis::ceres::ShapeMatrixParamBlock>(
      currFrameId, 0, okvis::HybridFilter::SensorStates::Imu,
      okvis::HybridFilter::ImuSensorStates::TA, Ta_est);
  rmsError->segment<9>(index) = (Ta_est - eye).cwiseAbs2();
  index += 9;
  Eigen::Matrix<double, 3, 1> p_CB_est;
  okvis::kinematics::Transformation T_SC_est;
  estimator->getSensorStateEstimateAs<okvis::ceres::PoseParameterBlock>(
      currFrameId, 0, okvis::HybridFilter::SensorStates::Camera,
      okvis::HybridFilter::CameraSensorStates::T_SCi, T_SC_est);
  p_CB_est = T_SC_est.inverse().r();
  rmsError->segment<3>(index) = p_CB_est.cwiseAbs2();
  index += 3;
  Eigen::VectorXd intrinsics_true;
  ref_camera_geometry->getIntrinsics(intrinsics_true);
  const int nDistortionCoeffDim =
      okvis::cameras::RadialTangentialDistortion::NumDistortionIntrinsics;
  Eigen::VectorXd distIntrinsic_true =
      intrinsics_true.tail<nDistortionCoeffDim>();

  Eigen::Matrix<double, Eigen::Dynamic, 1> projectionIntrinsic;
  if (projOptModelDim > 0) {
    estimator->getSensorStateEstimateAs<okvis::ceres::EuclideanParamBlock>(
        currFrameId, 0, okvis::HybridFilter::SensorStates::Camera,
        okvis::HybridFilter::CameraSensorStates::Intrinsic,
        projectionIntrinsic);
    Eigen::VectorXd local_opt_params;
    okvis::ProjectionOptGlobalToLocal(
        projOptModelId, intrinsics_true, &local_opt_params);

    rmsError->segment(index, projOptModelDim) =
        (projectionIntrinsic - local_opt_params).cwiseAbs2();
    index += projOptModelDim;
  }

  Eigen::Matrix<double, Eigen::Dynamic, 1> cameraDistortion_est(nDistortionCoeffDim);
  estimator->getSensorStateEstimateAs<okvis::ceres::EuclideanParamBlock>(
      currFrameId, 0, okvis::HybridFilter::SensorStates::Camera,
      okvis::HybridFilter::CameraSensorStates::Distortion,
      cameraDistortion_est);
  rmsError->segment(index, nDistortionCoeffDim) =
      (cameraDistortion_est - distIntrinsic_true).cwiseAbs2();
  index += nDistortionCoeffDim;

  double td_est(0.0), tr_est(0.0);
  estimator->getSensorStateEstimateAs<okvis::ceres::CameraTimeParamBlock>(
      currFrameId, 0, okvis::HybridFilter::SensorStates::Camera,
      okvis::HybridFilter::CameraSensorStates::TD, td_est);
  (*rmsError)[index] = td_est * td_est;
  ++index;

  estimator->getSensorStateEstimateAs<okvis::ceres::CameraTimeParamBlock>(
      currFrameId, 0, okvis::HybridFilter::SensorStates::Camera,
      okvis::HybridFilter::CameraSensorStates::TR, tr_est);
  (*rmsError)[index] = tr_est * tr_est;
  ++index;
}

void check_tail_mse(
    const Eigen::VectorXd& mse_tail, int projOptModelId) {
  int index = 0;
  EXPECT_LT(mse_tail.head<3>().norm(), std::pow(0.3, 2)) << "Position MSE";
  index = 3;
  EXPECT_LT(mse_tail.segment<3>(index).norm(), std::pow(0.08, 2)) << "Orientation MSE";
  index += 3;
  EXPECT_LT(mse_tail.segment<3>(index).norm(), std::pow(0.1, 2)) << "Velocity MSE";
  index += 3;
  EXPECT_LT(mse_tail.segment<3>(index).norm(), std::pow(0.002, 2)) << "Gyro bias MSE";
  index += 3;
  EXPECT_LT(mse_tail.segment<3>(index).norm(), std::pow(0.02, 2)) << "Accelerometer bias MSE";
  index += 3;
  EXPECT_LT(mse_tail.segment<9>(index).norm(), std::pow(4e-3, 2)) << "Tg MSE";
  index += 9;
  EXPECT_LT(mse_tail.segment<9>(index).norm(), std::pow(1e-3, 2)) << "Ts MSE";
  index += 9;
  EXPECT_LT(mse_tail.segment<9>(index).norm(), std::pow(5e-3, 2)) << "Ta MSE";
  index += 9;
  EXPECT_LT(mse_tail.segment<3>(index).norm(), std::pow(0.04, 2)) << "p_CS MSE";
  index += 3;
  int projIntrinsicDim = okvis::ProjectionOptGetMinimalDim(projOptModelId);
  EXPECT_LT(mse_tail.segment(index, projIntrinsicDim).norm(), std::pow(1, 2)) << "fxy cxy MSE";
  index += projIntrinsicDim;
  EXPECT_LT(mse_tail.segment<4>(index).norm(), std::pow(0.002, 2)) << "k1 k2 p1 p2 MSE";
  index += 4;
  EXPECT_LT(mse_tail.segment<2>(index).norm(), std::pow(1e-3, 2)) << "td tr MSE";
  index += 2;
}

void check_tail_nees(const Eigen::Vector3d &nees_tail) {
  EXPECT_LT(nees_tail[0], 8) << "Position NEES";
  EXPECT_LT(nees_tail[1], 5) << "Orientation NEES";
  EXPECT_LT(nees_tail[2], 10) << "Pose NEES";
}

okvis::Optimization initOptimizationConfig() {
  okvis::Optimization optConfig;
  optConfig.translationThreshold = 0.4;
  optConfig.rotationThreshold = 0.2618;
  optConfig.trackingRateThreshold = 0.5;
  optConfig.minTrackLength = 3u;
  return optConfig;
}

/**
 * @brief testHybridFilterSinusoid
 * @param outputPath
 * @param runs
 * @param trajectoryId: 0: yarn torus, 1: yarn ball, 2: rounded square,
 *     3: circle, 4: dot
 * @param cameraOrientation 0: forward 1: sideways inward 2: sidways outward
 *     Sideways orientation is geometrically favorable for motion estimation.
 */
void testHybridFilterSinusoid(const std::string& outputPath,
                              const int runs = 100,
                              const int trajectoryId=1,
                              const int cameraOrientation=0) {
  const double DURATION = 300.0;     // length of motion in seconds

  const double maxTrackLength = 60;  // maximum length of a feature track
  double imageNoiseMag = 1.0;        // pixel unit

  // definition of NEES in Huang et al. 2007 Generalized Analysis and
  // Improvement of the consistency of EKF-based SLAM
  // https://pdfs.semanticscholar.org/4881/2a9d4a2ae5eef95939cbee1119e9f15633e8.pdf
  // each entry, timestamp, nees in position, orientation, and pose, the
  // expected NEES is 6 for pose error, see Li ijrr high precision
  // nees for one run, neesSum for multiple runs
  std::vector<std::pair<okvis::Time, Eigen::Vector3d>> nees, neesSum;

  // each entry state timestamp, rmse in xyz, \alpha, v_WS, bg, ba, Tg, Ts, Ta,
  // p_CB, fx, fy, cx, cy, k1, k2, p1, p2, td, tr
  // rmse for one run, rmseSum for multiple runs
  std::vector<std::pair<okvis::Time, Eigen::VectorXd>,
      Eigen::aligned_allocator<std::pair<okvis::Time, Eigen::VectorXd>>>
      rmse, rmseSum;

  std::string neesFile = outputPath + "/sinusoidNEES.txt";
  std::string rmseFile = outputPath + "/sinusoidRMSE.txt";
  std::string truthFile = outputPath + "/sinusoidTruth.txt";
  std::ofstream truthStream;

  // number of features tracked in a frame
  boost::accumulators::accumulator_set<
      double, boost::accumulators::features<boost::accumulators::tag::count,
                                            boost::accumulators::tag::density>>
      frameFeatureTally(boost::accumulators::tag::density::num_bins = 20,
                        boost::accumulators::tag::density::cache_size = 40);
  std::string featureHistFile = outputPath + "/sinusoidFeatureHist.txt";

  okvis::timing::Timer filterTimer("msckf timer", true);

  std::string estimator_label = FLAGS_estimator_algorithm == 2 ? "PAVIO" : "MSCKF";
  LOG(INFO) << "Estimator algorithm: " << FLAGS_estimator_algorithm << " "
            << estimator_label << "\n";

  // only output the ground truth and data for the first successful trial
  bool bVerbose = false;
  int successRuns = 0;

  std::string projOptModelName = "FXY_CXY";
  std::string extrinsicModelName = "P_CS";

  for (int run = 0; run < runs; ++run) {
    bVerbose = successRuns == 0;
    filterTimer.start();

    srand((unsigned int)time(0)); // comment out to make tests deterministic
    imu::TestSetting cases[] = {
        imu::TestSetting(true, true, true, true)};  // noisy data, vins integration

    size_t c = 0;
    LOG(INFO) << "Run " << run << " " << cases[c].print();

    std::string pointFile = outputPath + "/sinusoidPoints.txt";
    std::string imuSampleFile = outputPath + "/sinusoidInertial.txt";
    std::ofstream inertialStream;
    if (bVerbose) {
      truthStream.open(truthFile, std::ofstream::out);
      truthStream << "%state timestamp, frameIdInSource, T_WS(xyz, xyzw), "
                     "v_WS, bg, ba, Tg, Ts, Ta, "
                     "p_CB, fx, fy, cx, cy, k1, k2, p1, p2, td, tr"
                  << std::endl;

      inertialStream.open(imuSampleFile, std::ofstream::out);
      inertialStream << "% timestamp, gx, gy, gz[rad/sec], acc x, acc y, acc "
                        "z[m/s^2], and noisy gxyz, acc xyz"
                     << std::endl;
    }

    std::stringstream ss;
    ss << run;
    std::string outputFile =
        outputPath + "/sinusoid_" + estimator_label + "_" + ss.str() + ".txt";
    std::string trackStatFile = outputPath + "/sinusoid_" + estimator_label +
                                "_trackstat_" + ss.str() + ".txt";

    double pCB_std = 2e-2;
    double ba_std = 2e-2;
    double Ta_std = 5e-3;

    okvis::ExtrinsicsEstimationParameters extrinsicsEstimationParameters;
    initCameraNoiseParams(&extrinsicsEstimationParameters, pCB_std);

    okvis::ImuParameters imuParameters;
    initImuNoiseParams(
        &imuParameters, cases[c].addPriorNoise, 5e-3, ba_std, Ta_std,
        extrinsicsEstimationParameters.sigma_td);

    okvis::InitialPVandStd pvstd;
    pvstd.std_p_WS = Eigen::Vector3d(1e-8, 1e-8, 1e-8);
    pvstd.std_q_WS = Eigen::Vector3d(1e-8, 1e-8, 1e-8);
    pvstd.std_v_WS = Eigen::Vector3d(5e-2, 5e-2, 5e-2);

    std::vector<okvis::kinematics::Transformation> ref_T_WS_list;
    std::vector<okvis::Time> times;
    const okvis::Time tStart(20);
    const okvis::Time tEnd(20 + DURATION);

    std::shared_ptr<imu::CircularSinusoidalTrajectory> cst;
    switch (trajectoryId) {
      case 0:
        cst.reset(new imu::TorusTrajectory(
            imuParameters.rate, Eigen::Vector3d(0, 0, -imuParameters.g)));
        break;
      case 2:
        cst.reset(new imu::RoundedSquare(
            imuParameters.rate, Eigen::Vector3d(0, 0, -imuParameters.g)));
        projOptModelName = "FIXED";
        break;
      case 3:
        cst.reset(new imu::RoundedSquare(
            imuParameters.rate, Eigen::Vector3d(0, 0, -imuParameters.g),
            okvis::Time(0, 0), 1.0, 0, 0.8));
        projOptModelName = "FIXED";
        break;
      case 4:
        cst.reset(new imu::RoundedSquare(
            imuParameters.rate, Eigen::Vector3d(0, 0, -imuParameters.g),
            okvis::Time(0, 0), 1e-3, 0, 0.8e-3));
        projOptModelName = "FIXED";
        break;
      case 1:
      default:
        cst.reset(new imu::SphereTrajectory(
            imuParameters.rate, Eigen::Vector3d(0, 0, -imuParameters.g)));
        break;
    }
    int projOptModelId = okvis::ProjectionOptNameToId(projOptModelName);

    cst->getTruePoses(tStart, tEnd, ref_T_WS_list);
    cst->getSampleTimes(tStart, tEnd, times);
    ASSERT_EQ(ref_T_WS_list.size(), times.size()) << "timestamps and true poses should have the same size!";
    okvis::ImuMeasurementDeque imuMeasurements;
    cst->getTrueInertialMeasurements(tStart - okvis::Duration(1),
                                    tEnd + okvis::Duration(1), imuMeasurements);
    okvis::ImuMeasurementDeque trueBiases;  // true biases used for computing RMSE

    if (cases[c].addImuNoise) {
      imu::addImuNoise(imuParameters, &imuMeasurements, &trueBiases,
                  bVerbose ? &inertialStream : nullptr);
    } else {
      trueBiases = imuMeasurements;
      for (size_t i = 0; i < imuMeasurements.size(); ++i) {
        trueBiases[i].measurement.gyroscopes.setZero();
        trueBiases[i].measurement.accelerometers.setZero();
      }
    }
    // remove the padding part of trueBiases to prepare for computing bias rmse
    auto tempIter = trueBiases.begin();
    for (; tempIter != trueBiases.end(); ++tempIter) {
      if (fabs((tempIter->timeStamp - times.front()).toSec()) < 1e-8) break;
    }
    trueBiases.erase(trueBiases.begin(), tempIter);
    // create the map
    std::shared_ptr<okvis::ceres::Map> mapPtr(new okvis::ceres::Map);

    CameraSystemCreator csc(cameraOrientation, projOptModelName, extrinsicModelName);
    // reference camera system
    std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry0;
    std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem0;
    csc.createNominalCameraSystem(&cameraGeometry0, &cameraSystem0);

    // dummy camera to keep camera info secret from the estimator
    std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry1;
    std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem1;
    csc.createDummyCameraSystem(&cameraGeometry1, &cameraSystem1);

    // camera system used for initilizing the estimator
    std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry2;
    std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem2;
    if (cases[c].addPriorNoise) {
      csc.createNoisyCameraSystem(&cameraGeometry2, &cameraSystem2,
                                  extrinsicsEstimationParameters);
    } else {
      csc.createNominalCameraSystem(&cameraGeometry2, &cameraSystem2);
    }

    std::ofstream debugStream;  // record state history of a trial
    if (!debugStream.is_open()) {
      debugStream.open(outputFile, std::ofstream::out);
      std::string headerLine;
      okvis::StreamHelper::composeHeaderLine(
            imuParameters.model_type,
            cameraSystem2->projOptRep(0),
            cameraSystem2->extrinsicOptRep(0),
            cameraSystem2->cameraGeometry(0)->distortionType(),
            okvis::FULL_STATE_WITH_ALL_CALIBRATION,
            &headerLine);
      debugStream << headerLine << std::endl;
    }

    double trNoisy(0);
    if (cases[c].addPriorNoise)
      trNoisy = vio::gauss_rand(0, extrinsicsEstimationParameters.sigma_tr);

    std::shared_ptr<okvis::HybridFilter> estimator;
    if (FLAGS_estimator_algorithm == 2) {
      estimator.reset(new okvis::TFVIO(mapPtr, trNoisy));
    } else {
      estimator.reset(new okvis::MSCKF2(mapPtr, trNoisy));
    }

    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
        homogeneousPoints;
    std::vector<uint64_t> lmIds;

    create_landmark_grid(&homogeneousPoints, &lmIds, bVerbose ? pointFile : "");

    estimator->addCamera(extrinsicsEstimationParameters);
    estimator->addImu(imuParameters);

    std::vector<uint64_t> multiFrameIds;

    size_t kale = 0;  // imu data counter
    bool bStarted = false;
    int k = -1;               // number of frames used in estimator
    int trackedFeatures = 0;  // feature tracks observed in a frame
    const int cameraIntervalRatio = 10; // number imu meas for 1 camera frame
    okvis::Time lastKFTime = times.front();
    okvis::ImuMeasurementDeque::const_iterator trueBiasIter =
        trueBiases.begin();
    nees.clear();
    rmse.clear();
    try {
      for (auto iter = times.begin(), iterEnd = times.end(); iter != iterEnd;
           iter += cameraIntervalRatio, kale += cameraIntervalRatio,
           trueBiasIter += cameraIntervalRatio) {
        okvis::kinematics::Transformation T_WS(ref_T_WS_list[kale]);
        // assemble a multi-frame
        std::shared_ptr<okvis::MultiFrame> mf(new okvis::MultiFrame);
        mf->setId(okvis::IdProvider::instance().newId());
        mf->setTimestamp(*iter);

        // reference ID will be and stay the first frame added.
        uint64_t id = mf->id();
        multiFrameIds.push_back(id);

        okvis::Time currentKFTime = *iter;
        okvis::Time imuDataEndTime = currentKFTime + okvis::Duration(1);
        okvis::Time imuDataBeginTime = lastKFTime - okvis::Duration(1);
        okvis::ImuMeasurementDeque imuSegment = okvis::getImuMeasurements(
            imuDataBeginTime, imuDataEndTime, imuMeasurements, nullptr);

        // add it in the window to create a new time instance
        if (bStarted == false) {
          bStarted = true;
          k = 0;
          mf->resetCameraSystemAndFrames(*cameraSystem2);
          okvis::kinematics::Transformation truePose =
              cst->computeGlobalPose(*iter);
          Eigen::Vector3d p_WS = truePose.r();
          Eigen::Vector3d v_WS = cst->computeGlobalLinearVelocity(*iter);

          pvstd.p_WS = p_WS;
          pvstd.q_WS = truePose.q();
          pvstd.v_WS = v_WS;

          if (cases[c].addPriorNoise) {
            //                p_WS += 0.1*Eigen::Vector3d::Random();
            v_WS += vio::Sample::gaussian(1, 3).cwiseProduct(pvstd.std_v_WS);
          }
          estimator->resetInitialPVandStd(pvstd, true);
          estimator->addStates(mf, imuSegment, true);

          ASSERT_EQ(estimator->covariance_.rows(),
                    60 + okvis::ProjectionOptGetMinimalDim(projOptModelId))
              << "Initial cov with one cloned state has a wrong dim";
        } else {
          // use the dummy because the estimator should no longer use info of the cameraSystem
          mf->resetCameraSystemAndFrames(*cameraSystem1);
          estimator->addStates(mf, imuSegment, true);
          ++k;
        }

        // add landmark observations
        trackedFeatures = 0;
        if (cases[c].useImageObservs) {
          size_t numFrames = mf->numFrames();
          std::vector<std::vector<size_t>> frame_landmark_index;
          std::vector<std::vector<cv::KeyPoint>> frame_keypoints;
          // project landmarks onto frames of mf
          for (size_t i = 0; i < mf->numFrames(); ++i) {
            std::vector<size_t> lmk_indices;
            std::vector<cv::KeyPoint> keypoints;
            for (size_t j = 0; j < homogeneousPoints.size(); ++j) {
              Eigen::Vector2d projection;
              Eigen::Vector4d point_C = cameraSystem0->T_SC(i)->inverse() *
                                        T_WS.inverse() * homogeneousPoints[j];
              okvis::cameras::CameraBase::ProjectionStatus status =
                  cameraSystem0->cameraGeometry(i)->projectHomogeneous(
                      point_C, &projection);
              if (status ==
                  okvis::cameras::CameraBase::ProjectionStatus::Successful) {
                Eigen::Vector2d measurement(projection);
                if (cases[c].addImageNoise) {
                  measurement[0] += vio::gauss_rand(0, imageNoiseMag);
                  measurement[1] += vio::gauss_rand(0, imageNoiseMag);
                }
                keypoints.emplace_back(measurement[0], measurement[1], 8.0);
                lmk_indices.emplace_back(j);
              }
            }
            frame_landmark_index.emplace_back(lmk_indices);
            frame_keypoints.emplace_back(keypoints);
            mf->resetKeypoints(i, keypoints);
          }

          // add landmark observations in the mf to the estimator
          for (size_t i = 0; i < numFrames; ++i) {
            const std::vector<size_t>& lmk_indices = frame_landmark_index[i];
            const std::vector<cv::KeyPoint>& keypoints = frame_keypoints[i];
            for (size_t k = 0; k < keypoints.size(); ++k) {
              size_t j = lmk_indices[k];
              size_t numObs = estimator->numObservations(lmIds[j]);
              if (numObs == 0) {
                // use dummy values to keep info secret from the estimator
                //                  estimator->addLandmark(lmIds[j],
                //                  homogeneousPoints[j]);
                Eigen::Vector4d unknown = Eigen::Vector4d::Zero();
                estimator->addLandmark(lmIds[j], unknown);
                estimator->addObservation<okvis::cameras::PinholeCamera<
                    okvis::cameras::RadialTangentialDistortion>>(
                    lmIds[j], mf->id(), i, k);
                ++trackedFeatures;
              } else {
                double sam = vio::Sample::uniform();
                if (sam > numObs / maxTrackLength) {
                  estimator->addObservation<okvis::cameras::PinholeCamera<
                      okvis::cameras::RadialTangentialDistortion>>(
                      lmIds[j], mf->id(), i, k);
                  ++trackedFeatures;
                }
              }
            }
          }
        }
        frameFeatureTally(trackedFeatures);

        estimator->optimize(1, 1, false);
        okvis::Optimization sharedOptConfig = initOptimizationConfig();
        estimator->setKeyframeRedundancyThresholds(
            sharedOptConfig.translationThreshold,
            sharedOptConfig.rotationThreshold,
            sharedOptConfig.trackingRateThreshold,
            sharedOptConfig.minTrackLength);
        okvis::MapPointVector removedLandmarks;
        estimator->applyMarginalizationStrategy(5, 25, removedLandmarks);
        estimator->print(debugStream);

        Eigen::Vector3d v_WS_true = cst->computeGlobalLinearVelocity(*iter);
        if (bVerbose) {
          Eigen::VectorXd allIntrinsics;
          cameraGeometry0->getIntrinsics(allIntrinsics);
          std::shared_ptr<const okvis::kinematics::Transformation> T_SC_0
              = cameraSystem0->T_SC(0);
          Eigen::IOFormat SpaceInitFmt(Eigen::StreamPrecision,
                                       Eigen::DontAlignCols, " ", " ", "", "",
                                       "", "");
          truthStream << *iter << " " << id << " " << std::setfill(' ')
                      << T_WS.parameters().transpose().format(SpaceInitFmt)
                      << " " << v_WS_true.transpose().format(SpaceInitFmt)
                      << " 0 0 0 0 0 0 "
                      << "1 0 0 0 1 0 0 0 1 "
                      << "0 0 0 0 0 0 0 0 0 "
                      << "1 0 0 0 1 0 0 0 1 "
                      << T_SC_0->inverse().r().transpose().format(SpaceInitFmt)
                      << " " << allIntrinsics.transpose().format(SpaceInitFmt)
                      << " 0 0" << std::endl;
        }

        Eigen::Vector3d normalizedError;
        Eigen::VectorXd rmsError;
        compute_errors(estimator.get(), T_WS, v_WS_true, trueBiasIter->measurement,
                       cameraGeometry0, projOptModelId, &normalizedError, &rmsError);
        nees.push_back(std::make_pair(*iter, normalizedError));
        rmse.push_back(std::make_pair(*iter, rmsError));
        lastKFTime = currentKFTime;
      }  // every keyframe

      if (neesSum.empty()) {
        neesSum = nees;
        rmseSum = rmse;
      } else {
        for (size_t jack = 0; jack < neesSum.size(); ++jack) {
          neesSum[jack].second += nees[jack].second;
          rmseSum[jack].second += rmse[jack].second;
        }
      }
      check_tail_mse(rmse.back().second, projOptModelId);
      check_tail_nees(nees.back().second);

      LOG(INFO) << "Run " << run << " finishes with last added frame " << k
                << " of tracked features " << trackedFeatures << std::endl;

      // output track length distribution
      std::ofstream trackStatStream(trackStatFile, std::ios_base::out);
      estimator->printTrackLengthHistogram(trackStatStream);
      trackStatStream.close();
      // end output track length distribution
      if (truthStream.is_open())
        truthStream.close();

      ++successRuns;
    } catch (std::exception &e) {
      if (truthStream.is_open()) truthStream.close();
      LOG(INFO) << "Run and last added frame " << run << " " << k << " "
                << e.what();
      // revert the accumulated errors and delete the corresponding file
      if (debugStream.is_open()) debugStream.close();
//      unlink(outputFile.c_str());
    }
    double elapsedTime = filterTimer.stop();
    LOG(INFO) << "Run " << run << " using time [sec] " << elapsedTime;
  }  // next run

  HistogramType hist = boost::accumulators::density(frameFeatureTally);
  outputFeatureHistogram(featureHistFile, hist);

  for (auto it = neesSum.begin(); it != neesSum.end(); ++it)
    (it->second) /= successRuns;
  std::ofstream neesStream;
  neesStream.open(neesFile, std::ofstream::out);
  neesStream << "%state timestamp, NEES of p_WS, \\alpha_WS, T_WS "
             << std::endl;
  for (auto it = neesSum.begin(); it != neesSum.end(); ++it)
    neesStream << it->first << " " << it->second.transpose() << std::endl;
  neesStream.close();

  EXPECT_GT(successRuns, 0)
      << "number of successful runs " << successRuns << " out of runs " << runs;
  for (auto it = rmseSum.begin(); it != rmseSum.end(); ++it)
    it->second = ((it->second) / successRuns).cwiseSqrt();

  std::ofstream rmseStream;
  rmseStream.open(rmseFile, std::ofstream::out);
  std::string headerLine;
  okvis::StreamHelper::composeHeaderLine(
        "BG_BA_TG_TS_TA", projOptModelName,
        extrinsicModelName, "RadialTangentialDistortion",
        okvis::FULL_STATE_WITH_ALL_CALIBRATION,
        &headerLine, false);
  rmseStream << headerLine << std::endl;
  for (auto it = rmseSum.begin(); it != rmseSum.end(); ++it)
    rmseStream << it->first << " " << it->second.transpose() << std::endl;
  rmseStream.close();
}

// FLAGS_log_dir can be passed in commandline as --log_dir=/some/log/dir
TEST(FILTER, MSCKF_BALL) {
  int32_t old_algorithm = FLAGS_estimator_algorithm;
  FLAGS_estimator_algorithm = 1;
  testHybridFilterSinusoid(FLAGS_log_dir, 5, 1);
  FLAGS_estimator_algorithm = old_algorithm;
}

TEST(FILTER, PAVIO_BALL) {
  int32_t old_algorithm = FLAGS_estimator_algorithm;
  FLAGS_estimator_algorithm = 2;
  testHybridFilterSinusoid(FLAGS_log_dir, 5, 1);
  FLAGS_estimator_algorithm = old_algorithm;
}

TEST(FILTER, MSCKF_TORUS) {
  int32_t old_algorithm = FLAGS_estimator_algorithm;
  FLAGS_estimator_algorithm = 1;
  testHybridFilterSinusoid(FLAGS_log_dir, 5, 0);
  FLAGS_estimator_algorithm = old_algorithm;
}

TEST(FILTER, PAVIO_TORUS) {
  int32_t old_algorithm = FLAGS_estimator_algorithm;
  FLAGS_estimator_algorithm = 2;
  testHybridFilterSinusoid(FLAGS_log_dir, 5, 0);
  FLAGS_estimator_algorithm = old_algorithm;
}

TEST(FILTER, MSCKF_SQUIRCLE) {
  int32_t old_algorithm = FLAGS_estimator_algorithm;
  FLAGS_estimator_algorithm = 1;
  testHybridFilterSinusoid(FLAGS_log_dir, 5, 2);
  FLAGS_estimator_algorithm = old_algorithm;
}

TEST(FILTER, PAVIO_SQUIRCLE) {
  int32_t old_algorithm = FLAGS_estimator_algorithm;
  FLAGS_estimator_algorithm = 2;
  testHybridFilterSinusoid(FLAGS_log_dir, 5, 2);
  FLAGS_estimator_algorithm = old_algorithm;
}

TEST(FILTER, MSCKF_CIRCLE) {
  int32_t old_algorithm = FLAGS_estimator_algorithm;
  FLAGS_estimator_algorithm = 1;
  testHybridFilterSinusoid(FLAGS_log_dir, 5, 3);
  FLAGS_estimator_algorithm = old_algorithm;
}

TEST(FILTER, PAVIO_CIRCLE) {
  int32_t old_algorithm = FLAGS_estimator_algorithm;
  FLAGS_estimator_algorithm = 2;
  testHybridFilterSinusoid(FLAGS_log_dir, 5, 3);
  FLAGS_estimator_algorithm = old_algorithm;
}

TEST(FILTER, MSCKF_DOT) {
  int32_t old_algorithm = FLAGS_estimator_algorithm;
  FLAGS_estimator_algorithm = 1;
  testHybridFilterSinusoid(FLAGS_log_dir, 5, 4);
  FLAGS_estimator_algorithm = old_algorithm;
}

TEST(FILTER, PAVIO_DOT) {
  int32_t old_algorithm = FLAGS_estimator_algorithm;
  FLAGS_estimator_algorithm = 2;
  testHybridFilterSinusoid(FLAGS_log_dir, 5, 4);
  FLAGS_estimator_algorithm = old_algorithm;
}
