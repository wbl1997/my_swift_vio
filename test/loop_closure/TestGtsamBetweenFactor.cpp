#include "gtest/gtest.h"

#include "loop_closure/GtsamWrap.hpp"
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

TEST(BetweenFactor, JacobianToCustomRetract) {
  const double eps = 1e-6;
  const double h = 1e-6;
  gtsam::SharedNoiseModel noiseOdom =
      gtsam::noiseModel::Isotropic::Variance(6, 0.1);
  gtsam::Pose3 Tx(gtsam::Rot3(Eigen::Quaterniond::UnitRandom()),
                  Eigen::Vector3d::Random());
  gtsam::Pose3 Ty(gtsam::Rot3(Eigen::Quaterniond::UnitRandom()),
                  Eigen::Vector3d::Random());
  gtsam::Pose3 Tz = Tx.inverse() * Ty;
  gtsam::BetweenFactor<gtsam::Pose3> bf(0, 1, Tz, noiseOdom);
  Eigen::VectorXd residual = bf.evaluateError(Tx, Ty);
  EXPECT_LT(residual.lpNorm<Eigen::Infinity>(), eps);

  Eigen::Matrix3d analJToCustomRetract = -Tz.rotation().matrix().transpose();
  Eigen::Matrix<double, 6, 6> numericJ;
  for (int i = 0; i < 6; ++i) {
    Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
    delta[i] = h;
    gtsam::Pose3 Tzp = VIO::GtsamWrap::retract(Tz, delta);
    gtsam::BetweenFactor<gtsam::Pose3> bfp(0, 1, Tzp, noiseOdom);
    Eigen::VectorXd residualp = bfp.evaluateError(Tx, Ty);
    numericJ.col(i) = (residualp - residual) / h;
  }
  EXPECT_LT((numericJ.topRightCorner<3, 3>() - analJToCustomRetract).lpNorm<Eigen::Infinity>(),
            eps) << "numericJ.topRightCorner<3, 3>()\n" << numericJ.topRightCorner<3, 3>()
      << "\nanalJToCustomRetract\n" << analJToCustomRetract;
  EXPECT_LT((numericJ.topLeftCorner<3, 3>()).lpNorm<Eigen::Infinity>(), eps)
      << "numericJ.topLeftCorner<3, 3>()\n" << numericJ.topLeftCorner<3, 3>();
  EXPECT_LT((numericJ.bottomLeftCorner<3, 3>() - analJToCustomRetract)
                .lpNorm<Eigen::Infinity>(),
            eps);
  EXPECT_LT((numericJ.bottomRightCorner<3, 3>()).lpNorm<Eigen::Infinity>(),
            eps);

  // Test the Jacobians in a bundle.
  Eigen::Matrix<double, 6, 6> J6;
  VIO::GtsamWrap::toMeasurementJacobianBetweenFactor(Tz, &J6);
  EXPECT_LT((numericJ - J6).lpNorm<Eigen::Infinity>(), eps);
}


TEST(BetweenFactor, JacobianToCustomRetractRandomObs) {
  const double eps = 1e-6;
  const double h = 1e-6;
  gtsam::SharedNoiseModel noiseOdom =
      gtsam::noiseModel::Isotropic::Variance(6, 0.1);
  gtsam::Pose3 Tx(gtsam::Rot3(Eigen::Quaterniond::UnitRandom()),
                  Eigen::Vector3d::Random());
  gtsam::Pose3 Ty(gtsam::Rot3(Eigen::Quaterniond::UnitRandom()),
                  Eigen::Vector3d::Random());

  gtsam::Pose3 Tz = Tx.inverse() * Ty;
  Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Random();
  delta.head<3>() *= (5 * M_PI / 180);
  Tz = gtsam::Pose3::Expmap(delta) * Tz;

  gtsam::BetweenFactor<gtsam::Pose3> bf(0, 1, Tz, noiseOdom);
  Eigen::VectorXd residual = bf.evaluateError(Tx, Ty);

  Eigen::Matrix<double, 6, 6> numericJ;
  for (int i = 0; i < 6; ++i) {
    Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
    delta[i] = h;
    gtsam::Pose3 Tzp = VIO::GtsamWrap::retract(Tz, delta);
    gtsam::BetweenFactor<gtsam::Pose3> bfp(0, 1, Tzp, noiseOdom);
    Eigen::VectorXd residualp = bfp.evaluateError(Tx, Ty);
    numericJ.col(i) = (residualp - residual) / h;
  }

  Eigen::Matrix<double, 6, 6> analJ;
  VIO::GtsamWrap::toMeasurementJacobianBetweenFactor(Tz, Tx, Ty, &analJ);
//  Eigen::Matrix<double, 6, 6> diff = numericJ - analJ;
//  EXPECT_LT((diff.topRightCorner<3, 3>()).lpNorm<Eigen::Infinity>(), eps)
//      << "numericJ.topRightCorner<3, 3>()\n"
//      << numericJ.topRightCorner<3, 3>() << "\nanalJ.topRightCorner<3, 3>()\n"
//      << analJ.topRightCorner<3, 3>();
//  EXPECT_LT((diff.topLeftCorner<3, 3>()).lpNorm<Eigen::Infinity>(), eps)
//      << "numericJ.topLeftCorner<3, 3>()\n"
//      << numericJ.topLeftCorner<3, 3>() << "\nanalJ.topLeftCorner<3, 3>()\n"
//      << analJ.topLeftCorner<3, 3>();
//  EXPECT_LT((diff.bottomLeftCorner<3, 3>()).lpNorm<Eigen::Infinity>(), eps)
//      << "numericJ.bottomLeftCorner<3, 3>()\n"
//      << numericJ.bottomLeftCorner<3, 3>()
//      << "\nanalJ.bottomLeftCorner<3, 3>()\n"
//      << analJ.bottomLeftCorner<3, 3>();
//  EXPECT_LT((diff.bottomRightCorner<3, 3>()).lpNorm<Eigen::Infinity>(), eps)
//      << "numericJ.bottomRightCorner<3, 3>()\n"
//      << numericJ.bottomRightCorner<3, 3>()
//      << "\nanalJ.bottomRightCorner<3, 3>()\n"
//      << analJ.bottomRightCorner<3, 3>();

  // analytic jacobian
  VIO::BetweenFactorPose3Wrap bfWrap(Tz, Tx, Ty);
  Eigen::Matrix<double, 6, 6, Eigen::RowMajor> autoJ;
  Eigen::Matrix<double, 6, 1> autoResidual;
  bfWrap.toMeasurmentJacobian(&autoJ, &autoResidual);
  EXPECT_LT((residual - autoResidual).lpNorm<Eigen::Infinity>(), eps)
      << "ref residual\n" << residual.transpose()
      << "\nautoResidual\n" << autoResidual.transpose();

  EXPECT_LT((numericJ - autoJ).lpNorm<Eigen::Infinity>(), eps)
      << "autoJ\n" << autoJ << "\nnumericJ\n" << numericJ
      << "\ndiff\n" << (numericJ - autoJ);
}

TEST(PriorFactor, definition) {
  gtsam::SharedNoiseModel noise =
      gtsam::noiseModel::Isotropic::Variance(6, 0.1);
  gtsam::Pose3 Tz(gtsam::Rot3(Eigen::Quaterniond::UnitRandom()),
                  Eigen::Vector3d::Random());
  gtsam::Pose3 Tx(gtsam::Rot3(Eigen::Quaterniond::UnitRandom()),
                  Eigen::Vector3d::Random());
  gtsam::Pose3 Ty(gtsam::Rot3(Eigen::Quaterniond::UnitRandom()),
                  Eigen::Vector3d::Random());
  gtsam::Pose3 Txy = Tx.inverse() * Ty;

  gtsam::PriorFactor<gtsam::Pose3> pf(gtsam::Symbol(0), Tz, noise);
  Eigen::VectorXd rp = pf.evaluateError(Txy);
  gtsam::BetweenFactor<gtsam::Pose3> bf(0, 1, Tz, noise);
  Eigen::VectorXd rb = bf.evaluateError(Tx, Ty);
  EXPECT_LT((rp - rb).lpNorm<Eigen::Infinity>(), 1e-8);
}

TEST(RobustNoiseModel, sqrtInfoR) {
  Eigen::Matrix<double, 6, 6> sqrtInfoR;
  Eigen::Matrix<double, 6, 1> sigmas;
  sigmas << 0.01, 0.01, 0.01, 0.1, 0.1, 0.1;
  sqrtInfoR = sigmas.asDiagonal();

  // add a prior factor and between factor to a graph
  using namespace gtsam;
  Key firstKey = 0;
  Key secondKey = 1;
  Values initial;
  initial.insert(firstKey, Pose3());
  initial.insert(secondKey, Pose3());

  // Add prior on the first key
  NonlinearFactorGraph graphWithPrior;

  {
    SharedNoiseModel priorModel = VIO::createRobustNoiseModelSqrtR(sqrtInfoR);

    graphWithPrior.add(PriorFactor<Pose3>(firstKey, Pose3(), priorModel));

    sqrtInfoR << 5.03229,-0.0712813,0.0235663,-0.0258729,00-1.23496,0-0.329889
        ,0,4.81623,1.03861,1.23393,0.00811096,0.0864412
        ,0,0,1.46313,0.0854986,0-0.236276,-0.0539238
        ,0,0,0,-0.0529346,0.276297,-0.0294727
        ,0,0,0,0.321847,0,0.289415
        ,0,0,0,0,0,0.298538;

    sqrtInfoR << 0.01,0,0,0,0,0
        ,0,0.01,0,0,0,0
        ,0,0,0.01,0,0,0
        ,0,0,0,0.1,0,0
        ,0,0,0,0,0.1,0
        ,0,0,0,0,0,0.1;

    SharedNoiseModel noiseModel = VIO::createRobustNoiseModelSqrtR(sqrtInfoR);
    NonlinearFactorGraph nfg;
    nfg.add(BetweenFactor<Pose3>(firstKey, secondKey, Pose3(), noiseModel));

    gtsam::BetweenFactor<gtsam::Pose3> castFactor =
        *boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(
            nfg[0]);

    gtsam::Pose3 pose = castFactor.measured();
    auto castNoiseModel = boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(
        castFactor.noiseModel());
    EXPECT_EQ(castNoiseModel.get(), nullptr);
//    gtsam::Matrix covar = castNoiseModel->covariance();

    graphWithPrior.add(castFactor);
  }

  GaussNewtonParams params;
  params.setVerbosity("TERMINATION"); // this will show info about stopping conditions
  GaussNewtonOptimizer optimizer(graphWithPrior, initial, params);
  Values result = optimizer.optimize();

  EXPECT_LT(graphWithPrior.error(initial), 1e-7) << "initial error";
  EXPECT_LT(graphWithPrior.error(result), 1e-7) << "final error";
}

TEST(GtsamUtil, covariancebvx2xvb) {
  srand(1000000);

  // Create random covariance
  gtsam::Matrix cov_xx = gtsam::Matrix::Random(6, 6);
  gtsam::Matrix cov_xv = gtsam::Matrix::Random(6, 3);
  gtsam::Matrix cov_xb = gtsam::Matrix::Random(6, 6);
  gtsam::Matrix cov_vv = gtsam::Matrix::Random(3, 3);
  gtsam::Matrix cov_vb = gtsam::Matrix::Random(3, 6);
  gtsam::Matrix cov_bb = gtsam::Matrix::Random(6, 6);

  gtsam::Matrix cov_vx = cov_xv.transpose();
  gtsam::Matrix cov_bx = cov_xb.transpose();
  gtsam::Matrix cov_bv = cov_vb.transpose();

  gtsam::Matrix expected_cov_xvb = gtsam::Matrix(15, 15);
  expected_cov_xvb.block<6, 15>(0, 0) << cov_xx, cov_xv, cov_xb;
  expected_cov_xvb.block<3, 15>(6, 0) << cov_vx, cov_vv, cov_vb;
  expected_cov_xvb.block<6, 15>(9, 0) << cov_bx, cov_bv, cov_bb;

  gtsam::Matrix cov_bvx = gtsam::Matrix(15, 15);
  cov_bvx.block<6, 15>(0, 0) << cov_bb, cov_bv, cov_bx;
  cov_bvx.block<3, 15>(6, 0) << cov_vb, cov_vv, cov_vx;
  cov_bvx.block<6, 15>(9, 0) << cov_xb, cov_xv, cov_xx;

  gtsam::Matrix cov_actual_xvb = VIO::Covariance_bvx2xvb(cov_bvx);
  EXPECT_LE((expected_cov_xvb - cov_actual_xvb).lpNorm<Eigen::Infinity>(), 1e-8);
}
