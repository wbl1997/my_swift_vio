#include "gtest/gtest.h"

#include <Eigen/Core>
#include "swift_vio/memory.h"
#include "okvis/kinematics/Transformation.hpp"

#include "KimeraRPGO/RobustSolver.h"
#include "KimeraRPGO/SolverParams.h"
#include "loop_closure/GtsamWrap.hpp"

// test RPGO can deal with multiple odometry edges at once.
class RobustSolverTest {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  enum class TestType {
    Planar = 0,
    Cubic = 1,
  };
  RobustSolverTest() : kTranslationStd(0.1), kRotationStd(0.02) {}

  void SetUp(TestType type) {
    // we traverse the corners of a cube.
    // The orientation of the body frame at every corner:
    // the z axis points forward to the next corner to be visited
    // (denoted Z corner).
    // The other two axis points to the other two corners connected to the
    // current corner (denoted X and Y corners).
    // The world frame is at the body frame of the start corner.
    corners_.resize(8);
    corners_[0] << 0, 0, 0;  // A
    corners_[1] << 0, 0, 1;  // B
    corners_[2] << 0, 1, 1;  // C
    corners_[3] << 0, 1, 0;  // D
    corners_[4] << 1, 1, 0;  // E
    corners_[5] << 1, 1, 1;  // F
    corners_[6] << 1, 0, 1;  // G
    corners_[7] << 1, 0, 0;  // H

    // generate a path
    switch (type) {
      case TestType::Planar:
        visitedCornerIndices_ = "ABCFEH";
        // neighboring corners in x,y,z directions of the local frame.
        neighborCornerIndices_ =
            std::vector<std::string>{"HDB", "GAC", "BDF", "GCE", "FDH", "GEA"};
        break;
      case TestType::Cubic:
      default:
        visitedCornerIndices_ = "ABCD";
        // neighboring corners in x,y,z directions of the local frame.
        neighborCornerIndices_ =
            std::vector<std::string>{"HDB", "GAC", "FBD", "ECA"};
        break;
    }
    T_WB_list_.reserve(visitedCornerIndices_.size());
    for (size_t j = 0; j < visitedCornerIndices_.size(); ++j) {
      int cornerIndex = visitedCornerIndices_[j] - 'A';
      Eigen::Vector3d t_WB = corners_[cornerIndex];
      Eigen::Matrix3d R_WB;
      for (size_t i = 0; i < 3u; ++i) {
        int tipIndex = neighborCornerIndices_[j][i] - 'A';
        R_WB.col(i) = corners_[tipIndex] - t_WB;
      }
      T_WB_list_.emplace_back(t_WB, Eigen::Quaterniond(R_WB));
    }

    perturbed_T_WB_list_.resize(T_WB_list_.size());
    perturbed_T_WB_list_[0] = T_WB_list_.front();
    for (size_t j = 1; j < T_WB_list_.size(); ++j) {
      okvis::kinematics::Transformation T_PrevNew =
          T_WB_list_[j - 1].inverse() * T_WB_list_[j];
      T_PrevNew = perturbRelativePose(T_PrevNew);
      perturbed_T_WB_list_[j] = perturbed_T_WB_list_[j - 1] * T_PrevNew;
    }

    diagonalSigmas_.head<3>() << kRotationStd, kRotationStd, kRotationStd;
    diagonalSigmas_.tail<3>() << kTranslationStd, kTranslationStd,
        kTranslationStd;
  }

  okvis::kinematics::Transformation perturbRelativePose(
      const okvis::kinematics::Transformation& T) {
    okvis::kinematics::Transformation T_disturb;
    T_disturb.setRandom(kTranslationStd, kRotationStd);
    return T * T_disturb;
  }

  void run(bool usePriorFactor) {
    using namespace KimeraRPGO;
    RobustSolverParams params;
    params.setPcmSimple3DParams(3 * kTranslationStd, 3 * kRotationStd,
                                Verbosity::QUIET);
    std::vector<char> special_symbs{'l'};  // for landmarks
    params.specialSymbols = special_symbs;

    std::unique_ptr<RobustSolver> pgo =
        KimeraRPGO::make_unique<RobustSolver>(params);
    const gtsam::SharedNoiseModel noiseModel =
        gtsam::noiseModel::Diagonal::Sigmas(diagonalSigmas_);

    gtsam::NonlinearFactorGraph nfg;
    gtsam::Values est;

    // initialize first
    size_t priorFactorIncrement = usePriorFactor ? 1 : 0;
    gtsam::Key init_key_a = gtsam::Symbol('a', 0);
    gtsam::Values init_vals;
    const okvis::kinematics::Transformation& T_WNew = T_WB_list_[0];
    gtsam::Pose3 prior_pose = gtsam::Pose3(gtsam::Rot3(T_WNew.q()), T_WNew.r());
    init_vals.insert(init_key_a, prior_pose);
    gtsam::NonlinearFactorGraph priorFactorGraph;

    Eigen::Matrix<double, 6, 6> sqrtInfoR;
    sqrtInfoR = diagonalSigmas_.asDiagonal();
    gtsam::SharedNoiseModel priorNoiseModel =
        swift_vio::createRobustNoiseModelSqrtR(sqrtInfoR);

    priorFactorGraph.add(
        gtsam::PriorFactor<gtsam::Pose3>(init_key_a, prior_pose, priorNoiseModel));
    if (usePriorFactor) {
      pgo->update(priorFactorGraph, init_vals);
    } else {
      pgo->update(gtsam::NonlinearFactorGraph(), init_vals);
    }

    int tailPoseIndex = visitedCornerIndices_.size() - 1;
    okvis::kinematics::Transformation T_HeadTail =
        T_WB_list_[0].inverse() * T_WB_list_[tailPoseIndex];
    gtsam::Pose3 Head_T_Tail = swift_vio::GtsamWrap::toPose3(T_HeadTail);

    gtsam::Key head_pose_key = gtsam::Symbol('a', 0);
    gtsam::Key tail_pose_key = gtsam::Symbol('a', tailPoseIndex);

    nfg = pgo->getFactorsUnsafe();
    est = pgo->calculateEstimate();
    EXPECT_EQ(nfg.size(), size_t(priorFactorIncrement));
    EXPECT_EQ(est.size(), size_t(1));

    // add odometries.
    size_t numExpectedFactors = 0u;
    for (size_t i = 1; i < visitedCornerIndices_.size(); ++i) {
      numExpectedFactors = i < 2 ? 1u : size_t(i * 2 - 2);
      numExpectedFactors += priorFactorIncrement;
      {
        gtsam::Values odom_val;
        gtsam::NonlinearFactorGraph odom_factor;
        const okvis::kinematics::Transformation& T_WPrev =
            perturbed_T_WB_list_[i - 1];
        const okvis::kinematics::Transformation& T_WNew =
            perturbed_T_WB_list_[i];
        okvis::kinematics::Transformation T_PrevNew =
            T_WPrev.inverse() * T_WNew;
        gtsam::Pose3 odom =
            gtsam::Pose3(gtsam::Rot3(T_PrevNew.q()), T_PrevNew.r());

        gtsam::Key key_prev = gtsam::Symbol('a', i - 1);
        gtsam::Key key_new = gtsam::Symbol('a', i);

        odom_val.insert(key_new, swift_vio::GtsamWrap::toPose3(T_WNew));

        Eigen::Matrix<double, 6, 6> bfSqrtInfoR;
        bfSqrtInfoR = diagonalSigmas_.asDiagonal();
        gtsam::SharedNoiseModel bfNoiseModel = swift_vio::createRobustNoiseModelSqrtR(bfSqrtInfoR);
        odom_factor.add(gtsam::BetweenFactor<gtsam::Pose3>(key_prev, key_new,
                                                           odom, bfNoiseModel));
        bool runOpt = pgo->update(odom_factor, odom_val);
        nfg = pgo->getFactorsUnsafe();
        est = pgo->calculateEstimate();

        EXPECT_EQ(nfg.size(), numExpectedFactors);
        EXPECT_EQ(est.size(), size_t(i + 1));
        EXPECT_FALSE(runOpt);
      }
      // also add nonsequential odometry factors.
      if (i > 1) {
        gtsam::Values odom_val;
        gtsam::NonlinearFactorGraph odom_factor;
        const okvis::kinematics::Transformation& T_WPenult =
            perturbed_T_WB_list_[i - 2];
        const okvis::kinematics::Transformation& T_WNew =
            perturbed_T_WB_list_[i];
        okvis::kinematics::Transformation T_PenuNew =
            T_WPenult.inverse() * T_WNew;

        gtsam::Pose3 odom =
            gtsam::Pose3(gtsam::Rot3(T_PenuNew.q()), T_PenuNew.r());
        gtsam::Key key_penu = gtsam::Symbol('a', i - 2);
        gtsam::Key key_new = gtsam::Symbol('a', i);

        odom_factor.add(gtsam::BetweenFactor<gtsam::Pose3>(key_penu, key_new,
                                                           odom, noiseModel));
        bool runOpt = pgo->update(odom_factor, odom_val,
                                  FactorType::NONSEQUENTIAL_ODOMETRY);

        nfg = pgo->getFactorsUnsafe();
        est = pgo->calculateEstimate();
        EXPECT_EQ(nfg.size(), numExpectedFactors + 1);
        EXPECT_EQ(est.size(), size_t(i + 1));
        EXPECT_FALSE(runOpt);
      }
    }
    nfg = pgo->getFactorsUnsafe();
    est = pgo->calculateEstimate();
    EXPECT_EQ(nfg.size(), numExpectedFactors + 1);
    EXPECT_EQ(est.size(), visitedCornerIndices_.size());

    gtsam::Pose3 estimatedTailPose = est.at<gtsam::Pose3>(tail_pose_key);
    gtsam::Pose3 estimatedHeadPose = est.at<gtsam::Pose3>(head_pose_key);
    gtsam::Pose3 estimatedHead_T_Tail =
        estimatedHeadPose.inverse() * estimatedTailPose;

    std::pair<double, double> error = swift_vio::computeRotationAndTranslationErrors(
        estimatedHead_T_Tail, Head_T_Tail, true);
    LOG(INFO) << "Initial error " << error.first << " " << error.second;
//    EXPECT_GT(error.second, 2 * kTranslationStd);
//    EXPECT_GT(error.first, kRotationStd);

    LOG(INFO) << "Last correct pose\n"
              << T_WB_list_.back().T() << "\nInitial value\n"
              << est.at<gtsam::Pose3>(tail_pose_key).matrix();

    std::vector<std::pair<int, int>> loopPoseIndices{
        {0, visitedCornerIndices_.size() - 1},
        {1, visitedCornerIndices_.size() - 1}};

    for (const std::pair<int, int>& loopPair : loopPoseIndices) {
      gtsam::NonlinearFactorGraph lc_factors;
      gtsam::Key keyLoop = gtsam::Symbol('a', loopPair.first);
      gtsam::Key keyQuery = gtsam::Symbol('a', loopPair.second);

      gtsam::Values odom_val;
      gtsam::NonlinearFactorGraph odom_factor;
      const okvis::kinematics::Transformation& T_WBl =
          T_WB_list_[loopPair.first];
      const okvis::kinematics::Transformation& T_WBq =
          T_WB_list_[loopPair.second];
      okvis::kinematics::Transformation T_BlBq = T_WBl.inverse() * T_WBq;
      T_BlBq = perturbRelativePose(T_BlBq);
      gtsam::Pose3 odom = gtsam::Pose3(gtsam::Rot3(T_BlBq.q()), T_BlBq.r());

      lc_factors.add(gtsam::BetweenFactor<gtsam::Pose3>(keyLoop, keyQuery, odom,
                                                        noiseModel));

      bool runOpt = pgo->update(lc_factors, gtsam::Values());
      EXPECT_TRUE(runOpt);
    }

    nfg = pgo->getFactorsUnsafe();
    est = pgo->calculateEstimate();

    EXPECT_EQ(nfg.size(), numExpectedFactors + 3);
    EXPECT_EQ(est.size(), visitedCornerIndices_.size());
    LOG(INFO) << "Last optimized pose\n"
              << est.at<gtsam::Pose3>(tail_pose_key).matrix();
    estimatedTailPose = est.at<gtsam::Pose3>(tail_pose_key);
    estimatedHeadPose = est.at<gtsam::Pose3>(head_pose_key);
    estimatedHead_T_Tail = estimatedHeadPose.inverse() * estimatedTailPose;

    std::pair<double, double> finalError =
        swift_vio::computeRotationAndTranslationErrors(estimatedHead_T_Tail,
                                                 Head_T_Tail, true);
    LOG(INFO) << "Final error " << finalError.first << " " << finalError.second;
    EXPECT_LT(finalError.second, error.second);
    EXPECT_LT(finalError.first, error.first);
  }

 protected:
  Eigen::AlignedVector<Eigen::Vector3d>
      corners_;  ///< coordinates of cube corners in the world frame.
  std::string
      visitedCornerIndices_;  ///< indices of visited corners in corners_
  std::vector<std::string>
      neighborCornerIndices_;  ///< indices of neighboring X, Y, Z corners

  Eigen::AlignedVector<okvis::kinematics::Transformation>
      T_WB_list_;  ///< body to world transforms at every visited corner.
  Eigen::AlignedVector<okvis::kinematics::Transformation> perturbed_T_WB_list_;
  const double kTranslationStd;
  const double kRotationStd;
  Eigen::Matrix<double, 6, 1> diagonalSigmas_;
};

TEST(RobustSolverTest, MultiOdometryConstraints1) {
  RobustSolverTest rst;
  rst.SetUp(RobustSolverTest::TestType::Planar);
  rst.run(true);
}

TEST(RobustSolverTest, MultiOdometryConstraints2) {
  RobustSolverTest rst;
  rst.SetUp(RobustSolverTest::TestType::Planar);
  rst.run(false);
}

TEST(RobustSolverTest, MultiOdometryConstraints3) {
  RobustSolverTest rst;
  rst.SetUp(RobustSolverTest::TestType::Cubic);
  rst.run(true);
}

TEST(RobustSolverTest, MultiOdometryConstraints4) {
  RobustSolverTest rst;
  rst.SetUp(RobustSolverTest::TestType::Cubic);
  rst.run(false);
}
