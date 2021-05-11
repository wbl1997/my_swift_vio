#include <iostream>
#include "gtest/gtest.h"

#include <swift_vio/ExtrinsicModels.hpp>
#include <swift_vio/MSCKF.hpp>

class ExtrinsicJacobianTest : public ::testing::Test {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ExtrinsicJacobianTest() {
    for (size_t j = 0; j < numCameras_; ++j) {
      okvis::kinematics::Transformation T_BC;
      T_BC.setRandom(10.0, M_PI);
      T_BC_list_.push_back(T_BC);
    }
  }

  void numericJacobianMainCamera(
      size_t camExtrinsicModelId,
      Eigen::Matrix<double, 6, 6>* dT_BC_dExtrinsic) {
    for (size_t j = 0u; j < 6u; ++j) {
      Eigen::Matrix<double, 6, 1> deltaExtrinsic;
      deltaExtrinsic.setZero();
      deltaExtrinsic[j] = h_;
      okvis::kinematics::Transformation T_XC = T_BC_list_[0];
      swift_vio::ExtrinsicModelOplus(camExtrinsicModelId, deltaExtrinsic.data(),
                                 &T_XC);
      Eigen::Matrix<double, 6, 1> delta;
      delta.setZero();
      swift_vio::Extrinsic_p_BC_q_BC::ominus(T_BC_list_[0].parameters().data(),
                                  T_XC.parameters().data(), delta.data());
      dT_BC_dExtrinsic->col(j) = delta / h_;
    }
  }

  void numericJacobianSecondCamera(
      size_t camExtrinsicModelId, size_t mainCamExtrinsicModelId,
      Eigen::Matrix<double, 6, 6>* dT_BC_dSecondExt,
      Eigen::Matrix<double, 6, 6>* dT_BC_dMainExt) {
    okvis::kinematics::Transformation T_XC;
    if (camExtrinsicModelId == swift_vio::Extrinsic_p_C0C_q_C0C::kModelId) {
      T_XC = T_BC_list_[0].inverse() * T_BC_list_[1];
    } else {
      T_XC = T_BC_list_[1];
    }

    for (size_t j = 0u; j < 6u; ++j) {
      Eigen::Matrix<double, 6, 1> deltaExtrinsic;
      deltaExtrinsic.setZero();
      deltaExtrinsic[j] = h_;
      okvis::kinematics::Transformation T_XC_plus = T_XC;
      swift_vio::ExtrinsicModelOplus(camExtrinsicModelId, deltaExtrinsic.data(),
                                 &T_XC_plus);
      okvis::kinematics::Transformation T_BC_plus;
      if (camExtrinsicModelId == swift_vio::Extrinsic_p_C0C_q_C0C::kModelId) {
        T_BC_plus = T_BC_list_[0] * T_XC_plus;
      } else {
        T_BC_plus = T_XC_plus;
      }
      Eigen::Matrix<double, 6, 1> delta;
      swift_vio::Extrinsic_p_BC_q_BC::ominus(T_BC_list_[1].parameters().data(),
                                         T_BC_plus.parameters().data(),
                                         delta.data());
      dT_BC_dSecondExt->col(j) = delta / h_;
    }
    for (size_t j = 0u; j < 6u; ++j) {
      Eigen::Matrix<double, 6, 1> deltaExtrinsic;
      deltaExtrinsic.setZero();
      deltaExtrinsic[j] = h_;
      okvis::kinematics::Transformation T_XC_plus = T_BC_list_[0];
      swift_vio::ExtrinsicModelOplus(mainCamExtrinsicModelId, deltaExtrinsic.data(),
                                 &T_XC_plus);
      okvis::kinematics::Transformation T_BC_plus;
      if (camExtrinsicModelId == swift_vio::Extrinsic_p_C0C_q_C0C::kModelId) {
        T_BC_plus = T_XC_plus * T_XC;
      } else {
        T_BC_plus = T_BC_list_[1];
      }
      Eigen::Matrix<double, 6, 1> delta;
      delta.setZero();
      swift_vio::Extrinsic_p_BC_q_BC::ominus(T_BC_list_[1].parameters().data(),
                                  T_BC_plus.parameters().data(), delta.data());
      dT_BC_dMainExt->col(j) = delta / h_;
    }
  }

  void checkJacobian(size_t mainCamExtModelId, size_t secondCamExtModelId, size_t mainCamExtDim) {
    {
      std::vector<size_t> involvedCameraIndices{0u};
      Eigen::AlignedVector<Eigen::MatrixXd> dT_BCi_dExtrinsics;
      swift_vio::computeExtrinsicJacobians(
          T_BC_list_[0], T_BC_list_[0], mainCamExtModelId, mainCamExtModelId,
          &dT_BCi_dExtrinsics, &involvedCameraIndices, kMainCameraIndex);

      EXPECT_EQ(involvedCameraIndices.size(), 1u);
      EXPECT_EQ(involvedCameraIndices[0], 0u);
      EXPECT_EQ(dT_BCi_dExtrinsics.size(), 1u);
      Eigen::Matrix<double, 6, 6> numeric_dT_dMainExt;
      numericJacobianMainCamera(mainCamExtModelId, &numeric_dT_dMainExt);
      EXPECT_LT(
          (dT_BCi_dExtrinsics[0] - numeric_dT_dMainExt.leftCols(mainCamExtDim))
              .lpNorm<Eigen::Infinity>(),
          eps_)
          << "dT_BCi_dExtrinsics[0]\n"
          << dT_BCi_dExtrinsics[0] << "\nnumeric_dT_dMainExt\n"
          << numeric_dT_dMainExt.leftCols(mainCamExtDim);
    }
    {
      std::vector<size_t> involvedCameraIndices{1u};
      Eigen::AlignedVector<Eigen::MatrixXd> dT_BCi_dExtrinsics;
      swift_vio::computeExtrinsicJacobians(
          T_BC_list_[1], T_BC_list_[0], secondCamExtModelId, mainCamExtModelId,
          &dT_BCi_dExtrinsics, &involvedCameraIndices, kMainCameraIndex);

      if (secondCamExtModelId == swift_vio::Extrinsic_p_C0C_q_C0C::kModelId) {
        EXPECT_EQ(involvedCameraIndices.size(), 2u);
        EXPECT_EQ(involvedCameraIndices[0], 1u);
        EXPECT_EQ(involvedCameraIndices[1], 0u);
        EXPECT_EQ(dT_BCi_dExtrinsics.size(), 2u);
      } else {
        EXPECT_EQ(involvedCameraIndices.size(), 1u);
        EXPECT_EQ(involvedCameraIndices[0], 1u);
        EXPECT_EQ(dT_BCi_dExtrinsics.size(), 1u);
      }

      Eigen::Matrix<double, 6, 6> numeric_dT_dSecondExt;
      Eigen::Matrix<double, 6, 6> numeric_dT_dMainExt;
      numericJacobianSecondCamera(secondCamExtModelId, mainCamExtModelId,
                                  &numeric_dT_dSecondExt, &numeric_dT_dMainExt);
      EXPECT_LT((dT_BCi_dExtrinsics[0] -
                 numeric_dT_dSecondExt)
                    .lpNorm<Eigen::Infinity>(),
                eps_);
      if (dT_BCi_dExtrinsics.size() > 1u) {
        EXPECT_LT((dT_BCi_dExtrinsics[1] -
                   numeric_dT_dMainExt.leftCols(mainCamExtDim))
                      .lpNorm<Eigen::Infinity>(),
                  5 * eps_)
            << "dT_BCi_dExtrinsics[1]\n"
            << dT_BCi_dExtrinsics[1]
            << "\nnumeric_dT_dMainExt.leftCols(mainCamExtDim)\n"
            << numeric_dT_dMainExt.leftCols(mainCamExtDim);
      }
    }
  }

 protected:
  const size_t numCameras_ = 2u;
  const size_t kMainCameraIndex = 0u;
  Eigen::AlignedVector<okvis::kinematics::Transformation> T_BC_list_;
  okvis::kinematics::Transformation T_C0C_list_;
  const double eps_ = 1e-6;
  const double h_ = 1e-6;
};

TEST_F(ExtrinsicJacobianTest, P_CB_T_BC) {
  checkJacobian(swift_vio::Extrinsic_p_CB::kModelId,
                swift_vio::Extrinsic_p_BC_q_BC::kModelId,
                swift_vio::Extrinsic_p_CB::kNumParams);
}

TEST_F(ExtrinsicJacobianTest, P_CB_T_C0C) {
  checkJacobian(swift_vio::Extrinsic_p_CB::kModelId,
                swift_vio::Extrinsic_p_C0C_q_C0C::kModelId,
                swift_vio::Extrinsic_p_CB::kNumParams);
}

TEST_F(ExtrinsicJacobianTest, T_BC_T_BC) {
  checkJacobian(swift_vio::Extrinsic_p_BC_q_BC::kModelId,
                swift_vio::Extrinsic_p_BC_q_BC::kModelId,
                swift_vio::Extrinsic_p_BC_q_BC::kNumParams);
}

TEST_F(ExtrinsicJacobianTest, T_BC_T_C0C) {
  checkJacobian(swift_vio::Extrinsic_p_BC_q_BC::kModelId,
                swift_vio::Extrinsic_p_C0C_q_C0C::kModelId,
                swift_vio::Extrinsic_p_BC_q_BC::kNumParams);
}
