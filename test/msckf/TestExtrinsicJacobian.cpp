#include <iostream>
#include "gtest/gtest.h"

#include <msckf/ExtrinsicModels.hpp>

Eigen::Vector3d worldToImageAIDP(const Eigen::Vector4d& ab1rho, const okvis::kinematics::Transformation& T_GBa,
                      const okvis::kinematics::Transformation& T_GBf, const okvis::kinematics::Transformation& T_BC) {
    Eigen::Vector3d pC_rho = ((T_GBf * T_BC).inverse() * (T_GBa * T_BC) * ab1rho).head<3>();
    return pC_rho;
}

Eigen::Vector3d worldToImage(const Eigen::Vector4d& xyz1,
                      const okvis::kinematics::Transformation& T_GBf, const okvis::kinematics::Transformation& T_BC) {
    Eigen::Vector3d pC = ((T_GBf * T_BC).inverse() * xyz1).head<3>();
    return pC;
}

TEST(MSCKF2, ExtrinsicJacobianP_CB) {

}

TEST(MSCKF2, ExtrinsicJacobianP_BC_R_BC) {

}
