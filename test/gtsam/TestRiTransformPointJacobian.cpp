
/**
 * @file   TestRiTransformPointJacobian.h
 * @brief  unit tests of Jacobians for T * p, T^{-1} * p, and T_1^{a_1} * ... *
 * T_n^{a_2} * p. where p is a 4D homogeneous point, and T_i = Expmap(xi)
 * \hat{T}_i.
 * @author Jianzhu Huai
 */

#include "gtsam/RiTransformPointJacobian.h"

#include "gtsam/RiExtendedPose3.h"
#include "gtsam/base/numericalDerivative.h"

#include "gtest/gtest.h"

#include <msckf/MultipleTransformPointJacobian.hpp>

Eigen::Matrix<double, 4, 1> TpWrap(const gtsam::RiExtendedPose3& state,
                                   const Eigen::Matrix<double, 4, 1>& hp) {
  okvis::kinematics::Transformation T(state.position(),
                                      state.rotation().toQuaternion());
  gtsam::RiTransformPointJacobian tpj;
  tpj.initialize(T, hp);
  return tpj.evaluate();
}

Eigen::Matrix<double, 4, 1> invTpWrap(const gtsam::RiExtendedPose3& state,
                                      const Eigen::Matrix<double, 4, 1>& hp) {
  okvis::kinematics::Transformation T(state.position(),
                                      state.rotation().toQuaternion());
  gtsam::RiInverseTransformPointJacobian itpj;
  itpj.initialize(T, hp);
  return itpj.evaluate();
}

okvis::kinematics::Transformation toOkvisTransform(
    const gtsam::RiExtendedPose3& state) {
  return okvis::kinematics::Transformation(state.position(),
                                           state.rotation().toQuaternion());
}

/**
 * @brief cutoutVelocityPart
 * @param numericalJ dp/d[\phi, \delta v, \delta p]
 * @return dp/d[\phi, \delta p]
 */
Eigen::Matrix<double, 4, 6> cutoutVelocityPart(
    const Eigen::Matrix<double, 4, 9> numericalJ) {
  Eigen::Matrix<double, 4, 6> res;
  res << numericalJ.leftCols(3), numericalJ.rightCols(3);
  return res;
}

TEST(RiTransformPointJacobian, random) {
  gtsam::RiExtendedPose3 state;
  state.setRandom();
  Eigen::Matrix<double, 4, 1> hpB;
  hpB.setRandom();
  gtsam::RiTransformPointJacobian tpj;
  okvis::kinematics::Transformation T_WB(state.position(),
                                         state.rotation().toQuaternion());
  tpj.initialize(T_WB, hpB);
  Eigen::Matrix<double, 4, 6> jT;
  Eigen::Matrix<double, 4, 4> jp;
  tpj.dhpA_dT_AB(&jT);
  tpj.dhpA_dhpB(&jp);

  Eigen::Matrix<double, 4, 9> expectedjT =
      gtsam::numericalDerivative21<Eigen::Matrix<double, 4, 1>,
                                   gtsam::RiExtendedPose3,
                                   Eigen::Matrix<double, 4, 1>>(&TpWrap, state,
                                                                hpB);
  Eigen::Matrix<double, 4, 6> njT = cutoutVelocityPart(expectedjT);
  EXPECT_TRUE(gtsam::assert_equal(njT, jT));

  Eigen::Matrix<double, 4, 4> expectedjp =
      gtsam::numericalDerivative22<Eigen::Matrix<double, 4, 1>,
                                   gtsam::RiExtendedPose3,
                                   Eigen::Matrix<double, 4, 1>>(&TpWrap, state,
                                                                hpB);

  EXPECT_TRUE(gtsam::assert_equal(expectedjp, jp));
}

TEST(RiInverseTransformPointJacobian, random) {
  gtsam::RiExtendedPose3 state;
  state.setRandom();
  Eigen::Matrix<double, 4, 1> hpB;
  hpB.setRandom();
  gtsam::RiInverseTransformPointJacobian itpj;
  okvis::kinematics::Transformation T_BW(state.position(),
                                         state.rotation().toQuaternion());
  itpj.initialize(T_BW, hpB);
  Eigen::Matrix<double, 4, 6> jT;
  Eigen::Matrix<double, 4, 4> jp;
  itpj.dhpB_dT_AB(&jT);
  itpj.dhpB_dhpA(&jp);

  Eigen::Matrix<double, 4, 9> expectedjT =
      gtsam::numericalDerivative21<Eigen::Matrix<double, 4, 1>,
                                   gtsam::RiExtendedPose3,
                                   Eigen::Matrix<double, 4, 1>>(&invTpWrap,
                                                                state, hpB);
  Eigen::Matrix<double, 4, 6> njT = cutoutVelocityPart(expectedjT);
  EXPECT_TRUE(gtsam::assert_equal(njT, jT));

  Eigen::Matrix<double, 4, 4> expectedjp =
      gtsam::numericalDerivative22<Eigen::Matrix<double, 4, 1>,
                                   gtsam::RiExtendedPose3,
                                   Eigen::Matrix<double, 4, 1>>(&invTpWrap,
                                                                state, hpB);
  EXPECT_TRUE(gtsam::assert_equal(expectedjp, jp));
}

TEST(RiMultipleTransformPointJacobian, random) {
  gtsam::RiExtendedPose3 state_BC, state_WBj, state_WBa;
  state_BC.setRandom();
  state_WBj.setRandom();
  state_WBa.setRandom();
  Eigen::Matrix<double, 4, 1> hpCa;
  hpCa.setRandom();
  okvis::MultipleTransformPointJacobian mtpj;

  Eigen::AlignedVector<okvis::kinematics::Transformation> transforms{
      toOkvisTransform(state_BC), toOkvisTransform(state_WBj),
      toOkvisTransform(state_WBa), toOkvisTransform(state_BC)};
  std::vector<int> exponents{-1, -1, 1, 1};
  std::shared_ptr<okvis::TransformPointJacobian> tpj(
      new gtsam::RiTransformPointJacobian());
  std::shared_ptr<okvis::InverseTransformPointJacobian> itpj(
      new gtsam::RiInverseTransformPointJacobian());

  mtpj.initialize(transforms, exponents, hpCa, tpj, itpj);
  Eigen::Matrix<double, 4, 4> jp = mtpj.dp_dpoint();
  Eigen::AlignedVector<Eigen::Matrix<double, 4, 6>> jTList(4);
  for (size_t i = 0; i < transforms.size(); ++i) {
    jTList[i] = mtpj.dp_dT(i);
  }
  jTList[0] += jTList[3];

  boost::function<Eigen::Matrix<double, 4, 1>(
      const gtsam::RiExtendedPose3&, const gtsam::RiExtendedPose3&,
      const gtsam::RiExtendedPose3&, const Eigen::Matrix<double, 4, 1>&)>
      hpCj = [=](const gtsam::RiExtendedPose3& state_BC,
                 const gtsam::RiExtendedPose3& state_WBj,
                 const gtsam::RiExtendedPose3& state_WBa,
                 const Eigen::Matrix<double, 4, 1>& hpCa) {
        okvis::MultipleTransformPointJacobian mtpj;

        Eigen::AlignedVector<okvis::kinematics::Transformation> transforms{
            toOkvisTransform(state_BC), toOkvisTransform(state_WBj),
            toOkvisTransform(state_WBa), toOkvisTransform(state_BC)};
        std::vector<int> exponents{-1, -1, 1, 1};
        std::shared_ptr<okvis::TransformPointJacobian> tpj(
            new gtsam::RiTransformPointJacobian());
        std::shared_ptr<okvis::InverseTransformPointJacobian> itpj(
            new gtsam::RiInverseTransformPointJacobian());
        mtpj.initialize(transforms, exponents, hpCa, tpj, itpj);
        return mtpj.evaluate();
      };
  Eigen::AlignedVector<Eigen::Matrix<double, 4, 6>> njList(3);

  njList[0] = cutoutVelocityPart(
      gtsam::numericalDerivative41(hpCj, state_BC, state_WBj, state_WBa, hpCa));
  njList[1] = cutoutVelocityPart(
      gtsam::numericalDerivative42(hpCj, state_BC, state_WBj, state_WBa, hpCa));
  njList[2] = cutoutVelocityPart(
      gtsam::numericalDerivative43(hpCj, state_BC, state_WBj, state_WBa, hpCa));
  Eigen::Matrix<double, 4, 4> njp =
      gtsam::numericalDerivative44(hpCj, state_BC, state_WBj, state_WBa, hpCa);
  for (size_t i = 0; i < njList.size(); ++i) {
    EXPECT_TRUE(gtsam::assert_equal(njList[i], jTList[i]));
  }
  EXPECT_TRUE(gtsam::assert_equal(jp, njp));
}
