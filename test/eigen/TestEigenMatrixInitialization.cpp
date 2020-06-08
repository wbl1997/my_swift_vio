
#include <iostream>
#include<Eigen/Core>
#include <gtest/gtest.h>

TEST(EigenMatrix, Initialization){
    using namespace std;

    Eigen::MatrixXd a = Eigen::MatrixXd::Random(3,2);
    Eigen::MatrixXd b = Eigen::MatrixXd::Random(2,4);
    Eigen::MatrixXd c = a*b;

    Eigen::MatrixXd cov;
    Eigen::Matrix3d cov3= Eigen::Matrix3d::Identity()*0.2;
    Eigen::Matrix2d cov2= Eigen::Matrix2d::Identity()*0.3;
    Eigen::Matrix<double, 3,2> cov32= Eigen::Matrix<double,3,2>::Random();
    ASSERT_LT((cov32.lpNorm<Eigen::Infinity>()), 1);
    //    cov.block<0,0,3,5>()<< cov3, cov32<<std::endl; //compiler error

    cov = Eigen::MatrixXd::Zero(5,5);//or  cov.resize(5,5);
    cov<<cov3, cov32, cov32.transpose(), cov2;
    ASSERT_NEAR((cov.block<3,3>(0,0).lpNorm<1>()), 0.6, 1e-8);
    ASSERT_NEAR((cov.block<2,2>(3,3).lpNorm<1>()), 0.6, 1e-8);

    cov<< cov3*cov.topLeftCorner<3,5>(), cov2* cov.bottomLeftCorner<2,5>();
    ASSERT_NEAR((cov.block<3,3>(0,0).lpNorm<1>()), 0.12, 1e-8);
    ASSERT_NEAR((cov.block<2,2>(3,3).lpNorm<1>()), 0.18, 1e-8);

    // after matrix.resize, its elements are eliminated
    Eigen::MatrixXd newCov(7,7);
    newCov<< cov, cov.topLeftCorner(5,2), cov.topLeftCorner(2,5), cov.topLeftCorner(2,2);
    cov = newCov;
    ASSERT_NEAR((cov.block<3,3>(0,0).lpNorm<1>()), 0.12, 1e-8);
    ASSERT_NEAR((cov.block<2,2>(3,3).lpNorm<1>()), 0.18, 1e-8);
    ASSERT_NEAR((cov.block<2,2>(5,5).lpNorm<1>()), 0.08, 1e-8);
    ASSERT_NEAR((cov.block<2,3>(5,0).lpNorm<1>()), 0.08, 1e-8);
    ASSERT_NEAR((cov.block<3,2>(0,5).lpNorm<1>()), 0.08, 1e-8);
}

TEST(EigenMatrix, VectorSize) {
  Eigen::Matrix<double, 8, 1> testVec;
  testVec << 1, 2, 3, 4, 5, 6, 7, 8;
  EXPECT_EQ(testVec.size(), 8);
  EXPECT_EQ(testVec.tail<4>().size(), 4);
}

TEST(EigenMatrix, RowMajor) {
  Eigen::Matrix<double, 1, 5, Eigen::RowMajor> row5;
  Eigen::Map<Eigen::Matrix<double, 1, 5, Eigen::RowMajor>> row5map(row5.data());
  for (int j = 0; j < row5map.size(); ++j) {
    row5map[j] = j + 1;
  }
  Eigen::Matrix<double, 1, 5> expectedRow5;
  expectedRow5 << 1, 2, 3, 4, 5;
  EXPECT_TRUE(expectedRow5.isApprox(row5, 1e-8));

//  Eigen::Matrix<double, 4, 1, Eigen::RowMajor> col4; // compile error
}

TEST(EigenMatrix, DynamicResize) {
  Eigen::Matrix<double, -1, 1> vec;
  EXPECT_EQ(vec.size(), 0);

  Eigen::MatrixXd mat;
  EXPECT_EQ(mat.size(), 0);

  mat.resize(0, 5);
  EXPECT_EQ(mat.rows(), 0);
  EXPECT_EQ(mat.cols(), 5);

  mat.resize(3, Eigen::NoChange);
  EXPECT_EQ(mat.rows(), 3);
  EXPECT_EQ(mat.cols(), 5);

  Eigen::VectorXd vec3(3, 1);
  vec3 << 3, 2, 1;
  Eigen::VectorXd oldvec3 = vec3;
  Eigen::VectorXd vec0;

  vec3.tail(0) = vec0;
  EXPECT_LT((vec3 - oldvec3).lpNorm<Eigen::Infinity>(), 1e-8);
}

TEST(EigenMatrix, setIdentity) {
  const double eps = 1e-8;
  Eigen::Matrix<double, 2, 4> m24;
  m24.setIdentity();
  Eigen::Matrix<double, 2, 4> n24 = Eigen::Matrix<double, 2, 4>::Identity();

  Eigen::Matrix<double, 2, 4> r24;
  r24.setZero();
  r24(0, 0) = 1;
  r24(1, 1) = 1;
  EXPECT_TRUE(r24.isApprox(m24, eps));
  EXPECT_TRUE(r24.isApprox(n24, eps));

  Eigen::Matrix<double, 4, 2> m42;
  m42.setIdentity();
  Eigen::Matrix<double, 4, 2> n42 = Eigen::Matrix<double, 4, 2>::Identity();

  Eigen::Matrix<double, 4, 2> r42;
  r42.setZero();
  r42(0, 0) = 1;
  r42(1, 1) = 1;
  EXPECT_TRUE(r42.isApprox(m42, eps));
  EXPECT_TRUE(r42.isApprox(n42, eps));
}
