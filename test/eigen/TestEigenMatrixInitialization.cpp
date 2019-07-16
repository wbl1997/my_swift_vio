
#include <iostream>
#include<Eigen/Core>
#include<Eigen/Geometry>
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

TEST(EigenMatrix, AngleAxis){
    Eigen::Matrix3d m;
    Eigen::Vector3d axisAngles(0.1, 0.2, 0.3);
    axisAngles*=M_PI;
    m = Eigen::AngleAxisd(axisAngles[0], Eigen::Vector3d::UnitX())
            * Eigen::AngleAxisd(axisAngles[1], Eigen::Vector3d::UnitY())
            * Eigen::AngleAxisd(axisAngles[2], Eigen::Vector3d::UnitZ());
    Eigen::Vector3d ea = m.eulerAngles(0, 1, 2);
    ASSERT_NEAR((ea- axisAngles).lpNorm<Eigen::Infinity>(), 0, 1e-8);

    double theta= M_PI/6;
    Eigen::Matrix3d Rx = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitX()).toRotationMatrix();
    Eigen::Matrix3d Ry = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Matrix3d Rz = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::Matrix3d Rxs2f, Rys2f, Rzs2f; //rotation of angle theta from start frame to finish frame
    double st =sin(theta), ct = cos(theta);
    Rxs2f <<1, 0, 0, 0, ct, st, 0, -st, ct;
    Rys2f <<ct, 0, -st, 0,1,0, st, 0, ct;
    Rzs2f <<ct, st, 0, -st, ct, 0, 0,0,1;
    ASSERT_NEAR((Rx- Rxs2f.transpose()).lpNorm<Eigen::Infinity>(), 0, 1e-8);
    ASSERT_NEAR((Ry- Rys2f.transpose()).lpNorm<Eigen::Infinity>(), 0, 1e-8);
    ASSERT_NEAR((Rz- Rzs2f.transpose()).lpNorm<Eigen::Infinity>(), 0, 1e-8);

}
