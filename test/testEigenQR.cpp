#include <gtest/gtest.h>
#include <vector>
#include <Eigen/Geometry>
#include <iostream>

#include "vio/rand_sampler.h"
#include "vio/eigen_utils.h"
#include <okvis/timing/Timer.hpp>

TEST(Eigen, Quaternion)
{
    using namespace std;
    using namespace Eigen;
    std::vector<int> meas;
    for(int i=0; i<9; ++i)
        meas.push_back(i+1);
    meas.front()=meas.back();
    meas.resize(1);
    ASSERT_EQ(meas[0], 9);
    Quaterniond quat(0.2,0.5,-0.7,1.5);
    quat.normalize();
    
    Vector3d rotvec(3,-2,1);

    Vector3d rotres= quat._transformVector(rotvec);
    Vector3d rotres0(-1.16502, 0.442244, 3.52805);
    EXPECT_LT((rotres - rotres0).norm(), 1e-5);

    Quaterniond quat2(3,-2,5,0.7);
    quat2.normalize();
    Eigen::Matrix3d rot2 = quat2.toRotationMatrix();
    Vector3d euler=rotro2eu(quat2.toRotationMatrix());
    Eigen::Matrix3d rot0= roteu2ro(euler);
    EXPECT_LT((rot2 - rot0).norm(), 1e-8);
}

TEST(Eigen, RowColMajor)
{
    using namespace Eigen;
    Matrix<double, 3,5, Eigen::RowMajor> rand1;
    rand1<< 10, 20, 30,40,50,60,70,80,90,10,11,12,
            13,14,15;
    Matrix<double, 3,5, Eigen::ColMajor> rand2 = rand1;
    Matrix<double, 3,5, Eigen::ColMajor> rand20;
    rand20<< 10, 20, 30,40,50,60,70,80,90,10,11,12,
            13,14,15;
    Matrix<double, 3,2, Eigen::ColMajor> rand3 = rand1.block<3,2>(0,3);
    Matrix<double, 3,2, Eigen::ColMajor> rand30;
    rand30<< 40,50,90,10,14, 15;

    Matrix3d rand4= skew3d(rand1.col(4));
    Matrix3d rand40;
    rand40 << 0, -15, 10, 15, 0, -50, -10, 50, 0;
    ASSERT_LT((rand2 - rand20).norm(), 1e-8);
    ASSERT_LT((rand3 - rand30).norm(), 1e-8);
    ASSERT_LT((rand4 - rand40).norm(), 1e-8);
}

TEST(Eigen, doubleQR){
    MatrixXd Ad(MatrixXd::Random(10,3));
    Ad.setRandom();
    HouseholderQR<MatrixXd> qr(Ad); //don't use column pivoting because in that case Q*R-A!=0
    Eigen::MatrixXd Q = qr.householderQ();
    MatrixXd R = qr.matrixQR().triangularView<Upper>();
    ASSERT_TRUE( Ad.isApprox(Q*R));
    
    Eigen::MatrixXd nQ= nullspace(Ad);
    ASSERT_LT((nQ.transpose()*Ad).norm(), 1E-8);
}

TEST(Eigen, floatQR){
    MatrixXf A(MatrixXf::Random(10,3)), thinQ(MatrixXf::Identity(10,3)), Q, R;
    A.setRandom();
    
    HouseholderQR<MatrixXf> qr(A);
    Q = qr.householderQ();
    thinQ = qr.householderQ() * thinQ;
   
    MatrixXf thinQ2;
    thinQ2.setIdentity(10, 3);
    qr.householderQ().applyThisOnTheLeft(thinQ2);
    ASSERT_LT( (thinQ - thinQ2).norm(), 1e-10);
    
    R=qr.matrixQR().triangularView<Upper>();   
    MatrixXf nullQ= qr.householderQ();
    nullQ= nullQ.block(0,3,10,7).eval();
   
    ASSERT_LT( (nullQ.transpose()*A).norm(), 1e-6);

    MatrixXf y= MatrixXf(10,1);
    y<<1,2,3,4,5,6,7,8,9,10;
    MatrixXf x;
    A<< 10, 20, 30,40,50,60,70,80,90,10,11,12,
            13,14,15,16,17,18,19,20,121,122,123,124,
            125,126,127,128,129,130;
    x = A.householderQr().solve(y); 
    
    ASSERT_LT((y-A*x).norm(), 8);
}


//solve scales in pose graph optimization
TEST(Eigen, QRvsSVD){

    int m= 600, n=201, runs= 2;

    okvis::timing::Timer timer[3]={okvis::timing::Timer("svd", true),
okvis::timing::Timer("double_svd+qr", true),
okvis::timing::Timer("single_svd+qr", true)};
    double elapsedTime[3] ={0,0,0};
    for(int zinc=0; zinc< runs ;++ zinc){
    Eigen::MatrixXd sm1(m, n);
    Eigen::MatrixXf sf1(m,n);
        sm1.setZero();
        sf1.setZero();
        srand(time(NULL));
        int jack=0;
        for(int jack = 0; jack< n-1; ++ jack)
        {
            double sji= vio::gauss_rand(1, 0.05);
            sm1(jack, jack) = sji;
            sm1(jack, jack+1) = -1;
            sf1(jack, jack) = (float)sji;
            sf1(jack, jack+1) = -1.f;
        }
        for(; jack< m; ++ jack)
        {
            double sji= vio::gauss_rand(1, 0.1);
            int i= std::rand()%n;
            int j= std::rand()%n;
            while(i==j)
            {
                i= std::rand()%n;
                j= std::rand()%n;
            }
            sm1(jack, i) = sji;

            sm1(jack, j) = -1;
            sf1(jack, i) = (float)sji;
            sf1(jack, j) = -1.f;
        }
// method 1 svd double
        timer[0].start();
        Eigen::JacobiSVD<Eigen::MatrixXd > svd0(sm1, Eigen::ComputeThinV);
        Eigen::Matrix<double, Eigen::Dynamic, 1> allScales0(n, 1);
        allScales0= svd0.matrixV().block(0, n-1, n, 1);
        allScales0= allScales0/allScales0[0];
        double t1= timer[0].stop();
        elapsedTime[0]+=t1;

        // method 2: use double qr decomposition to speed up processing. I suspect this sacrifice much stability for a small speed improvement
        timer[1].start();
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(sm1);
        Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
        sm1= R.block(0, 0, n, n).eval();

        Eigen::JacobiSVD<Eigen::MatrixXd > svd(sm1, Eigen::ComputeThinV);
        Eigen::Matrix<double, Eigen::Dynamic, 1> allScales(n, 1);
        allScales= svd.matrixV().block(0, n-1, n, 1);
        allScales= allScales/allScales[0];
        double t2= timer[1].stop();

        elapsedTime[1]+=t2;
// emthod 3 qr+ svd float

        timer[2].start();
        Eigen::HouseholderQR<Eigen::MatrixXf> qr1(sf1);
        Eigen::MatrixXf R1 = qr1.matrixQR().triangularView<Eigen::Upper>();
        sf1= R1.block(0, 0, n, n).eval();

        Eigen::JacobiSVD<Eigen::MatrixXf > svd1(sf1, Eigen::ComputeThinV);
        Eigen::Matrix<double, Eigen::Dynamic, 1> allScales1(n, 1);
        allScales1= svd1.matrixV().block(0, n-1, n, 1).cast<double>();
        allScales1= allScales1/allScales1[0];
        double t3= timer[2].stop();
        elapsedTime[2]+=t3;

        //"value deviation of qr+svd to svd and float qr+svd to double svd:"
        EXPECT_LT((allScales0- allScales).norm(), 1E-10);
        EXPECT_LT((allScales1 - allScales).norm(), 1E-4);
    }
    cout<<"time difference svd - float (qr+svd) for each run :"<< (elapsedTime[0] - elapsedTime[2])/ runs<<endl;
    cout<<"cummulated time svd vs qr+svd vs float qr+svd "<< elapsedTime[0]<<" "<< elapsedTime[1]<<" "<< elapsedTime[2]<<endl;
}
