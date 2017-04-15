#include <gtest/gtest.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <okvis/cameras/RadialTangentialDistortion.hpp>

#include <iostream>
#include <fstream>

using namespace std;

std::vector<cv::Point3f> Generate3DPoints(int npts)
{
    std::vector<cv::Point3f> points;
    float x,y,z;
    cv::RNG rng;
    for (int i=0; i<npts; ++i){
        x=rng.uniform(-0.5, 0.5);y=rng.uniform(-0.5, 0.5);z=rng.uniform(1, 15);
        points.push_back(cv::Point3f(x,y,z));
    }
    //    for(unsigned int i = 0; i < points.size(); ++i)
    //    {
    //        std::cout << points[i] << std::endl;
    //    }
    return points;
}
// this proves that opencv2.4 indeed used the inverse polynomial radial tangential distortion model which agrees with okvis,
// camera calibration toolbox in matlab, but it does not agree with the checkerboard calibration procedure in opencv
void testOpenCVDistortion(){
    const size_t NUM_POINTS = 100;

    // instantiate all possible versions of test cameras
    std::vector<std::shared_ptr<okvis::cameras::CameraBase> > cameras;
    cameras.push_back(
                okvis::cameras::PinholeCamera<
                okvis::cameras::RadialTangentialDistortion>::createTestObject());

    //opencv camera
    Eigen::VectorXd allIntrinsics;
    cameras[0]->getIntrinsics(allIntrinsics);
    std::cout <<"all intrinsics "<< allIntrinsics.transpose()<<std::endl;
    double K1[9]={allIntrinsics[0], 0, allIntrinsics[2], 0, allIntrinsics[1], allIntrinsics[3], 0,0,1};
    cv::Mat cameraMatrix(3,3,cv::DataType<double>::type, K1);
    std::cout << "Initial cameraMatrix: " << cameraMatrix << std::endl;
    cv::Mat distCoeffs(4,1,cv::DataType<double>::type);
    distCoeffs.at<double>(0) = allIntrinsics[4];
    distCoeffs.at<double>(1) = allIntrinsics[5];
    distCoeffs.at<double>(2) = allIntrinsics[6];
    distCoeffs.at<double>(3) = allIntrinsics[7];
    std::cout << "distCoeffs: " << distCoeffs << std::endl;

    //left image
    cv::Mat rvec(3,1,CV_64F);
    rvec.at<double>(0)=0;
    rvec.at<double>(1)=0;
    rvec.at<double>(2)=0;
    cv::Mat tvec(3,1,cv::DataType<double>::type);
    tvec.at<double>(0)=0;
    tvec.at<double>(1)=0;
    tvec.at<double>(2)=0;

    for (size_t c = 0; c < cameras.size(); ++c) {
        //std::cout << "Testing " << cameras.at(c)->type() << std::endl;
        // try quite a lot of points:
        for (size_t i = 0; i < NUM_POINTS; ++i) {
            // create a random point in the field of view:
            Eigen::Vector2d imagePoint = cameras.at(c)->createRandomImagePoint();

            // backProject
            Eigen::Vector3d ray;
            assert(cameras.at(c)->backProject(imagePoint, &ray));

            // randomise distance
            ray.normalize();
            Eigen::Vector2d normalizedPoint =  ray.head<2>()/ray[2];
            ray *= (0.2 + 8 * (Eigen::Vector2d::Random()[0] + 1.0));

            // project
            Eigen::Vector2d imagePoint2;
            Eigen::Matrix<double, 2, 3> J;
            Eigen::Matrix2Xd J_intrinsics;
            assert(cameras.at(c)->project(ray, &imagePoint2, &J, &J_intrinsics)
                        == okvis::cameras::CameraBase::ProjectionStatus::Successful);

            // check they are the same
            assert((imagePoint2 - imagePoint).norm() < 0.01);

            // check point Jacobian vs. NumDiff
            const double dp = 1.0e-7;
            Eigen::Matrix<double, 2, 3> J_numDiff;
            for (size_t d = 0; d < 3; ++d) {
                Eigen::Vector3d point_p = ray
                        + Eigen::Vector3d(d == 0 ? dp : 0, d == 1 ? dp : 0,
                                          d == 2 ? dp : 0);
                Eigen::Vector3d point_m = ray
                        - Eigen::Vector3d(d == 0 ? dp : 0, d == 1 ? dp : 0,
                                          d == 2 ? dp : 0);
                Eigen::Vector2d imagePoint_p;
                Eigen::Vector2d imagePoint_m;
                cameras.at(c)->project(point_p, &imagePoint_p);
                cameras.at(c)->project(point_m, &imagePoint_m);
                J_numDiff.col(d) = (imagePoint_p - imagePoint_m) / (2 * dp);
            }
            assert((J_numDiff - J).norm() < 0.0001);

            // check intrinsics Jacobian
            const int numIntrinsics = cameras.at(c)->noIntrinsicsParameters();
            Eigen::VectorXd intrinsics;
            cameras.at(c)->getIntrinsics(intrinsics);
            Eigen::Matrix2Xd J_numDiff_intrinsics;
            J_numDiff_intrinsics.resize(2,numIntrinsics);
            for (int d = 0; d < numIntrinsics; ++d) {
                Eigen::VectorXd di;
                di.resize(numIntrinsics);
                di.setZero();
                di[d] = dp;
                Eigen::Vector2d imagePoint_p;
                Eigen::Vector2d imagePoint_m;
                Eigen::VectorXd intrinsics_p = intrinsics+di;
                Eigen::VectorXd intrinsics_m = intrinsics-di;
                cameras.at(c)->projectWithExternalParameters(ray, intrinsics_p, &imagePoint_p);
                cameras.at(c)->projectWithExternalParameters(ray, intrinsics_m, &imagePoint_m);
                J_numDiff_intrinsics.col(d) = (imagePoint_p - imagePoint_m) / (2 * dp);
            }
            /*std::cout<<J_numDiff_intrinsics<<std::endl;
      std::cout<<"----------------"<<std::endl;
      std::cout<<J_intrinsics<<std::endl;
      std::cout<<"================"<<std::endl;*/
            assert((J_numDiff_intrinsics - J_intrinsics).norm() < 0.0001);

            std::vector<cv::Point3f> objectPoints(1, cv::Point3f(ray[0],ray[1],ray[2]));
            std::vector<cv::Point2f> imagePoints;
            cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);
            Eigen::Vector2d imagePointCV(imagePoints[0].x, imagePoints[0].y);
            std::cout <<"opencv image point "<< imagePointCV<< std::endl;
            std::cout <<"okvis image point "<< imagePoint<< std::endl;
            std::cout <<"norm of diff "<<(imagePointCV - imagePoint).norm()<<std::endl;
            assert((imagePointCV - imagePoint).norm() < 1e-4);

            cv::Mat obs(1,2,CV_32F);
            obs.at<float>(0,0)=imagePoints[0].x;
            obs.at<float>(0,1)=imagePoints[0].y;
            obs = obs.reshape(2);
            cv::Mat objectPoint;
            cv::undistortPoints(obs,objectPoint,cameraMatrix,distCoeffs,cv::Mat());
            Eigen::Vector2d objectPointEigen(objectPoint.at<float>(0),objectPoint.at<float>(1));
            std::cout <<"undistorted point "<< objectPointEigen.transpose()<<" and origin mappoint "<<normalizedPoint.transpose() <<std::endl;
            std::cout << "undistorted and original point diff "<< (normalizedPoint-objectPointEigen).norm() <<std::endl;
            assert((normalizedPoint-objectPointEigen).norm()<1e-3);

        }
    }
}
