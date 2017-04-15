#include <gtest/gtest.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
//forward distort and backward undistort the projection of a point
TEST(OpenCV, PointProjection)
{
    cv::Mat ginc(3,1,CV_32F);
    ginc.at<float>(0)= 9.8;
    ginc.at<float>(1)= 0;
    ginc.at<float>(2)= -0.0;

    cv::Mat mK, mDistCoef;

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    float fx=600, fy=600, cx=320, cy=240;
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = -0.1;
    DistCoef.at<float>(1) = 0.01;
    DistCoef.at<float>(2) = -0.00;
    DistCoef.at<float>(3) = 0.00;
    DistCoef.copyTo(mDistCoef);

    cv::normalize(ginc, ginc);
    cout<<"ginc:"<<ginc<<endl;

    cv::Mat norm_point(1,3,CV_32F), distort_point(1,2, CV_32F);
    cv::KeyPoint *keypoint=new cv::KeyPoint(100, 200, 7);
    float factor=max(fx*ginc.at<float>(0)+ cx*ginc.at<float>(2), fy*ginc.at<float>(1)+ cy*ginc.at<float>(2));
    cv::Mat rvec=cv::Mat::zeros(3,1, CV_32F);
    cv::Mat tvec=cv::Mat::zeros(3,1, CV_32F);
#if 0
    // one interface to use undistortPoints
    cv::Point2f src_point= keypoint->pt;
    Mat_<cv::Point2f> points(1,1);
    points(0)=src_point;
    cv::undistortPoints(points,points,mK,mDistCoef,cv::Mat(),mK);
    cout<<"rectified source point 1st way:"<<*points[0]<<endl;
    // one interface to projectPoints
    norm_point.at<float>(0)=( points[0]->x- cx)/ fx;
    norm_point.at<float>(1)=( points[0]->y - cy)/ fy;
    norm_point.at<float>(2)=1;

    norm_point+=(ginc*2/factor).t(); // how many pixel from p we expect p' to move
    vector<Point3f> norm_points;
    norm_points.push_back(Point3f(norm_point.at<float>(0), norm_point.at<float>(1), norm_point.at<float>(2)));
    vector<Point2f> distort_points;
    cout<<"normalized displaced point: "<<norm_point<<endl;
    cv::projectPoints(norm_points, rvec, tvec, mK, mDistCoef, distort_points);
    distort_point.at<float>(0)= distort_points[0].x;
    distort_point.at<float>(1)= distort_points[0].y;
    cout<<"Distorted 1st way:"<< distort_point<<endl;

    keypoint->angle = atan2(distort_point.at<float>(1)-keypoint->pt.y,
                            distort_point.at<float>(0)-keypoint->pt.x);
    cout<<"angle 1st way:"<< keypoint->angle<<endl;
#else
    // second interface to undistortPoints
    cv::Mat mat(1,2,CV_32F);
    mat.at<float>(0,0)=keypoint->pt.x;
    mat.at<float>(0,1)=keypoint->pt.y;
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    norm_point.at<float>(0)=( mat.at<float>(0,0)- cx)/ fx;
    norm_point.at<float>(1)=( mat.at<float>(0,1)- cy)/ fy;
    norm_point.at<float>(2)=1;
    norm_point+=(ginc*2/factor).t(); // how many pixel from p we expect p' to move
    // second interface to projectPoints
    //    norm_point.reshape(3);    // unnecessary
    //    distort_point.reshape(2); // unnecessary
    cv::projectPoints(norm_point, rvec, tvec, mK, mDistCoef, distort_point);
    cout<<"Distorted 2nd way:"<< distort_point<<endl;
    keypoint->angle = atan2(distort_point.at<float>(1)-keypoint->pt.y,
                            distort_point.at<float>(0)-keypoint->pt.x);
    cout<<"angle 1st way:"<< keypoint->angle<<endl;
#endif
}
