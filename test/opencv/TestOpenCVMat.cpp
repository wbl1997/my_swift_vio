#include <gtest/gtest.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <fstream>

#include <Eigen/Core>

using namespace std;

class Trump
{
public:
    Trump(cv::Mat initValue, int i=0):myse3(initValue.clone()), id(i)
    {
        cout<<"Trump() "<<i<<endl;
    }
    Trump(const Trump & rhs): myse3(rhs.myse3), id(rhs.id){cout<<"Trump(const Trump) "<<id<<endl;}
    cv::Mat myse3;
    int id;
};


TEST(OpenCV,MatCopy)
{
    int nmatches=5;
    cv::Mat descriptors1(nmatches, 3, CV_8U), descriptors2(nmatches-2, 3, CV_8U), descriptors3(nmatches*2, 3, CV_8U);
    cv::RNG rng(0);
    rng.fill(descriptors1, cv::RNG::UNIFORM, 0, 255);

    for(int i=0; i<nmatches-2; i++)
    {
        descriptors2.row(i)= descriptors1.row(nmatches-i-1);
        //        descriptors1.row(nmatches-i-1).copyTo(descriptors2.row(i));
    }
    descriptors3=descriptors1;
    cout<<"descp1"<<descriptors1<<endl<<"descp2"<<endl;
    cout<<descriptors2<<endl<<"descp3"<<endl;
    cout<<descriptors3<<endl;
}

TEST(OpenCV, Mat)
{

    cv::Mat arc= (Mat_<double>(2,2)<<1,2,3,4);
    cout<<"initialize a Mat with an array, Mat.empty() "<<arc.empty()<<endl;
    arc.release();
    cout<<"after release the Mat, Mat.empty() "<<arc.empty()<<endl;
    
    Vector3d a(1,2,3);
    a.normalize();
    cout<<a<<endl;
    cv::Mat ginw=cv::Mat::zeros(3,1, CV_32F);
    string strSettingPath="ScaViSLAM/data/viso2settings.yaml";
    cv::FileStorage mfsSettings(strSettingPath, cv::FileStorage::READ);
    cv::Mat gw;
    mfsSettings["gw"]>>ginw;
    mfsSettings["gw"]>>gw;
    //    cout<<"not iniiated gw:"<<gw<<endl;
    //    cout<<"iniiated ginw:"<<ginw<<endl;
   
    Eigen::Matrix4d em=  Eigen::Matrix4d::Random();
    cv::Mat matTemp;
    cv::eigen2cv(em, matTemp);
    cout<<"eigen mat:"<<endl<<em<<endl;
    cout<<"cv mat:"<<endl<<matTemp<<endl;
    cv::Mat matTemp2(4,4, CV_32F);
    cv::eigen2cv(em, matTemp2);
    //    cout<<"cv float mat:"<<endl<<matTemp2<<endl;
    cout<< matTemp.type() << " "<< matTemp2.type()<<endl;
    Trump trump(matTemp);
    cv::eigen2cv(em, trump.myse3);

    cout<<"cv mat in class:"<<endl<<trump.myse3<<endl;
    Eigen::Matrix4f se3f;
    cv::cv2eigen(trump.myse3, se3f);
    cout<<"eigen mat f:"<<endl<<se3f<<endl;
  
    cv::Mat Tcw= cv::Mat::eye(4,4,CV_32F);
    cout<<Tcw<<endl;
    cv::Mat Tcw2;
    Tcw.copyTo(Tcw2);
    cout<<Tcw2<<endl;
}


