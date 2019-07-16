#include <gtest/gtest.h>

#include <Eigen/Core>
#include <iostream>
#include <opencv/cv.h>

TEST(Eigen, Eigen2CV){
                //test eigen and cv convertion functions
                Eigen::Matrix2d eigenMat;
                eigenMat << 1, 2,
                        3, 4;

                Eigen::Vector3d trans(100, 200, 300);

                cv::Mat dest(2,2, CV_64F);
                cv::Mat destVec(3,1, CV_64F);
                eigen2cv(eigenMat, dest);
                eigen2cv(trans, destVec);
std::cout<<"Expected matrix "<< std::endl << "1, 2"<< std::endl<< "3, 4"<< std::endl;
std:;cout<<"Actual matrix "<< std::endl<< dest<<std::endl;

std::cout<<"Expected vector "<< std::endl << "100, 200, 300"<< std::endl;
std:;cout<<"Actual vector "<< std::endl<< destVec<<std::endl;

                dest.at<double>(0,0)=300;
                dest.at<double>(0,1)=500;
                dest.at<double>(1,0)=50;
                dest.at<double>(1,1)=200;

                destVec.at<double>(0)=4;
                destVec.at<double>(1)=5;
                destVec.at<double>(2)=6;
                cv2eigen(dest, eigenMat);
                cv2eigen(destVec, trans);

std::cout<<"Expected matrix "<< std::endl << "300, 500"<< std::endl<< "50, 200"<< std::endl;
std:;cout<<"Actual matrix "<< std::endl<< eigenMat<<std::endl;

std::cout<<"Expected vector "<< std::endl << "4, 5, 6"<< std::endl;
std:;cout<<"Actual vector "<< std::endl<< trans<<std::endl;

                //cv2eigen(destVec, eigenMat); //don't do this, although no error
}
