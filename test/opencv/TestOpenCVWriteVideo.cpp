#include <gtest/gtest.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
TEST(OpenCV, WriteVideo)
{
    //read an image
    string filename="../opencv249/samples/cpp/board.jpg";
    string videoFile="tempVideo.avi";
    Mat image= imread(filename);
    //show the image on window
    imshow("My image",image);
    //wait key for 5000ms
    VideoWriter outputVideo;
    outputVideo.open(videoFile, CV_FOURCC('M','J','P','G'), 10, image.size(), true);
    if(!outputVideo.isOpened())
    {
        cerr<<"Video not opened!"<<endl;
        return;
    }
    for(int i=0; i<100; ++i)
        outputVideo<<image;
    outputVideo.release();
}
