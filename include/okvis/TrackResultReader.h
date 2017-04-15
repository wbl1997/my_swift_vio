
#ifndef TRACKRESULTREADER_H
#define TRACKRESULTREADER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/features2d/features2d.hpp>

class TrackResultReader{
public:
    TrackResultReader(const std::string file);

    ~TrackResultReader();

    bool getNextFrame(const double timeStamp, std::vector<cv::KeyPoint> & keypoints,  std::vector<size_t>& mapPointIds,
                      std::vector<Eigen::Vector3d>& mapPointPositions, const size_t frameId);

    double currentTime;
    size_t currentFrameId;
    int trackingStatus; //0 failed 1 successful
    std::vector<cv::KeyPoint> keypoints_; //keypoints in the current image
    std::vector<size_t> mapPointIds_; //map point ids in the current image , if not empty should of the same length as keypoints
    std::vector<Eigen::Vector3d> mapPointPositions_; //map point positions in the current image, if not empty should of the same length as keypoints

private:
    std::ifstream stream; //sensor data stream
};

bool readKeyPointsFromStream(std::ifstream& kp_stream, std::vector<cv::KeyPoint> &keypoints,
                             std::vector<size_t> &mapPointIds, std::vector<Eigen::Vector3d> &mapPointPositions,
                             double & timeStamp, size_t & frameId, int & status, Eigen::Matrix<double,7,1> & tq_wc);


#endif
