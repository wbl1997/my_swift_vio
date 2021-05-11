/**
 * @file Player.cpp
 * @brief Source file for the Player class.
 * @author Jianzhu Huai
 */

#include <functional>
#include <string>
#include <thread>
#include <vector>

#include <glog/logging.h>
#include <io_wrap/Player.hpp>

#define THRESHOLD_DATA_DELAY_WARNING 0.1  // in seconds

namespace swift_vio {
Player::~Player() {}

Player::Player(okvis::VioInterface *vioInterfacePtr,
               const okvis::VioParameters &params)
    : mbFinished(false),
      vioInterface_(vioInterfacePtr),
      vioParameters_(params),
      mVideoFile(params.input.videoFile),
      mImageFolder(params.input.imageFolder),
      mTimeFile(params.input.timeFile),
      mImuFile(params.input.imuFile) {
  if (mVideoFile.empty()) {  // image sequence input
    mIG = std::make_shared<vio::IMUGrabber>(mImuFile, vio::PlainText);
    mFG = std::make_shared<vio::FrameGrabber>(mImageFolder, mTimeFile,
                                              params.input.startIndex,
                                              params.input.finishIndex);
  } else {
    mIG = std::make_shared<vio::IMUGrabber>(mImuFile);
    mFG = std::make_shared<vio::FrameGrabber>(mVideoFile, mTimeFile,
                                              params.input.startIndex,
                                              params.input.finishIndex);
  }
}

void Player::Run() {
  ros::Rate rate(vioParameters_.sensors_information.cameraRate);
  LOG(INFO) << "camera frame rate "
            << vioParameters_.sensors_information.cameraRate;
  // + advance to retrieve a little more imu data so as to avoid waiting
  // in processing frames which causes false warning of delayed frames.
  const double advance = 0.5;
  cv::Mat frame;
  double frameTime;
  int frameCounter(0);
  int progressReportInterval = 300;  // in number of frames
  if (!mFG->is_open()) {
    LOG(WARNING) << "Frame grabber is not opened properly. Make sure its input "
                    "files are OK";
    return;
  }
  while (mFG->grabFrame(frame, frameTime) && frameTime > 0) {
    cv::Mat filtered;
    if (vioParameters_.optimization.useMedianFilter) {
      cv::medianBlur(frame, filtered, 3);
    } else {
      filtered = frame;
    }
    okvis::Time t(frameTime);
    t -= okvis::Duration(vioParameters_.sensors_information.imageDelay);
    if (frameCounter % progressReportInterval == 0) {
      LOG(INFO) << "read in frame at " << t
                << " id " << mFG->getCurrentId();
    }
    vioInterface_->addImage(t, 0, filtered, NULL, mFG->getCurrentId());

    // add corresponding imu data
    if (frameCounter == 0) {
      mIG->getObservation(t.toSec() -
                          0.1);  // 0.1 to avoid reading the first entry that
                                 // may be useful for later processing
    }

    bool isMeasGood = mIG->getObservation(t.toSec() + advance);
    if (!isMeasGood) {
      // the measurements can be bad when appraoching the end of a file
      ++frameCounter;
      rate.sleep();
      continue;
    }
    std::vector<Eigen::Matrix<double, 7, 1>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>>
        imuObservations = mIG->getMeasurement();

    imuObservations.pop_back();  // remove the first entry which was the last in
                                 // the previous observations

    for (const Eigen::Matrix<double, 7, 1> &obs : imuObservations) {
      vioInterface_->addImuMeasurement(okvis::Time(obs[0]), obs.segment<3>(1),
                                       obs.segment<3>(4));
    }
    ++frameCounter;
    rate.sleep();
  }
  LOG(INFO) << "frame grabber finishes.";
  mbFinished.store(true);
}

void Player::RunBlocking() {
  // + advance to retrieve a little more imu data so as to avoid waiting
  // in processing frames which causes false warning of delayed frames.
  const double advance = 0.5;
  cv::Mat frame;
  double frameTime;
  int frameCounter(0);
  int progressReportInterval = 300;  // in number of frames
  if (!mFG->is_open()) {
    LOG(WARNING) << "Frame grabber is not opened properly. Make sure its input "
                    "files are OK";
    return;
  }
  while (mFG->grabFrame(frame, frameTime) && frameTime > 0) {
    vioInterface_->display();
    cv::Mat filtered;
    if (vioParameters_.optimization.useMedianFilter) {
      cv::medianBlur(frame, filtered, 3);
    } else {
      filtered = frame;
    }
    okvis::Time t(frameTime);
    t -= okvis::Duration(vioParameters_.sensors_information.imageDelay);
    if (frameCounter % progressReportInterval == 0) {
      LOG(INFO) << "read in frame at " << t << " id " << mFG->getCurrentId();
    }
    vioInterface_->addImage(t, 0, filtered, NULL, mFG->getCurrentId());

    // add corresponding imu data
    if (frameCounter == 0) {
      mIG->getObservation(t.toSec() -
                          0.1);  // 0.1 is to avoid reading and discarding the first few entries that
                                 // may be useful for later processing.
    }

    bool isMeasGood = mIG->getObservation(t.toSec() + advance);
    if (!isMeasGood) {
      // the measurements can be bad when approaching the end of a file.
      ++frameCounter;
      continue;
    }
    std::vector<Eigen::Matrix<double, 7, 1>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>>
        imuObservations = mIG->getMeasurement();

    imuObservations.pop_back();  // remove the first entry which was the last in
                                 // the previous observations

    for (const Eigen::Matrix<double, 7, 1> &obs : imuObservations) {
      vioInterface_->addImuMeasurement(okvis::Time(obs[0]), obs.segment<3>(1),
                                       obs.segment<3>(4));
    }
    ++frameCounter;
  }
  LOG(INFO) << "frame grabber finishes.";
  mbFinished.store(true);
}

}  // namespace swift_vio
