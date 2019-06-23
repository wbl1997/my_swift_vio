
#include "feature_tracker/TrackResultReader.h"
#include <iomanip>

namespace feature_tracker {
TrackResultReader::TrackResultReader(const std::string file) : stream(file) {
  if (!stream.is_open())
    std::cerr << "Cannot open feature track file at " << file << "."
              << std::endl;
  std::string tempStr;
  for (size_t jack = 0; jack < 4; ++jack)  // remove header
    getline(stream, tempStr);
}
TrackResultReader::~TrackResultReader() {
  if (stream.is_open()) {
    stream.close();
  }
}

bool TrackResultReader::getNextFrame(
    const double timeStamp, std::vector<cv::KeyPoint>& keypoints,
    std::vector<size_t>& mapPointIds,
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>&
        mapPointPositions,
    const size_t frameId) {
  const double epsilonTime = 1e-5;
  Eigen::Matrix<double, 7, 1> tq_wc;
  if (timeStamp - currentTime > epsilonTime) {
    while (!stream.eof()) {
      readKeyPointsFromStream(stream, keypoints_, mapPointIds_,
                              mapPointPositions_, currentTime, currentFrameId,
                              trackingStatus, tq_wc);

      if (std::fabs(currentTime - timeStamp) < epsilonTime) {
        //   assert(frameId == currentFrameId);
        assert(trackingStatus);
        break;
      } else if (currentTime - timeStamp >= epsilonTime) {
        std::cout << "Warn: no entry corresponding to time "
                  << std::setprecision(12) << timeStamp << std::endl;
        return false;
      }
    }
    if (stream.eof()) {
      return false;
    }
  } else if (std::fabs(timeStamp - currentTime) < epsilonTime) {
  } else {
    std::cout << " no valid keypoints for time " << std::setprecision(12)
              << timeStamp << " which is less than current time for keypoints "
              << currentTime << std::endl;
    return false;
  }

  keypoints = keypoints_;
  mapPointIds = mapPointIds_;
  mapPointPositions = mapPointPositions_;
  return true;
}

// TODO: handle the rare exception that only a fraction of all keypoints are
// logged
bool readKeyPointsFromStream(
    std::ifstream& kp_stream, std::vector<cv::KeyPoint>& keypoints,
    std::vector<size_t>& mapPointIds,
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>&
        mapPointPositions,
    double& timeStamp, size_t& frameId, int& status,
    Eigen::Matrix<double, 7, 1>& tq_wc) {
  size_t kpNum;
  // tq_wc; //xyz, qxyzw

  kp_stream >> timeStamp >> frameId >> status >> kpNum >> tq_wc[0] >>
      tq_wc[1] >> tq_wc[2] >> tq_wc[3] >> tq_wc[4] >> tq_wc[5] >> tq_wc[6];

  keypoints.resize(kpNum);
  size_t mpId;
  Eigen::Vector3d dummy3;
  if (status == 0) {
    for (size_t jack = 0; jack < kpNum; ++jack) {
      cv::KeyPoint& kp = keypoints[jack];
      kp_stream >> kp.pt.x >> kp.pt.y >> kp.size >> kp.angle >> kp.response >>
          kp.octave >> kp.class_id >> mpId >> dummy3[0] >> dummy3[1] >>
          dummy3[2];
    }
  } else {
    mapPointIds.resize(kpNum);
    mapPointPositions.resize(kpNum);
    for (size_t jack = 0; jack < kpNum; ++jack) {
      cv::KeyPoint& kp = keypoints[jack];
      kp_stream >> kp.pt.x >> kp.pt.y >> kp.size >> kp.angle >> kp.response >>
          kp.octave >> kp.class_id >> mpId >> dummy3[0] >> dummy3[1] >>
          dummy3[2];
      kp.size = kp.size * 8 / 31;  // HACK: to make it compatible with msckf2
                                   // optimizer observation noise
      mapPointIds[jack] = mpId;
      mapPointPositions[jack] = dummy3;
    }
  }
  return true;
}
}  // namespace feature_tracker
