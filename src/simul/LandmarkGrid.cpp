#include "okvis/IdProvider.hpp"
#include "simul/LandmarkGrid.h"
#include "vio/Sample.h"

namespace simul {
void saveLandmarkGrid(
    const std::vector<Eigen::Vector4d,
                      Eigen::aligned_allocator<Eigen::Vector4d>>
        &homogeneousPoints,
    const std::vector<uint64_t> &lmIds, std::string pointFile) {
  if (pointFile.size()) {
    std::ofstream pointStream(pointFile, std::ofstream::out);
    pointStream << "%id, x, y, z in the world frame " << std::endl;
    auto iter = homogeneousPoints.begin();
    for (auto it = lmIds.begin(); it != lmIds.end(); ++it, ++iter)
      pointStream << *it << " " << (*iter)[0] << " " << (*iter)[1] << " "
                  << (*iter)[2] << std::endl;
    pointStream.close();
    assert(iter == homogeneousPoints.end());
  }
}

void createBoxLandmarkGrid(
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
        *homogeneousPoints,
    std::vector<uint64_t> *lmIds, double halfz, double addFloorCeling) {
  const double xyLimit = 5, zLimit = halfz, xyIncrement = 1.0, zIncrement = 0.5,
               offsetNoiseMag = 0.0;
  // four walls
  double x(xyLimit), y(xyLimit), z(zLimit);
  for (y = -xyLimit; y <= xyLimit; y += xyIncrement) {
    for (z = -zLimit; z <= zLimit; z += zIncrement) {
      homogeneousPoints->push_back(
          Eigen::Vector4d(x + vio::gauss_rand(0, offsetNoiseMag),
                          y + vio::gauss_rand(0, offsetNoiseMag),
                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
      lmIds->push_back(okvis::IdProvider::instance().newId());
    }
  }

  for (y = -xyLimit; y <= xyLimit; y += xyIncrement) {
    for (z = -zLimit; z <= zLimit; z += zIncrement) {
      homogeneousPoints->push_back(
          Eigen::Vector4d(y + vio::gauss_rand(0, offsetNoiseMag),
                          x + vio::gauss_rand(0, offsetNoiseMag),
                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
      lmIds->push_back(okvis::IdProvider::instance().newId());
    }
  }

  x = -xyLimit;
  for (y = -xyLimit; y <= xyLimit; y += xyIncrement) {
    for (z = -zLimit; z <= zLimit; z += zIncrement) {
      homogeneousPoints->push_back(
          Eigen::Vector4d(x + vio::gauss_rand(0, offsetNoiseMag),
                          y + vio::gauss_rand(0, offsetNoiseMag),
                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
      lmIds->push_back(okvis::IdProvider::instance().newId());
    }
  }

  for (y = -xyLimit; y <= xyLimit; y += xyIncrement) {
    for (z = -zLimit; z <= zLimit; z += zIncrement) {
      homogeneousPoints->push_back(
          Eigen::Vector4d(y + vio::gauss_rand(0, offsetNoiseMag),
                          x + vio::gauss_rand(0, offsetNoiseMag),
                          z + vio::gauss_rand(0, offsetNoiseMag), 1));
      lmIds->push_back(okvis::IdProvider::instance().newId());
    }
  }

  if (addFloorCeling) {
    std::vector<double> zlist{-zLimit, zLimit};
    for (double z : zlist) {
      for (x = -xyLimit; x <= xyLimit; x += xyIncrement) {
        for (y = -xyLimit; y <= xyLimit; y += xyIncrement) {
          homogeneousPoints->push_back(
              Eigen::Vector4d(x + vio::gauss_rand(0, offsetNoiseMag),
                              y + vio::gauss_rand(0, offsetNoiseMag),
                              z + vio::gauss_rand(0, offsetNoiseMag), 1));
          lmIds->push_back(okvis::IdProvider::instance().newId());
        }
      }
    }
  }
}

void createCylinderLandmarkGrid(
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
        *homogeneousPoints,
    std::vector<uint64_t> *lmIds, double radius) {
  const int numSteps = 40;
  double zmin = -1.5, zmax = 1.5;
  double zstep = 0.5;
  if (radius >= kRangeThreshold) {
    zmin = -1.5;
    zmax = 3.5;
    zstep = 0.8;
  }
  double step = 2 * M_PI / numSteps;
  for (int j = 0; j < numSteps; ++j) {
    double theta = step * j;
    double px = radius * sin(theta);
    double py = radius * cos(theta);
    for (double pz = zmin; pz < zmax; pz += zstep) {
      homogeneousPoints->emplace_back(px, py, pz, 1.0);
      lmIds->push_back(okvis::IdProvider::instance().newId());
    }
  }
}

void addLandmarkNoise(
    const std::vector<Eigen::Vector4d,
                      Eigen::aligned_allocator<Eigen::Vector4d>>
        &homogeneousPoints,
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
        *noisyHomogeneousPoints,
    double axisSigma) {
  *noisyHomogeneousPoints = homogeneousPoints;
  for (auto iter = noisyHomogeneousPoints->begin();
       iter != noisyHomogeneousPoints->end(); ++iter) {
    iter->head<3>() = iter->head<3>() + Eigen::Vector3d::Random() * axisSigma;
  }
}

void initCameraNoiseParams(
    double sigma_abs_position, double sigma_abs_orientation,
    okvis::ExtrinsicsEstimationParameters *cameraNoiseParams) {
  cameraNoiseParams->sigma_absolute_translation = sigma_abs_position;
  cameraNoiseParams->sigma_absolute_orientation = sigma_abs_orientation;
  cameraNoiseParams->sigma_c_relative_translation = 0;
  cameraNoiseParams->sigma_c_relative_orientation = 0;

  cameraNoiseParams->sigma_focal_length = 5;
  cameraNoiseParams->sigma_principal_point = 5;
  cameraNoiseParams->sigma_distortion =
      std::vector<double>{5e-2, 5e-2, 5e-2, 5e-2, 5e-2};
  cameraNoiseParams->sigma_td = 1e-2;
  cameraNoiseParams->sigma_tr = 5e-3;
}
} // namespace simul
