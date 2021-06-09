#ifndef LANDMARKGRID_H
#define LANDMARKGRID_H

#include <okvis/Parameters.hpp>
#include <swift_vio/memory.h>

namespace simul {
enum class LandmarkGridType {
  FourWalls = 0,
  FourWallsFloorCeiling,
  Cylinder,
};
const double kRangeThreshold = 20;

void saveLandmarkGrid(
    const std::vector<Eigen::Vector4d,
                      Eigen::aligned_allocator<Eigen::Vector4d>>
        &homogeneousPoints,
    const std::vector<uint64_t> &lmIds, std::string pointFile);

void createBoxLandmarkGrid(
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
        *homogeneousPoints,
    std::vector<uint64_t> *lmIds, double halfz, double addFloorCeling);

void createCylinderLandmarkGrid(
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
        *homogeneousPoints,
    std::vector<uint64_t> *lmIds, double radius);

void addLandmarkNoise(
    const std::vector<Eigen::Vector4d,
                      Eigen::aligned_allocator<Eigen::Vector4d>>
        &homogeneousPoints,
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
        *noisyHomogeneousPoints,
    double axisSigma = 0.1);

void initCameraNoiseParams(
    double sigma_abs_position, double sigma_abs_orientation,
    okvis::ExtrinsicsEstimationParameters *cameraNoiseParams);

} // namespace simul
#endif // LANDMARKGRID_H
