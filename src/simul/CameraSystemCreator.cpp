#include <simul/CameraSystemCreator.hpp>

namespace simul {

const std::string CameraSystemCreator::distortName_ = "RadialTangentialDistortion";

Eigen::Matrix<double, 4, 4> create_T_BC(CameraOrientation orientationId,
                                        int /*camIdx*/) {
  Eigen::Matrix<double, 4, 4> matT_SC0;
  switch (orientationId) {
    case CameraOrientation::Backward: // Backward motion: The camera faces backward when the device goes straight forward.
      matT_SC0 << 0, 0, -1, 0,
                  1, 0, 0, 0,
                  0, -1, 0, 0,
                  0, 0, 0, 1;
      break;
    case CameraOrientation::Left: // Sideways motion: The camera faces left if the device goes straight forward.
      matT_SC0 << 1, 0, 0, 0,
                  0, 0, 1, 0,
                  0, -1, 0, 0,
                  0, 0, 0, 1;
      break;
    case CameraOrientation::Right: // Sideways motion: The camera faces right if the device goes straight forward.
      matT_SC0 << -1, 0, 0, 0,
                  0, 0, -1, 0,
                  0, -1, 0, 0,
                  0, 0, 0, 1;
      break;
    case CameraOrientation::Forward: // Forward motion: The camera faces forward when the device goes straight forward.
    default:
      matT_SC0 << 0, 0, 1, 0,
                 -1, 0, 0, 0,
                 0, -1, 0, 0,
                 0, 0, 0, 1;
      break;
  }
  return matT_SC0;
}

} // namespace simul
