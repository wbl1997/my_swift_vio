#ifndef INCLUDE_MSCKF_SIMULATION_NVIEW_H_
#define INCLUDE_MSCKF_SIMULATION_NVIEW_H_

#include <msckf/FeatureTriangulation.hpp>

#include <okvis/kinematics/Transformation.hpp>

#include <sophus/se3.hpp>

namespace simul {

class SimulationNView {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SimulationNView()
      : truePoint_(2.31054434, -1.58786347, 9.79390227, 1.0),
        obsDirections_(3),
        T_CWs_(3) {
    obsDirections_[0] = Eigen::Vector3d(0.1444439, -0.0433997, 1);
    obsDirections_[1] = Eigen::Vector3d(0.21640816, -0.34998059, 1);
    obsDirections_[2] = Eigen::Vector3d(0.23628522, -0.31005748, 1);
    Eigen::Matrix3d rot;
    rot << 0.99469755, -0.02749299, -0.09910054, 0.02924625, 0.99943961,
        0.0162823, 0.09859735, -0.01909428, 0.9949442;
    Eigen::Vector3d tcinw(0.37937094, -1.06289834, 1.93156378);
    T_CWs_[0] =
        okvis::kinematics::Transformation(-rot * tcinw,
                                          Eigen::Quaterniond(rot));

    rot << 0.99722659, -0.0628095, 0.03992603, 0.0671776, 0.99054536,
        -0.11961209, -0.03203577, 0.1219625, 0.99201757;
    tcinw << 0.78442247, 0.69195074, -0.10575422;
    T_CWs_[1] = okvis::kinematics::Transformation(-rot * tcinw,
                                                  Eigen::Quaterniond(rot));

    rot << 0.99958901, 0.01425856, 0.02486967, -0.01057666, 0.98975966,
        -0.14235147, -0.02664472, 0.14202993, 0.98950369;
    tcinw << 0.35451434, -0.09596801, 0.30737151;
    T_CWs_[2] = okvis::kinematics::Transformation(-rot * tcinw,
                                                  Eigen::Quaterniond(rot));
  }

  SimulationNView(int numViews) : obsDirections_(numViews), T_CWs_(numViews) {

  }

  size_t numObservations() const { return obsDirections_.size(); }
  okvis::kinematics::Transformation T_CW(int j) const { return T_CWs_[j]; }
  okvis::kinematics::Transformation T_WC(int j) const { return T_CWs_[j].inverse(); }
  Eigen::Matrix<double, 3, 1> obsDirection(int j) const {
    return obsDirections_[j];
  }
  Eigen::Matrix<double, 3, 1> obsUnitDirection(int j) const {
    return obsDirections_[j].normalized();
  }

  std::vector<Eigen::Vector3d,
              Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>
  obsDirections() const {
    return obsDirections_;
  }

  Eigen::Matrix<double, 2, 1> obsDirectionZ1(int j) const {
    return obsDirections_[j].head<2>() / obsDirections_[j][2];
  }

  std::vector<Eigen::Vector2d,
              Eigen::aligned_allocator<Eigen::Matrix<double, 2, 1>>>
  obsDirectionsZ1() const {
    std::vector<Eigen::Vector2d,
                Eigen::aligned_allocator<Eigen::Matrix<double, 2, 1>>>
        xyList;
    xyList.reserve(obsDirections_.size());
    for (auto item : obsDirections_) {
      xyList.emplace_back(item.head<2>() / item[2]);
    }
    return xyList;
  }

  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> se3_T_CWs()
      const {
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> se3_CWs;
    for (size_t jack = 0; jack < T_CWs_.size(); ++jack) {
      se3_CWs.emplace_back(T_CWs_[jack].q(), T_CWs_[jack].r());
    }
    return se3_CWs;
  }

  msckf_vio::CamStateServer camStates() const {
    msckf_vio::CamStateServer cam_states(T_CWs_.size());
    for (size_t i = 0; i < T_CWs_.size(); ++i) {
      msckf_vio::CAMState new_cam_state;
      new_cam_state.id = i;
      new_cam_state.orientation = T_CWs_[i].C();
      new_cam_state.position = -T_CWs_[i].C().transpose() * T_CWs_[i].r();
      cam_states[new_cam_state.id] = new_cam_state;
    }
    return cam_states;
  }

  Eigen::Vector4d truePoint() const { return truePoint_; }

  static double raySigma(int focalLength) {
    int kpSize = 9;
    double keypointAStdDev = kpSize;
    keypointAStdDev = 0.8 * keypointAStdDev / 12.0;
    return sqrt(sqrt(2)) * keypointAStdDev / focalLength;
  }

 protected:
  Eigen::Vector4d truePoint_;
  std::vector<Eigen::Vector3d,
              Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>
      obsDirections_;
  std::vector<okvis::kinematics::Transformation,
              Eigen::aligned_allocator<okvis::kinematics::Transformation>>
      T_CWs_;
};

class SimulationTwoView : public SimulationNView
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SimulationTwoView(int caseId = 0) : SimulationNView(2) {
      switch (caseId) {
        case 0:
          obsDirections_[0] = Eigen::Vector3d{-0.51809, 0.149922, 1};
          obsDirections_[1] =
              Eigen::Vector3d{-0.513966413218, 0.150864740297, 1.0};
          T_CWs_[0] = okvis::kinematics::Transformation();
          T_CWs_[1] = okvis::kinematics::Transformation(
              -Eigen::Vector3d{-0.00161695, 0.00450899, 0.0435501},
              Eigen::Quaterniond::Identity());
          truePoint_ = Eigen::Vector4d(-0.444625, 0.129972, 0.865497, 0.190606);
          break;
        case 1:
          obsDirections_[0] = Eigen::Vector3d{-0.130586, 0.0692999, 1};
          obsDirections_[1] = Eigen::Vector3d{-0.129893, 0.0710077, 0.998044};
          obsDirections_[1] /= obsDirections_[1][2];
          T_CWs_[0] = okvis::kinematics::Transformation();
          T_CWs_[1] = okvis::kinematics::Transformation(
              -Eigen::Vector3d{-0.00398223, 0.0133035, 0.112868},
              Eigen::Quaterniond::Identity());
          truePoint_ = Eigen::Vector4d(-0.124202, 0.0682118, 0.962293, 0.23219);
          break;
        case 2:
          obsDirections_[0] = Eigen::Vector3d{0.0604648, 0.0407913, 1};
          obsDirections_[1] = Eigen::Vector3d{0.0681188, 0.0312081, 1.00403};
          obsDirections_[1] /= obsDirections_[1][2];
          T_CWs_[0] = okvis::kinematics::Transformation();
          T_CWs_[1] = okvis::kinematics::Transformation(
              -Eigen::Vector3d{0.0132888, 0.0840325, 0.148991},
              Eigen::Quaterniond::Identity());
          break;
        default:
          break;
      }
    }
};

} // namespace simul
#endif // INCLUDE_MSCKF_SIMULATION_NVIEW_H_
