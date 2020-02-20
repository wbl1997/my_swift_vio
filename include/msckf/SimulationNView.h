#ifndef INCLUDE_MSCKF_SIMULATION_NVIEW_H_
#define INCLUDE_MSCKF_SIMULATION_NVIEW_H_

#include <okvis/kinematics/Transformation.hpp>

#include <sophus/se3.hpp>

#include <vio/Sample.h>

namespace simul {

class SimulationNView {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SimulationNView() {}
  ~SimulationNView() {}

  SimulationNView(int numViews) : obsDirections_(numViews), T_CWs_(numViews) {}

  size_t numObservations() const { return obsDirections_.size(); }
  okvis::kinematics::Transformation T_CW(int j) const { return T_CWs_[j]; }
  okvis::kinematics::Transformation T_WC(int j) const {
    return T_CWs_[j].inverse();
  }
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

  std::vector<okvis::kinematics::Transformation,
              Eigen::aligned_allocator<okvis::kinematics::Transformation>>
  camStates() const {
    std::vector<okvis::kinematics::Transformation,
                Eigen::aligned_allocator<okvis::kinematics::Transformation>>
        cam_states(T_CWs_.size());
    for (size_t i = 0; i < T_CWs_.size(); ++i) {
      cam_states[i] = T_CWs_[i].inverse();
    }
    return cam_states;
  }

  Eigen::Vector4d truePoint() const { return truePoint_; }

  bool project(int obsIndex, Eigen::Vector2d* xyatz1,
               Eigen::Matrix<double, 2, 6>* Hx,
               Eigen::Matrix<double, 2, 3>* Hf) const {
    return project(T_CWs_[obsIndex], truePoint_, xyatz1, Hx, Hf);
  }

  static double raySigma(int focalLength) {
    int kpSize = 9;
    double keypointAStdDev = kpSize;
    keypointAStdDev = 0.8 * keypointAStdDev / 12.0;
    return sqrt(sqrt(2)) * keypointAStdDev / focalLength;
  }

  /**
   * @brief project
   * @param T_CW
   * @param landmark
   * @param xyatz1
   * @param Hx states are defined such that
   *     $R_{WC} \approx (I + \theta_{WC}\times) \hat{R}_{WC}$ and
   *     $t_{WC} = \delta t_{WC} + t_{WC}$
   *     $h = \hat{h} + \delta h_{1:3}$
   * @param Hf
   * @return
   */
  static bool project(const okvis::kinematics::Transformation& T_CW,
                      const Eigen::Vector4d landmark, Eigen::Vector2d* xyatz1,
                      Eigen::Matrix<double, 2, 6>* Hx,
                      Eigen::Matrix<double, 2, 3>* Hf) {
    Eigen::Vector4d pinC = T_CW * landmark;
    if (pinC[2] / pinC[3] < 0) {
      return false;
    }
    *xyatz1 = pinC.head<2>() / pinC[2];
    Eigen::Matrix<double, 2, 3> Jh;
    Jh << pinC[2], 0, -pinC[0], 0, pinC[2], -pinC[1];
    Jh /= (pinC[2] * pinC[2]);
    Eigen::Matrix<double, 3, 6> Jx;
    Eigen::Vector3d t_WC = T_CW.inverse().r();
    Jx << T_CW.C() * okvis::kinematics::crossMx(landmark.head<3>() -
                                                landmark[3] * t_WC),
        -T_CW.C() * landmark[3];

    Eigen::Matrix<double, 3, 4> Jf;
    Jf = T_CW.T3x4();
    Eigen::Matrix<double, 4, 3> Jlift =
        Eigen::Matrix<double, 4, 3>::Identity(4, 3);
    *Hx = Jh * Jx;
    *Hf = Jh * Jf * Jlift;
    return true;
  }

  static bool projectAIDP(const okvis::kinematics::Transformation& T_WC,
                          const okvis::kinematics::Transformation& T_WA,
                      const Eigen::Vector4d ab1rho, Eigen::Vector2d* xyatz1,
                      Eigen::Matrix<double, 2, 12>* /*Hx*/,
                      Eigen::Matrix<double, 2, 3>* /*Hf*/) {
    Eigen::Vector4d rhopinC = T_WC.inverse() * T_WA * ab1rho;
    if (rhopinC[2] / rhopinC[3] < 0) {
      return false;
    }
    *xyatz1 = rhopinC.head<2>() / rhopinC[2];
// TODO(jhuai): compute the Jacobians
//    Eigen::Matrix<double, 2, 3> Jh;
//    Jh << rhopinC[2], 0, -rhopinC[0], 0, rhopinC[2], -rhopinC[1];
//    Jh /= (rhopinC[2] * rhopinC[2]);
//    Eigen::Matrix<double, 3, 6> Jx;
//    Eigen::Vector3d t_WC = T_WC.r();
//    okvis::kinematics::Transformation T_CW = T_WC.inverse();
//    Jx << T_CW.C() * okvis::kinematics::crossMx(ab1rho.head<3>() -
//                                                ab1rho[3] * t_WC),
//        -T_CW.C() * ab1rho[3];

//    Eigen::Matrix<double, 3, 4> Jf;
//    Jf = T_CW.T3x4();
//    Eigen::Matrix<double, 4, 3> Jlift =
//        Eigen::Matrix<double, 4, 3>::Identity(4, 3);
//    Hx->leftCols(6) = Jh * Jx;
//    *Hf = Jh * Jf * Jlift;
    return true;
  }

  static bool project(const okvis::kinematics::Transformation& T_CW,
                      const Eigen::Vector4d landmark, Eigen::Vector2d* xyatz1) {
    Eigen::Vector4d pinC = T_CW * landmark;
    if (pinC[2] / pinC[3] < 0) {
      return false;
    }
    *xyatz1 = pinC.head<2>() / pinC[2];
    return true;
  }

  static bool projectWithNumericDiff(const okvis::kinematics::Transformation& T_CW,
                      const Eigen::Vector4d landmark, Eigen::Vector2d* xyatz1,
                      Eigen::Matrix<double, 2, 6>* Hx,
                      Eigen::Matrix<double, 2, 3>* Hf) {
    bool projectOk = project(T_CW, landmark, xyatz1);
    if (!projectOk)
      return false;
    Hx->resize(2, 6);
    Hf->resize(2, 3);
    okvis::kinematics::Transformation T_WC = T_CW.inverse();
    Eigen::Matrix<double, 6, 1> delta;
    Eigen::Vector2d xybar;
    const double eps = 1e-6;
    for (int i = 0; i < 6; ++i) {
      delta.setZero();
      delta(i) = eps;
      okvis::kinematics::Transformation T_WC_bar = T_WC;
      T_WC_bar.oplus(delta);
      project(T_WC_bar.inverse(), landmark, &xybar);
      Hx->col(i) = (xybar - *xyatz1) / eps;
    }
    Eigen::Vector4d landmarkBar;
    for (int i = 0; i < 3; ++i) {
      landmarkBar = landmark;
      landmarkBar[i] += eps;
      project(T_CW, landmarkBar, &xybar);
      Hf->col(i) = (xybar - *xyatz1) / eps;
    }
    return true;
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

class SimulationThreeView : public SimulationNView {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // default 3 view test case
  SimulationThreeView()
      : SimulationNView(3) {
    truePoint_ = Eigen::Vector4d(2.31054434, -1.58786347, 9.79390227, 1.0);
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
  // random 3 view test case
  bool randomThreeView(int maxDistance) {
    truePoint_ = Eigen::Vector4d(
        1.5 * static_cast<double>(std::rand() % maxDistance) / maxDistance,
        3 * static_cast<double>(std::rand() % maxDistance) / maxDistance,
        25 * static_cast<double>(std::rand() % maxDistance) / maxDistance, 1.0);
    obsDirections_.resize(3);
    T_CWs_.resize(3);
    Eigen::Quaterniond so3M(
        0.8, 0.3 * static_cast<double>(std::rand() % maxDistance) / maxDistance,
        0.1, 0.2);
    so3M.normalize();
    T_CWs_[0] =
        okvis::kinematics::Transformation(Eigen::Vector3d(0.0, 0.0, 0.7), so3M);
    Eigen::Quaterniond so3N(
        5.8, 0.1,
        1.0 * static_cast<double>(std::rand() % maxDistance) / maxDistance,
        0.2);
    so3N.normalize();
    T_CWs_[1] = okvis::kinematics::Transformation(
        Eigen::Vector3d(
            1.0,
            -0.3 * static_cast<double>(std::rand() % maxDistance) / maxDistance,
            3.0),
        so3N);
    Eigen::Quaterniond so3P(
        4.5, 0.8, 1.0,
        1.4 * static_cast<double>(std::rand() % maxDistance) / maxDistance);
    so3P.normalize();
    T_CWs_[2] = okvis::kinematics::Transformation(
        Eigen::Vector3d(
            1.0,
            -0.2 * static_cast<double>(std::rand() % maxDistance) / maxDistance,
            -5.0),
        so3P);
    int numValid = 0;
    for (int i = 0; i < 3; ++i) {
      Eigen::Vector3d v3Cam = (T_CWs_[i] * truePoint_).head<3>();
      if (v3Cam[2] < 0) {
        continue;
      }
      obsDirections_[i].head<2>() =
          v3Cam.head<2>() / v3Cam[2] +
          Eigen::Vector2d(1, 1) * static_cast<double>(std::rand() % 10) / 1000;
      obsDirections_[i][2] = 1.0;
      ++numValid;
    }
    return numValid == 3;
  }
};

class SimulationTwoView : public SimulationNView
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SimulationTwoView(int caseId = 0, double depth = 1.0) : SimulationNView(2) {
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
          truePoint_ /= truePoint_[3];
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
          truePoint_ /= truePoint_[3];
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
        case 3:  // pure rotation
          T_CWs_[0] = okvis::kinematics::Transformation();
          T_CWs_[1] = okvis::kinematics::Transformation(
              Eigen::Vector3d::Zero(),
              Eigen::Quaterniond(Eigen::AngleAxisd(-30 * M_PI / 180,
                                                   Eigen::Vector3d::UnitY())));

          obsDirections_[0] =
              Eigen::Vector3d(cos(15 * M_PI / 180) * cos(45 * M_PI / 180),
                              -sin(15 * M_PI / 180),
                              cos(15 * M_PI / 180) * sin(45 * M_PI / 180));
          obsDirections_[1] = T_CWs_[1].C() * obsDirections_[0];
          obsDirections_[0] /= obsDirections_[0][2];
          obsDirections_[1] /= obsDirections_[1][2];
          truePoint_ << obsDirections_[0], 0.0;
          break;
        case 4:  // fake pure rotation observations for stationary motion
          T_CWs_[0] = okvis::kinematics::Transformation();
          T_CWs_[1] = okvis::kinematics::Transformation();

          obsDirections_[0] =
              Eigen::Vector3d(cos(15 * M_PI / 180) * cos(45 * M_PI / 180),
                              -sin(15 * M_PI / 180),
                              cos(15 * M_PI / 180) * sin(45 * M_PI / 180));
          obsDirections_[1] = T_CWs_[1].C().transpose() * obsDirections_[0];
          obsDirections_[0] /= obsDirections_[0][2];
          obsDirections_[1] /= obsDirections_[1][2];
          truePoint_ << obsDirections_[0], 0.0;
          break;
        case 5:  // points at varying depths
          T_CWs_[0] = okvis::kinematics::Transformation();
          T_CWs_[1] = okvis::kinematics::Transformation(
              Eigen::Vector3d::Random(),
              Eigen::Quaterniond(
                  Eigen::AngleAxisd(-30 * M_PI / 180, Eigen::Vector3d::UnitY())
                      .toRotationMatrix()));

          obsDirections_[0]
              << depth * cos(15 * M_PI / 180) * cos(45 * M_PI / 180),
              -depth * sin(15 * M_PI / 180),
              depth * cos(15 * M_PI / 180) * sin(45 * M_PI / 180);

          obsDirections_[1] = T_CWs_[1].C() * obsDirections_[0] + T_CWs_[1].r();
          truePoint_ << obsDirections_[0], 1.0;
          obsDirections_[0] = obsDirections_[0] / obsDirections_[0][2];
          obsDirections_[1] = obsDirections_[1] / obsDirections_[1][2];
          break;
        default:
          break;
      }
    }
};

class SimulationNViewSphere : public SimulationNView {
public:
  SimulationNViewSphere() : SimulationNView(6) {
    // Set the real feature at the origin of the world frame.
    truePoint_ = Eigen::Vector4d(0.5, 0.0, 0.0, 1.0);

    // Add 6 camera poses, all of which are able to see the
    // feature at the origin. For simplicity, the six camera
    // view are located at the six intersections between a
    // unit sphere and the coordinate system. And the z axes
    // of the camera frames are facing the origin.
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
        cam_poses(6);
    // Positive x axis.
    cam_poses[0].linear() << 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0;
    cam_poses[0].translation() << 1.0, 0.0, 0.0;
    // Positive y axis.
    cam_poses[1].linear() << -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0;
    cam_poses[1].translation() << 0.0, 1.0, 0.0;
    // Negative x axis.
    cam_poses[2].linear() << 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0;
    cam_poses[2].translation() << -1.0, 0.0, 0.0;
    // Negative y axis.
    cam_poses[3].linear() << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0;
    cam_poses[3].translation() << 0.0, -1.0, 0.0;
    // Positive z axis.
    cam_poses[4].linear() << 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0;
    cam_poses[4].translation() << 0.0, 0.0, 1.0;
    // Negative z axis.
    cam_poses[5].linear() << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    cam_poses[5].translation() << 0.0, 0.0, -1.0;

    // Set the camera states
    for (int i = 0; i < 6; ++i) {
      T_CWs_[i] = okvis::kinematics::Transformation(
          cam_poses[i].translation(),
          Eigen::Quaterniond(cam_poses[i].linear())).inverse();
    }

    // Compute measurements.
    vio::Sample noise_generator;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        measurements(6);
    for (int i = 0; i < 6; ++i) {
      Eigen::Isometry3d cam_pose_inv = cam_poses[i].inverse();
      Eigen::Vector3d p = cam_pose_inv.linear() * truePoint_.head<3>() +
                          cam_pose_inv.translation();
      double u = p(0) / p(2) + noise_generator.gaussian(0.01);
      double v = p(1) / p(2) + noise_generator.gaussian(0.01);
      // double u = p(0) / p(2);
      // double v = p(1) / p(2);
      measurements[i] = Eigen::Vector2d(u, v);
      obsDirections_[i] = Eigen::Vector3d(u, v, 1.0);
    }
  }
};

class SimulationNViewStatic : public SimulationNView {
 public:
  SimulationNViewStatic(bool addSidewaysView, bool addObsNoise) : SimulationNView(6) {
    truePoint_ = Eigen::Vector4d(0.2, 0.5, 1.5, 1.0);
    int nviews = addSidewaysView ? 5 : 6;
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
        cam_poses(nviews);
    // Set the camera states
    for (int i = 0; i < nviews; ++i) {
      cam_poses[i].setIdentity();
      T_CWs_[i] = okvis::kinematics::Transformation(
          cam_poses[i].translation(),
          Eigen::Quaterniond(cam_poses[i].linear())).inverse();
    }

    // Compute measurements.
    vio::Sample noise_generator;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> measurements(nviews);
    for (int i = 0; i < nviews; ++i) {
      Eigen::Isometry3d cam_pose_inv = cam_poses[i].inverse();
      Eigen::Vector3d p = cam_pose_inv.linear() * truePoint_.head<3>() + cam_pose_inv.translation();
      double u = p(0) / p(2);
      double v = p(1) / p(2);
      if (addObsNoise) {
        u += noise_generator.gaussian(0.01);
        v += noise_generator.gaussian(0.01);
      }
      measurements[i] = Eigen::Vector2d(u, v);
      obsDirections_[i] = Eigen::Vector3d(u, v, 1.0);
    }

    // add an observation from a sideways view
    if (addSidewaysView) {
      Eigen::Isometry3d cam_pose_sideways;
      cam_pose_sideways.linear() = Eigen::Matrix3d::Identity();
      cam_pose_sideways.translation() << 1.0, 0.0, 0.0;
      Eigen::Isometry3d cam_pose_inv = cam_pose_sideways.inverse();
      Eigen::Vector3d p = cam_pose_inv.linear() * truePoint_.head<3>() + cam_pose_inv.translation();
      double u = p(0) / p(2);
      double v = p(1) / p(2);
      if (addObsNoise) {
        u += noise_generator.gaussian(0.01);
        v += noise_generator.gaussian(0.01);
      }
      measurements.push_back(Eigen::Vector2d(u, v));
      obsDirections_[nviews] = Eigen::Vector3d(u, v, 1.0);
      T_CWs_[nviews] = okvis::kinematics::Transformation(
          cam_pose_sideways.translation(),
          Eigen::Quaterniond(cam_pose_sideways.linear())).inverse();

    }
  }
};

} // namespace simul
#endif // INCLUDE_MSCKF_SIMULATION_NVIEW_H_
