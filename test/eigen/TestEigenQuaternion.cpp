#include <gtest/gtest.h>
#include <Eigen/Geometry>

TEST(Eigen, QuaternionConstructor) {
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
        C_list;
    Eigen::Matrix3d C;
    // Why does the quaternion constructor have trouble with this SO(3) element?
    // It is not right-handed.
//    C << 0, 1, 0, -1, 0, 0, 0, 0, -1;
//    C_list.push_back(C);
    C << 0, -1, 0, -1, 0, 0, 0, 0, -1;
    C_list.push_back(C);
    for (auto C : C_list) {
      Eigen::Quaterniond q(C);
      q.normalize();
      Eigen::Matrix3d Ct = q.toRotationMatrix();
      EXPECT_LT((Ct - C).lpNorm<Eigen::Infinity>(), 1e-5)
          << "C\n" << C << "\nQuaternion\n" << q.coeffs().transpose()
          << "\nC through Quaternion constructor\n" << Ct;
    }
}

TEST(EigenMatrix, AngleAxis){
    Eigen::Matrix3d m;
    Eigen::Vector3d axisAngles(0.1, 0.2, 0.3);
    axisAngles*=M_PI;
    m = Eigen::AngleAxisd(axisAngles[0], Eigen::Vector3d::UnitX())
            * Eigen::AngleAxisd(axisAngles[1], Eigen::Vector3d::UnitY())
            * Eigen::AngleAxisd(axisAngles[2], Eigen::Vector3d::UnitZ());
    Eigen::Vector3d ea = m.eulerAngles(0, 1, 2);
    ASSERT_NEAR((ea- axisAngles).lpNorm<Eigen::Infinity>(), 0, 1e-8);

    double theta= M_PI/6;
    Eigen::Matrix3d Rx = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitX()).toRotationMatrix();
    Eigen::Matrix3d Ry = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Matrix3d Rz = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::Matrix3d Rxs2f, Rys2f, Rzs2f; //rotation of angle theta from start frame to finish frame
    double st =sin(theta), ct = cos(theta);
    Rxs2f <<1, 0, 0, 0, ct, st, 0, -st, ct;
    Rys2f <<ct, 0, -st, 0,1,0, st, 0, ct;
    Rzs2f <<ct, st, 0, -st, ct, 0, 0,0,1;
    ASSERT_NEAR((Rx- Rxs2f.transpose()).lpNorm<Eigen::Infinity>(), 0, 1e-8);
    ASSERT_NEAR((Ry- Rys2f.transpose()).lpNorm<Eigen::Infinity>(), 0, 1e-8);
    ASSERT_NEAR((Rz- Rzs2f.transpose()).lpNorm<Eigen::Infinity>(), 0, 1e-8);
}
