#include <gtest/gtest.h>
#include <swift_vio/FilterHelper.hpp>
#include <swift_vio/VectorOperations.hpp>
#include <simul/SimulationNView.h>

#include <random>

TEST(FilterHelper, removeUnsetElements) {
  std::vector<double> sampleElements, expectedElements;
  std::vector<bool> markers;
  for (int j = 0; j < 10; ++j) {
    double ele = std::rand() * 0.5;
    sampleElements.push_back(ele);
    bool push = std::rand() % 2 == 1 ? true : false;
    markers.push_back(push);
    if (push) {
      expectedElements.push_back(ele);
    }
  }
  swift_vio::removeUnsetElements(&sampleElements, markers);
  for (size_t j = 0; j < sampleElements.size(); ++j) {
    EXPECT_NEAR(sampleElements[j], expectedElements[j], 1e-8);
  }
}

TEST(FilterHelper, removeUnsetElementsStep) {
  std::vector<double> sampleElements, expectedElements;
  std::vector<bool> markers;
  int step = 2;
  for (int j = 0; j < 10; ++j) {
    std::vector<double> eleArray;
    for (int k = 0; k < step; ++k) {
      double ele = std::rand() * 0.5;
      eleArray.push_back(ele);
    }

    sampleElements.insert(sampleElements.end(), eleArray.begin(),
                          eleArray.end());
    bool push = std::rand() % 2 == 1 ? true : false;
    markers.push_back(push);
    if (push) {
      expectedElements.insert(expectedElements.end(), eleArray.begin(),
                              eleArray.end());
    }
  }
  swift_vio::removeUnsetElements(&sampleElements, markers, step);
  for (size_t j = 0; j < sampleElements.size(); ++j) {
    EXPECT_NEAR(sampleElements[j], expectedElements[j], 1e-8);
  }
}

TEST(FilterHelper, removeUnsetMatrices) {
  std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>>
      sampleMatrices, expectedMatrices;
  std::vector<bool> markers;
  for (int j = 0; j < 10; ++j) {
    Eigen::Matrix2d mat = Eigen::Matrix2d::Random();
    sampleMatrices.push_back(mat);
    bool push = std::rand() % 2 == 1 ? true : false;
    markers.push_back(push);
    if (push) {
      expectedMatrices.push_back(mat);
    }
  }
  swift_vio::removeUnsetMatrices(&sampleMatrices, markers);
  for (size_t j = 0; j < sampleMatrices.size(); ++j) {
    EXPECT_TRUE(sampleMatrices[j].isApprox(expectedMatrices[j], 1e-8));
  }
}

TEST(SimulationNView, projectWithJacobian) {
  simul::SimulationTwoView stv(0);
  okvis::kinematics::Transformation T_CW = stv.T_CW(1);
  Eigen::Vector2d xyatz1;
  Eigen::Matrix<double, 2, 6> Hx;
  Eigen::Matrix<double, 2, 3> Hf;
  bool projOk = simul::SimulationNView::project(T_CW, stv.truePoint(), &xyatz1, &Hx, &Hf);

  // numeric diff
  Eigen::Vector2d xyatz1Numeric;
  Eigen::Matrix<double, 2, 6> HxNumeric;
  Eigen::Matrix<double, 2, 3> HfNumeric;
  bool projOkNumeric = simul::SimulationNView::project(T_CW, stv.truePoint(), &xyatz1Numeric,
                                  &HxNumeric, &HfNumeric);
  EXPECT_TRUE(projOk && projOkNumeric);
  EXPECT_TRUE(xyatz1.isApprox(xyatz1Numeric, 1e-8));
  EXPECT_TRUE(Hx.isApprox(HxNumeric, 1e-8));
  EXPECT_TRUE(Hf.isApprox(HfNumeric, 1e-8));
}

TEST(FilterHelper, nullspaceWithRankCheck) {
  // simulation n views
  simul::SimulationNViewStatic snvs(false, false);
  const int nViews = snvs.numObservations();
  Eigen::Matrix<double, -1, -1> Hx(nViews * 2, 6);
  Eigen::Matrix<double, -1, -1> Hf(nViews * 2, 3);
  Eigen::Matrix<double, -1, 1> residual(nViews * 2, 1);
  int validRows = 0;
  for (int j = 0; j < nViews; ++j) {
      Eigen::Vector2d subresidual;
      Eigen::Matrix<double, 2, 6> subHx;
      Eigen::Matrix<double, 2, 3> subHf;
      bool projOk = snvs.project(j, &subresidual, &subHx, &subHf);
      if (!projOk)
        continue;
      residual.block<2, 1>(validRows, 0) = subresidual;
      Hx.block<2, 6>(validRows, 0) = subHx;
      Hf.block<2, 3>(validRows, 0) = subHf;
      validRows += 2;
  }
  Hx.conservativeResize(validRows, Eigen::NoChange);
  Hf.conservativeResize(validRows, Eigen::NoChange);
  residual.conservativeResize(validRows, Eigen::NoChange);

//  std::cout << "residuals:" << residual.transpose()
//            << "\nHx\n" << Hx << "\nHf\n" << Hf << "\n";

  Eigen::MatrixXd leftNullspace3 = swift_vio::FilterHelper::leftNullspaceWithRankCheck(Hf, 3);
  EXPECT_TRUE((leftNullspace3.transpose() * Hf).isMuchSmallerThan(1, 1e-8));
  Eigen::MatrixXd leftNullspace2 = swift_vio::FilterHelper::leftNullspaceWithRankCheck(Hf, 2);
  EXPECT_TRUE((leftNullspace2.transpose() * Hf).isMuchSmallerThan(1, 1e-8));
//  std::cout << "(3)Q2' * Hx\n" << leftNullspace3.transpose() * Hx << "\n";
//  std::cout << "(2)Q2' * Hx\n" << leftNullspace2.transpose() * Hx << "\n";
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(validRows, validRows);

  Eigen::Matrix<double, -1, -1> HxGivens = Hx;
  Eigen::Matrix<double, -1, -1> HfGivens = Hf;
  Eigen::Matrix<double, -1, 1> residualGivens = residual;
  swift_vio::FilterHelper::multiplyLeftNullspaceWithGivens(&HfGivens, &HxGivens, &residualGivens, &R, 3);

//  std::cout << "Hf Givens\n" << HfGivens << "\nHx\n" << HxGivens << "\n";
}
