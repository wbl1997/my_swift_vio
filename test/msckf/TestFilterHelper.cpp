#include <gtest/gtest.h>
#include <msckf/FilterHelper.hpp>

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
  removeUnsetElements(&sampleElements, markers);
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
  removeUnsetElements(&sampleElements, markers, step);
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
  removeUnsetMatrices(&sampleMatrices, markers);
  for (size_t j = 0; j < sampleMatrices.size(); ++j) {
    EXPECT_TRUE(sampleMatrices[j].isApprox(expectedMatrices[j], 1e-8));
  }
}
