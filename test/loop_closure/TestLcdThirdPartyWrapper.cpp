#include <ostream>

#include <loop_closure/LcdThirdPartyWrapper.h>
#include "gtest/gtest.h"

// test RPGO can deal with multiple odometry edges at once

class LcdThirdPartyWrapperTest : public ::testing::Test {
 public:
  LcdThirdPartyWrapperTest() {}

  // Standard gtest methods, unnecessary for now
  virtual void SetUp() {
    lcd_params.reset(new VIO::LoopClosureDetectorParams());

    lcd_params->max_distance_between_groups_ = 3;
    lcd_params->max_intragroup_gap_ = 3;
    lcd_params->min_matches_per_group_ = 2;

    lcd_wrap.reset(new VIO::LcdThirdPartyWrapper(lcd_params));
    lcd_wrap->setLatestQueryId(100u);
    VIO::MatchIsland island(10, 15, 0.001);
    lcd_wrap->setLatestMatchedIsland(island);
    lcd_wrap->setTemporalEntries(1);

  }
  virtual void TearDown() {}

 protected:
  std::shared_ptr<VIO::LoopClosureDetectorParams> lcd_params;
  std::shared_ptr<VIO::LcdThirdPartyWrapper> lcd_wrap;
};

TEST_F(LcdThirdPartyWrapperTest, checkTemporalConstraintOverlap1) {
  VIO::MatchIsland qisland(9, 15, 0.001);
  lcd_wrap->checkTemporalConstraint(101u, qisland);
  int temporal_consistent_groups = lcd_wrap->getTemporalEntries();
  EXPECT_EQ(temporal_consistent_groups, 2);
}

TEST_F(LcdThirdPartyWrapperTest, checkTemporalConstraintOverlap2) {
  VIO::MatchIsland qisland(12, 15, 0.001);
  lcd_wrap->checkTemporalConstraint(101u, qisland);
  int temporal_consistent_groups = lcd_wrap->getTemporalEntries();
  EXPECT_EQ(temporal_consistent_groups, 2);
}

TEST_F(LcdThirdPartyWrapperTest, checkTemporalConstraint1) {
  VIO::MatchIsland qisland(3, 7, 0.001);
  lcd_wrap->checkTemporalConstraint(101u, qisland);
  int temporal_consistent_groups = lcd_wrap->getTemporalEntries();
  EXPECT_EQ(temporal_consistent_groups, 2);
}

TEST_F(LcdThirdPartyWrapperTest, checkTemporalConstraint2) {
  VIO::MatchIsland qisland(3, 6, 0.001);
  lcd_wrap->checkTemporalConstraint(101u, qisland);
  int temporal_consistent_groups = lcd_wrap->getTemporalEntries();
  EXPECT_EQ(temporal_consistent_groups, 1);
}

TEST_F(LcdThirdPartyWrapperTest, checkTemporalConstraint3) {
  VIO::MatchIsland qisland(3, 11, 0.001);
  lcd_wrap->checkTemporalConstraint(101u, qisland);
  int temporal_consistent_groups = lcd_wrap->getTemporalEntries();
  EXPECT_EQ(temporal_consistent_groups, 2);
}

TEST_F(LcdThirdPartyWrapperTest, checkTemporalConstraint4) {
  VIO::MatchIsland qisland(15, 18, 0.001);
  lcd_wrap->checkTemporalConstraint(101u, qisland);
  int temporal_consistent_groups = lcd_wrap->getTemporalEntries();
  EXPECT_EQ(temporal_consistent_groups, 2);
}

TEST_F(LcdThirdPartyWrapperTest, checkTemporalConstraint5) {
  VIO::MatchIsland qisland(18, 23, 0.001);
  lcd_wrap->checkTemporalConstraint(101u, qisland);
  int temporal_consistent_groups = lcd_wrap->getTemporalEntries();
  EXPECT_EQ(temporal_consistent_groups, 2);
}

TEST_F(LcdThirdPartyWrapperTest, checkTemporalConstraint6) {
  VIO::MatchIsland qisland(19, 23, 0.001);
  lcd_wrap->checkTemporalConstraint(101u, qisland);
  int temporal_consistent_groups = lcd_wrap->getTemporalEntries();
  EXPECT_EQ(temporal_consistent_groups, 1);
}

// test loop closure can work with the two versions of the ORBvoc.yaml and
// ORBvoc.txt from ORBSLAM2
TEST_F(LcdThirdPartyWrapperTest, computeIslands1) {
  DBoW2::QueryResults results;
  results.emplace_back(40, 0.2);
  results.emplace_back(30, 0.3);
  results.emplace_back(33, 0.05);
  results.emplace_back(20, 0.3);
  std::vector<VIO::MatchIsland> islands;
  lcd_wrap->computeIslands(&results, &islands);
  EXPECT_EQ(islands.size(), 0u);
}

TEST_F(LcdThirdPartyWrapperTest, computeIslands2) {
  DBoW2::QueryResults results;
  results.emplace_back(27, 0.2);
  results.emplace_back(20, 0.3);
  results.emplace_back(30, 0.3);
  results.emplace_back(32, 0.05);

  std::vector<VIO::MatchIsland> islands;
  lcd_wrap->computeIslands(&results, &islands);
  EXPECT_EQ(islands.size(), 1u);
  EXPECT_EQ(islands[0].start_id_, 30u);
  EXPECT_EQ(islands[0].end_id_, 32u);
  EXPECT_EQ(islands[0].best_id_, 30u);
}

TEST_F(LcdThirdPartyWrapperTest, computeIslands3) {
  DBoW2::QueryResults results;
  results.emplace_back(27, 0.2);
  results.emplace_back(25, 0.3);
  results.emplace_back(30, 0.3);
  results.emplace_back(32, 0.05);

  std::vector<VIO::MatchIsland> islands;
  lcd_wrap->computeIslands(&results, &islands);
  EXPECT_EQ(islands.size(), 2u);
  EXPECT_EQ(islands[0].start_id_, 25u);
  EXPECT_EQ(islands[0].end_id_, 27u);
  EXPECT_EQ(islands[0].best_id_, 25u);
  EXPECT_EQ(islands[1].start_id_, 30u);
  EXPECT_EQ(islands[1].end_id_, 32u);
  EXPECT_EQ(islands[1].best_id_, 30u);
}
