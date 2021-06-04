/**
 * @file   testLoopClosureMethod.cpp
 * @brief  test LoopClosureMethod
 * @author 
 */

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "loop_closure/GtsamWrap.hpp"
#include "loop_closure/LoopClosureDetector.h"

#include <loop_closure/LoopClosureDetectorParams.h>

#include <swift_vio/memory.h>

#include <swift_vio/VectorOperations.hpp>
#include <swift_vio/VioFactoryMethods.hpp>
#include <swift_vio/TwoViewGeometry.hpp>

#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/triangulation/ProbabilisticStereoTriangulator.hpp>

#include <okvis/Parameters.hpp>

DEFINE_string(test_data_path, "test/data", "Path to data for unit tests.");

DECLARE_string(vocabulary_path);

namespace swift_vio {
using CAMERA_GEOMETRY_T = okvis::cameras::PinholeCamera<
  okvis::cameras::RadialTangentialDistortion>;

class LCDFixture :public ::testing::Test {
 protected:
  // Tolerances
  const double tol = 1e-7;
  const double rot_tol = 0.05;
  const double tran_tol = 0.20;

 public:
  LCDFixture()
      : lcd_test_data_path_(FLAGS_test_data_path +
                            std::string("/ForLoopClosureDetector")),
        equalize_image_(false), epipolarThreshold_(3.0f),
        cameraSystem_(new okvis::cameras::NCameraSystem),
        frontend_(new okvis::Frontend(2, swift_vio::FrontendOptions())),
        cut_matches_lowe_(true),
        lcd_params_(new swift_vio::LoopClosureDetectorParams()),
        ref1_to_cur1_pose_(),
        ref2_to_cur2_pose_(),
        id_ref1_(0),
        id_cur1_(1),
        id_ref2_(2),
        id_cur2_(3),
        timestamp_ref1_(1000),
        timestamp_cur1_(2000),
        timestamp_ref2_(3000),
        timestamp_cur2_(4000) {
    lcd_params_->parseYAML(lcd_test_data_path_+"/testLCDParameters.yaml");
    FLAGS_vocabulary_path =
        FLAGS_test_data_path +
        std::string("/ForLoopClosureDetector/small_voc.yml.gz");
    lcd_detector_ = std::static_pointer_cast<LoopClosureDetector>(
          swift_vio::createLoopClosureMethod(lcd_params_));

    feature_matcher_ = cv::DescriptorMatcher::create(
        cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    T_WB_list_[0] = gtsam::Pose3(
        gtsam::Rot3(gtsam::Quaternion(0.338337, 0.608466, -0.535476, 0.478082)),
        gtsam::Point3(1.573832, 2.023348, 1.738755));

    T_WB_list_[1] = gtsam::Pose3(
        gtsam::Rot3(gtsam::Quaternion(0.478634, 0.415595, -0.700197, 0.328505)),
        gtsam::Point3(1.872115, 1.786064, 1.586159));

    T_WB_list_[2] = gtsam::Pose3(
        gtsam::Rot3(gtsam::Quaternion(0.3394, -0.672895, -0.492724, -0.435018)),
        gtsam::Point3(-0.662997, -0.495046, 1.347300));

    T_WB_list_[3] = gtsam::Pose3(
        gtsam::Rot3(gtsam::Quaternion(0.39266, -0.590667, -0.58023, -0.400326)),
        gtsam::Point3(-0.345638, -0.501712, 1.320441));

    ref1_to_cur1_pose_ = T_WB_list_[0].between(T_WB_list_[1]);
    ref2_to_cur2_pose_ = T_WB_list_[2].between(T_WB_list_[3]);

    initializeFrontend();
    initializeData();
    LOG(INFO) << "LCDFixture ready!";
  }

  // Reads image and converts to 1 channel image.
  cv::Mat readAndConvertToGrayScale(const std::string& img_name,
                                                 bool equalize) {
    cv::Mat img = cv::imread(img_name, cv::IMREAD_ANYCOLOR);
    if (img.channels() > 1) {
      LOG(WARNING) << "Converting img from BGR to GRAY...";
      cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }
    // Apply Histogram Equalization
    if (equalize) {
      LOG(WARNING) << "- Histogram Equalization for image: " << img_name;
      cv::equalizeHist(img, img);
    }
    return img;
  }

 protected:
  void initializeFrontend() {
    frontendParams_.detectionThreshold = 30;
    frontend_->setBriskDetectionOctaves(frontendParams_.detectionOctaves);
    frontend_->setBriskDetectionThreshold(frontendParams_.detectionThreshold);
    frontend_->setBriskDetectionMaximumKeypoints(
        frontendParams_.maxNoKeypoints);
    frontend_->setKeyframeInsertionOverlapThreshold(
        frontendParams_.keyframeInsertionOverlapThreshold);
    frontend_->setKeyframeInsertionMatchingRatioThreshold(
        frontendParams_.keyframeInsertionMatchingRatioThreshold);
  }

  bool removeIntraMatchOutliers(std::shared_ptr<const okvis::MultiFrame> nframe,
                                size_t camIdi, size_t camIdj,
                                std::vector<int>* i_indices,
                                std::vector<int>* j_indices,
                                float epipolarThreshold) const {
    float threshold2 = epipolarThreshold * epipolarThreshold;
    okvis::kinematics::Transformation T_CjCi =
        nframe->T_SC(camIdj)->inverse() * (*nframe->T_SC(camIdi));
    double focal_length =
        nframe
            ->geometryAs<CAMERA_GEOMETRY_T>(camIdi)
            ->focalLengthU();
    std::vector<bool> status(i_indices->size(), true);
    for (size_t k = 0; k < i_indices->size(); ++k) {
      Eigen::Vector2d keypoint_i;
      nframe->getKeypoint(camIdi, i_indices->at(k), keypoint_i);
      Eigen::Vector3d bearing_i;
      nframe
          ->geometryAs<CAMERA_GEOMETRY_T>(camIdi)
          ->backProject(keypoint_i, &bearing_i);

      Eigen::Vector2d keypoint_j;
      nframe->getKeypoint(camIdj, j_indices->at(k), keypoint_j);
      Eigen::Vector3d bearing_j;
      nframe
          ->geometryAs<CAMERA_GEOMETRY_T>(camIdj)
          ->backProject(keypoint_j, &bearing_j);

      float dist2 = okvis::TwoViewGeometry::computeErrorEssentialMat(
          T_CjCi, bearing_i, bearing_j, focal_length, focal_length);
      if (dist2 >= threshold2) {
        status[k] = false;
      }
    }
    swift_vio::removeUnsetElements(i_indices, status);
    swift_vio::removeUnsetElements(j_indices, status);
    return true;
  }

  /**
   * @brief initializeKeyframeLandmarks by triangulating from stereo matches.
   * @param nframe
   * @param camIdi
   * @param camIdj
   * @param i_indices
   * @param j_indices
   */
  Eigen::AlignedVector<Eigen::Vector4d> initializeKeyframeLandmarks(
      std::shared_ptr<okvis::MultiFrame> nframe, size_t camIdi, size_t camIdj,
      std::vector<int>* i_indices, std::vector<int>* j_indices) const {
    okvis::kinematics::Transformation T_CiCj =
        nframe->T_SC(camIdi)->inverse() * (*nframe->T_SC(camIdj));
    okvis::triangulation::ProbabilisticStereoTriangulator<CAMERA_GEOMETRY_T>
        probabilisticStereoTriangulator(nframe, nframe, camIdi, camIdj, T_CiCj);
    std::vector<bool> status(i_indices->size(), true);
    Eigen::AlignedVector<Eigen::Vector4d> triangulatedLandmarks;
    Eigen::Vector4d hP_Ci;
    bool canBeInitialized;  // It is essentially if two rays are NOT parallel.

    double fA = nframe->geometryAs<CAMERA_GEOMETRY_T>(camIdi)->focalLengthU();
    double fourthRoot2 = sqrt(sqrt(2));

    for (size_t k = 0; k < i_indices->size(); ++k) {
      double keypointAStdDev;
      nframe->getKeypointSize(camIdi, i_indices->at(k), keypointAStdDev);
      keypointAStdDev = 0.8 * keypointAStdDev / 12.0;
      double raySigma = fourthRoot2 * keypointAStdDev / fA;

      // valid tells if all involved Chi2's are small enough.
      bool valid = probabilisticStereoTriangulator.stereoTriangulate(
          i_indices->at(k), j_indices->at(k), hP_Ci, canBeInitialized, raySigma,
          true);

      if (valid && canBeInitialized) {
        status[k] = true;
        triangulatedLandmarks.push_back((*nframe->T_SC(camIdi)) * hP_Ci);
      }
    }
    swift_vio::removeUnsetElements(i_indices, status);
    swift_vio::removeUnsetElements(j_indices, status);
    return triangulatedLandmarks;
  }

  std::shared_ptr<LoopQueryKeyframeMessage>
  createLoopQueryKeyframeMessage(
      std::shared_ptr<okvis::MultiFrame> nframe,
      const std::vector<size_t>& neighborKeyframeIds =
          std::vector<size_t>(), bool uniform_weight = true) const {
    uint64_t keyframeId = nframe->id();
    okvis::kinematics::Transformation T_WBr = GtsamWrap::toTransform(T_WB_list_[keyframeId]);
    std::shared_ptr<LoopQueryKeyframeMessage> queryKeyframe(
        new LoopQueryKeyframeMessage(nframe->id(), nframe->timestamp(),
                                            T_WBr, nframe));
    if (uniform_weight) {
      queryKeyframe->setZeroCovariance();
    } else {
      queryKeyframe->setCovariance(lcd_detector_->uniform_noise_sigmas().asDiagonal() * sqrtInfoAliasFactor);
    }
    for (auto neighborKeyframeId : neighborKeyframeIds) {
      std::shared_ptr<okvis::MultiFrame> neighborFrame = stereo_frames_[neighborKeyframeId];
      okvis::kinematics::Transformation T_WBn = GtsamWrap::toTransform(T_WB_list_[neighborKeyframeId]);
      okvis::kinematics::Transformation T_BnBr = T_WBn.inverse() * T_WBr;
      std::shared_ptr<swift_vio::NeighborConstraintMessage> neighborMessage(
          new swift_vio::NeighborConstraintMessage(
              neighborFrame->id(), neighborFrame->timestamp(), T_BnBr, T_WBn));
      if (uniform_weight) {
        neighborMessage->core_.squareRootInfo_.setIdentity();
      } else {
        neighborMessage->core_.squareRootInfo_ =
            lcd_detector_->uniform_noise_sigmas().asDiagonal() * sqrtInfoAliasFactor;
        neighborMessage->cov_T_WB_.setIdentity();
        neighborMessage->cov_T_WBr_T_WB_.setZero();
      }
      queryKeyframe->odometryConstraintListMutable().push_back(neighborMessage);
    }
    queryKeyframe->keypointIndexForLandmarkListMutable() =
        matchedLeftKeypointIndices_[keyframeId];
    queryKeyframe->landmarkPositionListMutable() = triangulatedLandmarks_[keyframeId];
    return queryKeyframe;
  }

  void stereoMatchingAndInitialization(
      std::shared_ptr<okvis::MultiFrame> nframe) {
    // knnMatch + 2d2d RANSAC + stereo triangulation.
    bool visualize = false;
    std::string windowName = "stereo matches";
    std::vector<int> leftIndices;
    std::vector<int> rightIndices;
    double lowe_ratio = 1.0;
    if (cut_matches_lowe_) lowe_ratio = lcd_params_->lowe_ratio_;
    cv::Mat img_matches = nframe->computeIntraMatches(
        feature_matcher_, &leftIndices, &rightIndices, lowe_ratio, true);
    if (visualize) {
      cv::namedWindow(windowName);
      cv::imshow(windowName, img_matches);
      cv::waitKey(pauseMillisec);
    }

    removeIntraMatchOutliers(nframe, 0, 1, &leftIndices, &rightIndices, epipolarThreshold_);
    triangulatedLandmarks_[nframe->id()] =
        initializeKeyframeLandmarks(nframe, 0, 1, &leftIndices, &rightIndices);
    LOG(INFO) << "Triangulated 2d 2d stereo matches " << leftIndices.size();
    matchedLeftKeypointIndices_[nframe->id()] = leftIndices;

    std::vector<cv::DMatch> good_matches;
    good_matches.reserve(leftIndices.size());
    for (size_t k = 0 ; k < leftIndices.size(); ++k) {
        good_matches.emplace_back(leftIndices[k], rightIndices[k], 1.0f);
    }
    if (visualize) {
      cv::Mat img_matches_rest = nframe->drawStereoMatches(good_matches);
      cv::namedWindow(windowName);
      cv::imshow(windowName, img_matches_rest);
      cv::waitKey(pauseMillisec);
    }
  }

  void initializeKeyframe(okvis::Time stamp, uint64_t id,
                          const std::string& img_name_left,
                          const std::string& img_name_right,
                          const okvis::kinematics::Transformation& T_WB,
                          std::shared_ptr<okvis::MultiFrame>* nframe) {
    *nframe = std::make_shared<okvis::MultiFrame>(*cameraSystem_, stamp, id);
    (*nframe)->setImage(0,
                     readAndConvertToGrayScale(img_name_left, equalize_image_));
    (*nframe)->setImage(
        1, readAndConvertToGrayScale(img_name_right, equalize_image_));
    for (size_t i = 0; i < cameraSystem_->numCameras(); ++i) {
      okvis::kinematics::Transformation T_WC = T_WB * (*cameraSystem_->T_SC(i));
      frontend_->detectAndDescribe(i, *nframe, T_WC, nullptr);
    }
    stereoMatchingAndInitialization(*nframe);
  }

  void initializeCameraSystem() {
    Eigen::Matrix4d T_BC0_mat;
    T_BC0_mat << 0.0148655429818, -0.999880929698, 0.00414029679422,
        -0.0216401454975, 0.999557249008, 0.0149672133247, 0.025715529948,
        -0.064676986768, -0.0257744366974, 0.00375618835797, 0.999660727178,
        0.00981073058949, 0.0, 0.0, 0.0, 1.0;
    std::shared_ptr<okvis::kinematics::Transformation> T_BC0(
        new okvis::kinematics::Transformation(T_BC0_mat));
    std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry0(
        new CAMERA_GEOMETRY_T(
            752, 480, 458.654, 457.296, 367.215, 248.375,
            okvis::cameras::RadialTangentialDistortion(
                -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05),
            0.0, 0.0));
    Eigen::Matrix4d T_BC1_mat;
    T_BC1_mat << 0.0125552670891, -0.999755099723, 0.0182237714554,
        -0.0198435579556, 0.999598781151, 0.0130119051815, 0.0251588363115,
        0.0453689425024, -0.0253898008918, 0.0179005838253, 0.999517347078,
        0.00786212447038, 0.0, 0.0, 0.0, 1.0;
    std::shared_ptr<okvis::kinematics::Transformation> T_BC1(
        new okvis::kinematics::Transformation(T_BC1_mat));
    std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry1(
        new CAMERA_GEOMETRY_T(
            752, 480, 457.587, 456.134, 379.999, 255.238,
            okvis::cameras::RadialTangentialDistortion(
                -0.28368365, 0.07451284, -0.00010473, -3.55590700e-05),
            0.0, 0.0));

    okvis::cameras::NCameraSystem::DistortionType distortionId =
        okvis::cameras::NCameraSystem::DistortionType::RadialTangential;

    bool computeExpensiveOverlaps = false;
    cameraSystem_->addCamera(T_BC0, cameraGeometry0, distortionId, "", "",
                             computeExpensiveOverlaps);
    cameraSystem_->addCamera(T_BC1, cameraGeometry1, distortionId, "", "",
                             computeExpensiveOverlaps);
  }

  void initializeData() {
    std::string img_name_ref1_left = lcd_test_data_path_ + "/left_img_0.png";
    std::string img_name_ref1_right = lcd_test_data_path_ + "/right_img_0.png";

    std::string img_name_cur1_left = lcd_test_data_path_ + "/left_img_1.png";
    std::string img_name_cur1_right = lcd_test_data_path_ + "/right_img_1.png";

    std::string img_name_ref2_left = lcd_test_data_path_ + "/left_img_2.png";
    std::string img_name_ref2_right = lcd_test_data_path_ + "/right_img_2.png";

    std::string img_name_cur2_left = lcd_test_data_path_ + "/left_img_3.png";
    std::string img_name_cur2_right = lcd_test_data_path_ + "/right_img_3.png";

    initializeCameraSystem();

    okvis::kinematics::Transformation ref1_T_WB = GtsamWrap::toTransform(T_WB_list_[0]);
    initializeKeyframe(timestamp_ref1_, id_ref1_, img_name_ref1_left,
                       img_name_ref1_right, ref1_T_WB, &stereo_frames_[0]);

    okvis::kinematics::Transformation cur1_T_WB = GtsamWrap::toTransform(T_WB_list_[1]);
    initializeKeyframe(timestamp_cur1_, id_cur1_, img_name_cur1_left,
                       img_name_cur1_right, cur1_T_WB, &stereo_frames_[1]);

    okvis::kinematics::Transformation ref2_T_WB = GtsamWrap::toTransform(T_WB_list_[2]);
    initializeKeyframe(timestamp_ref2_, id_ref2_, img_name_ref2_left,
                       img_name_ref2_right, ref2_T_WB, &stereo_frames_[2]);

    okvis::kinematics::Transformation cur2_T_WB = GtsamWrap::toTransform(T_WB_list_[3]);
    initializeKeyframe(timestamp_cur2_, id_cur2_, img_name_cur2_left,
                       img_name_cur2_right, cur2_T_WB, &stereo_frames_[3]);
  }

  // Standard gtest methods, unnecessary for now
  virtual void SetUp() {}
  virtual void TearDown() {}

 protected:
  // Data-related members
  std::string lcd_test_data_path_;

  okvis::Optimization frontendParams_;
  bool equalize_image_;
  float epipolarThreshold_;
  std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem_;
  std::shared_ptr<okvis::Frontend> frontend_;      ///< The frontend to detect and describe image keypoints.

  // LCD members
  bool cut_matches_lowe_;
  std::shared_ptr<LoopClosureDetectorParams> lcd_params_;
  std::shared_ptr<LoopClosureDetector> lcd_detector_;
  cv::Ptr<cv::DescriptorMatcher> feature_matcher_;

  // Stored frame members
  gtsam::Pose3 T_WB_list_[4]; // ref1, cur1. ref2, cur2
  gtsam::Pose3 ref1_to_cur1_pose_, ref2_to_cur2_pose_;
  std::shared_ptr<okvis::MultiFrame> stereo_frames_[4]; // ref1, cur1. ref2, cur2
  Eigen::AlignedVector<Eigen::Vector4d> triangulatedLandmarks_[4]; // ref1, cur1, ref2, cur2, triangulated landmark positions in the body frame.
  std::vector<int> matchedLeftKeypointIndices_[4]; // ref1, cur1, ref2, cur2

  const FrameId id_ref1_;
  const FrameId id_cur1_;
  const FrameId id_ref2_;
  const FrameId id_cur2_;

  const swift_vio::Timestamp timestamp_ref1_;
  const swift_vio::Timestamp timestamp_cur1_;
  const swift_vio::Timestamp timestamp_ref2_;
  const swift_vio::Timestamp timestamp_cur2_;
  const int pauseMillisec = 2000;
  const size_t totalKeyframes = 4u;
  const double sqrtInfoAliasFactor = 5.0;
};  // class LCDFixture

TEST_F(LCDFixture, defaultConstructor) {
  /* Test default constructor to ensure that vocabulary is loaded correctly. */
  EXPECT_GT(lcd_detector_->getBoWDatabase()->getVocabulary()->size(), 0u);
}

TEST_F(LCDFixture, detectLoop) {
  std::pair<double, double> error;

  /* Test the detectLoop method against two images without closure */
  std::shared_ptr<swift_vio::LoopFrameAndMatches> output_0;
  std::shared_ptr<swift_vio::KeyframeInDatabase> queryKeyframeInDatabase;

  lcd_detector_->detectLoop(
      createLoopQueryKeyframeMessage(stereo_frames_[2]),
      queryKeyframeInDatabase, output_0);
  EXPECT_EQ(output_0.get(), nullptr);

  std::shared_ptr<swift_vio::LoopFrameAndMatches> output_1;
  lcd_detector_->detectLoop(
      createLoopQueryKeyframeMessage(stereo_frames_[0]),
      queryKeyframeInDatabase, output_1);
  EXPECT_TRUE(lcd_detector_->isUniformWeight());
  EXPECT_EQ(output_1.get(), nullptr);

  /* Test the detectLoop method against two images that are identical */
  std::shared_ptr<swift_vio::LoopFrameAndMatches> output_2;
  lcd_detector_->detectLoop(
      createLoopQueryKeyframeMessage(stereo_frames_[0]),
      queryKeyframeInDatabase, output_2);
  EXPECT_TRUE(lcd_detector_->isUniformWeight());
  EXPECT_NE(output_2.get(), nullptr);
  EXPECT_EQ(output_2->id_, 0u);
  EXPECT_EQ(output_2->queryKeyframeId_, 0u);
  error = computeRotationAndTranslationErrors(
      gtsam::Pose3(), swift_vio::GtsamWrap::toPose3(output_2->T_BlBq_), true);
  EXPECT_LT(error.first, rot_tol);
  EXPECT_LT(error.second, tran_tol);
  EXPECT_LT((output_2->relativePoseSqrtInfo().diagonal() -
             lcd_detector_->uniform_noise_sigmas())
                .lpNorm<Eigen::Infinity>(),
            1e-7);

  /* Test the detectLoop method against two unidentical, similar images */
  std::shared_ptr<swift_vio::LoopFrameAndMatches> output_3;
  lcd_detector_->detectLoop(
      createLoopQueryKeyframeMessage(stereo_frames_[1]),
      queryKeyframeInDatabase, output_3);
  EXPECT_TRUE(lcd_detector_->isUniformWeight());
  EXPECT_NE(output_3.get(), nullptr);
  if (output_3.get()) {
    EXPECT_EQ(output_3->id_, 0u);
    EXPECT_EQ(output_3->queryKeyframeId_, 1u);
    error = computeRotationAndTranslationErrors(
        ref1_to_cur1_pose_, swift_vio::GtsamWrap::toPose3(output_3->T_BlBq_), true);
    LOG(INFO) << "Loop frame relative pose estimation error " << error.first
              << " " << error.second;
    EXPECT_LT(error.first, rot_tol);
    EXPECT_LT(error.second, tran_tol);
    EXPECT_LT((output_3->relativePoseSqrtInfo().diagonal() -
               lcd_detector_->uniform_noise_sigmas())
                  .lpNorm<Eigen::Infinity>(),
              1e-7);
  }
}

TEST_F(LCDFixture, addOdometryFactors) {
  /* Test the addition of odometry factors to the PGO */
  lcd_detector_->initializePGO();

  gtsam::Values pgo_trajectory = lcd_detector_->getPGOTrajectory();
  gtsam::NonlinearFactorGraph pgo_nfg = lcd_detector_->getPGOnfg();

  EXPECT_EQ(pgo_trajectory.size(), 1u);
  EXPECT_EQ(pgo_nfg.size(), 0u);
}

TEST_F(LCDFixture, spinOnce) {
  std::vector<size_t> processOrder{
      0, 2, 1, 3};  // frame index of keyframes to be provided to the loop
                    // closure module, namely, ref1, ref2, cur1, cur2
  std::vector<std::vector<size_t>> neighborKeyframeIds{
      {},
      {0},
      {2, 0},
      {1, 2}};  // frame index of neighboring keyframes for each query keyframe.
                // note the most adjacent neighbor is at the front.
  std::shared_ptr<LoopQueryKeyframeMessage>
      kfs[totalKeyframes];  // query keyframes fed to the loop closure module.
  std::shared_ptr<LoopFrameAndMatches>
      outputs[totalKeyframes];  // outputs for each query keyframe.

  size_t j = 0u;
  std::vector<size_t> expected_nfg_increment{1, 1, 2 + 1, 2 + 1};
  size_t expected_nfg_size = expected_nfg_increment[j];
  for (auto i : processOrder) {
    kfs[j] = createLoopQueryKeyframeMessage(stereo_frames_[i],
    neighborKeyframeIds[j], false);
    std::shared_ptr<swift_vio::KeyframeInDatabase> queryKeyframeInDatabase;
    LOG(INFO) << "Detect loop for keyframe " << i << " order " << j;
    lcd_detector_->detectLoop(kfs[j], queryKeyframeInDatabase, outputs[j]);
    EXPECT_FALSE(lcd_detector_->isUniformWeight());
    PgoResult pgoResult;
    LOG(INFO) << "Add constraints for keyframe " << i;
    lcd_detector_->addConstraintsAndOptimize(*queryKeyframeInDatabase, outputs[j], pgoResult);
    EXPECT_EQ(lcd_detector_->getPGOTrajectory().size(), j + 1);
    EXPECT_EQ(lcd_detector_->getPGOnfg().size(), expected_nfg_size);
    ++j;
    expected_nfg_size += expected_nfg_increment[j];
  }

  EXPECT_EQ(outputs[0], nullptr);

  EXPECT_EQ(outputs[1], nullptr);
  size_t outputIndex = 2u;
  EXPECT_NE(outputs[outputIndex], nullptr);
  if (outputs[outputIndex]) {
    EXPECT_EQ(outputs[outputIndex]->id_, 0u);
    EXPECT_EQ(outputs[outputIndex]->dbowId_, 0u);
    EXPECT_EQ(outputs[outputIndex]->stamp_, timestamp_ref1_);
    EXPECT_EQ(outputs[outputIndex]->queryKeyframeId_, 1u);
    EXPECT_EQ(outputs[outputIndex]->queryKeyframeDbowId_, 2u);
    EXPECT_EQ(outputs[outputIndex]->queryKeyframeStamp_, timestamp_cur1_);

    std::pair<double, double> error = computeRotationAndTranslationErrors(
        ref1_to_cur1_pose_, swift_vio::GtsamWrap::toPose3(outputs[outputIndex]->T_BlBq_), true);
    LOG(INFO) << "Loop frame relative pose ref1_to_cur1_pose_ PnP error "
              << error.first << " " << error.second;

    EXPECT_LT(error.first, rot_tol);
    EXPECT_LT(error.second, tran_tol);
  }
  ++outputIndex;
  EXPECT_NE(outputs[outputIndex], nullptr);
  if (outputs[outputIndex]) {
    EXPECT_EQ(outputs[outputIndex]->id_, 2u);
    EXPECT_EQ(outputs[outputIndex]->dbowId_, 1u);
    EXPECT_EQ(outputs[outputIndex]->stamp_, timestamp_ref2_);
    EXPECT_EQ(outputs[outputIndex]->queryKeyframeId_, 3u);
    EXPECT_EQ(outputs[outputIndex]->queryKeyframeDbowId_, 3u);
    EXPECT_EQ(outputs[outputIndex]->queryKeyframeStamp_, timestamp_cur2_);

    std::pair<double, double> error = computeRotationAndTranslationErrors(
        ref2_to_cur2_pose_, swift_vio::GtsamWrap::toPose3(outputs[outputIndex]->T_BlBq_), true);
    LOG(INFO) << "Loop frame relative pose ref1_to_cur1_pose_ PnP error "
              << error.first << " " << error.second;
    EXPECT_GT((outputs[outputIndex]->relativePoseSqrtInfo().diagonal() -
               lcd_detector_->uniform_noise_sigmas())
                  .lpNorm<Eigen::Infinity>(),
              1);
    LOG(INFO) << "relative pose sqrt info\n" << outputs[outputIndex]->relativePoseSqrtInfo();

    EXPECT_LT(error.first, rot_tol);
    EXPECT_LT(error.second, tran_tol);
  }
  gtsam::Values pgo_trajectory = lcd_detector_->getPGOTrajectory();
  gtsam::NonlinearFactorGraph pgo_nfg = lcd_detector_->getPGOnfg();
  for (size_t j = 0; j < totalKeyframes; ++j) {
    gtsam::Pose3 T_WB = pgo_trajectory.at<gtsam::Pose3>(gtsam::Symbol(j));
    std::pair<double, double> error = computeRotationAndTranslationErrors(
        T_WB_list_[processOrder[j]], T_WB, true);
    LOG(INFO) << "PGO relative pose error for " << processOrder[j]
              << " dbowid " << j << " " << error.first << " " << error.second;
    EXPECT_LT(error.first, rot_tol) << "rotation";
    EXPECT_LT(error.second, tran_tol) << "translation";
  }
}

}  // namespace swift_vio
