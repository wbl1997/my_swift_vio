#ifndef INCLUDE_OKVIS_VIOFRAMEMATCHINGALGORITHM_HPP_
#define INCLUDE_OKVIS_VIOFRAMEMATCHINGALGORITHM_HPP_

#include <memory>

#include <okvis/DenseMatcher.hpp>
#include <okvis/MatchingAlgorithm.hpp>

#include <okvis/msckf2.hpp>

#include <brisk/internal/hamming.h>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/triangulation/ProbabilisticStereoTriangulator.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

/**
 * \brief A MatchingAlgorithm implementation
 * \tparam CAMERA_GEOMETRY_T Camera geometry model. See also
 * okvis::cameras::CameraBase.
 */
template <class CAMERA_GEOMETRY_T>
class VioFrameMatchingAlgorithm : public okvis::MatchingAlgorithm {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  typedef CAMERA_GEOMETRY_T camera_geometry_t;

  enum MatchingTypes {
    Match3D2D = 1,  ///< Match 3D position of established landmarks to 2D
                    ///< keypoint position
    Match2D2D = 2   ///< Match 2D position of established landmarks to 2D
                    ///< keypoint position
  };

  /**
   * @brief Constructor.
   * @param estimator           HybridFilter.
   * @param matchingType        Matching type. See MatchingTypes enum.
   * @param distanceThreshold   Descriptor distance threshold.
   * @param usePoseUncertainty  Use the pose uncertainty for matching.
   */
  VioFrameMatchingAlgorithm(
#ifdef USE_MSCKF2
      okvis::MSCKF2& estimator,
#else
      okvis::HybridFilter& estimator,
#endif
      int matchingType, float distanceThreshold,
      bool usePoseUncertainty = true);

  virtual ~VioFrameMatchingAlgorithm();

  /**
   * @brief Set which frames to match.
   * @param mfIdA   The multiframe ID to match against.
   * @param mfIdB   The new multiframe ID.
   * @param camIdA  ID of the frame inside multiframe A to match.
   * @param camIdB  ID of the frame inside multiframe B to match.
   */
  void setFrames(uint64_t mfIdA, uint64_t mfIdB, size_t camIdA, size_t camIdB);

  /**
   * \brief Set the matching type.
   * \see MatchingTypes
   */
  void setMatchingType(int matchingType);

  /// \brief This will be called exactly once for each call to
  /// DenseMatcher::match().
  virtual void doSetup();

  /// \brief What is the size of list A?
  virtual size_t sizeA() const;
  /// \brief What is the size of list B?
  virtual size_t sizeB() const;

  /// \brief Get the distance threshold for which matches exceeding it will not
  /// be returned as matches.
  virtual float distanceThreshold() const;
  /// \brief Set the distance threshold for which matches exceeding it will not
  /// be returned as matches.
  void setDistanceThreshold(float distanceThreshold);

  /// \brief Should we skip the item in list A? This will be called once for
  /// each item in the list
  virtual bool skipA(size_t indexA) const { return skipA_[indexA]; }

  /// \brief Should we skip the item in list B? This will be called many times.
  virtual bool skipB(size_t indexB) const { return skipB_[indexB]; }

  /**
   * @brief Calculate the distance between two keypoints.
   * @param indexA Index of the first keypoint.
   * @param indexB Index of the other keypoint.
   * @return Distance between the two keypoint descriptors.
   * @remark Points that absolutely don't match will return float::max.
   */
  virtual float distance(size_t indexA, size_t indexB) const {
    OKVIS_ASSERT_LT_DBG(MatchingAlgorithm::Exception, indexA, sizeA(),
                        "index A out of bounds");
    OKVIS_ASSERT_LT_DBG(MatchingAlgorithm::Exception, indexB, sizeB(),
                        "index B out of bounds");
    const float dist = static_cast<float>(specificDescriptorDistance(
        frameA_->keypointDescriptor(camIdA_, indexA),
        frameB_->keypointDescriptor(camIdB_, indexB)));

    if (dist < distanceThreshold_) {
      if (verifyMatch(indexA, indexB)) return dist;
    }
    return std::numeric_limits<float>::max();
  }

  /// \brief Geometric verification of a match.
  bool verifyMatch(size_t indexA, size_t indexB) const;

  /// \brief A function that tells you how many times setMatching() will be
  /// called. \warning Currently not implemented to do anything.
  virtual void reserveMatches(size_t numMatches);

  /// \brief At the end of the matching step, this function is called once
  ///        for each pair of matches discovered.
  virtual void setBestMatch(size_t indexA, size_t indexB, double distance);

  /// \brief Get the number of matches.
  size_t numMatches();
  /// \brief Get the number of uncertain matches.
  size_t numUncertainMatches();

  /// \brief access the matching result.
  const okvis::Matches& getMatches() const;

  /// \brief assess the validity of the relative uncertainty computation.
  bool isRelativeUncertaintyValid() { return validRelativeUncertainty_; }

 private:
  /// \brief This is essentially the map.
#ifdef USE_MSCKF2
  okvis::MSCKF2* estimator_;
#else
  okvis::HybridFilter* estimator_;
#endif

  /// \name Which frames to take
  /// \{
  uint64_t mfIdA_ = 0;
  uint64_t mfIdB_ = 0;
  size_t camIdA_ = 0;
  size_t camIdB_ = 0;

  std::shared_ptr<okvis::MultiFrame> frameA_;
  std::shared_ptr<okvis::MultiFrame> frameB_;
  /// \}

  /// Distances above this threshold will not be returned as matches.
  float distanceThreshold_;

  /// \name Store some transformations that are often used
  /// \{
  /// use a fully relative formulation
  okvis::kinematics::Transformation T_CaCb_;
  okvis::kinematics::Transformation T_CbCa_;
  okvis::kinematics::Transformation T_SaCa_;
  okvis::kinematics::Transformation T_SbCb_;
  okvis::kinematics::Transformation T_WSa_;
  okvis::kinematics::Transformation T_WSb_;
  okvis::kinematics::Transformation T_SaW_;
  okvis::kinematics::Transformation T_SbW_;
  okvis::kinematics::Transformation T_WCa_;
  okvis::kinematics::Transformation T_WCb_;
  okvis::kinematics::Transformation T_CaW_;
  okvis::kinematics::Transformation T_CbW_;
  /// \}

  /// The number of matches.
  size_t numMatches_ = 0;
  /// The number of uncertain matches.
  size_t numUncertainMatches_ = 0;

  /// Focal length of camera used in frame A.
  double fA_ = 0;
  /// Focal length of camera used in frame B.
  double fB_ = 0;

  /// Stored the matching type. See MatchingTypes().
  int matchingType_;

  /// temporarily store all projections
  Eigen::Matrix<double, Eigen::Dynamic, 2> projectionsIntoB_;
  /// temporarily store all projection uncertainties
  Eigen::Matrix<double, Eigen::Dynamic, 2> projectionsIntoBUncertainties_;

  /// Should keypoint[index] in frame A be skipped
  std::vector<bool> skipA_;
  /// Should keypoint[index] in frame B be skipped
  std::vector<bool> skipB_;

  /// Camera center of frame A.
  Eigen::Vector3d pA_W_;
  /// Camera center of frame B.
  Eigen::Vector3d pB_W_;

  /// Temporarily store ray sigmas of frame A.
  std::vector<double> raySigmasA_;
  /// Temporarily store ray sigmas of frame B.
  std::vector<double> raySigmasB_;

  /// Stereo triangulator.
  okvis::triangulation::ProbabilisticStereoTriangulator<camera_geometry_t>
      probabilisticStereoTriangulator_;

  bool validRelativeUncertainty_ = false;
  bool usePoseUncertainty_ = false;

  /// \brief Calculates the distance between two descriptors.
  // copy from BriskDescriptor.hpp
  uint32_t specificDescriptorDistance(const unsigned char* descriptorA,
                                      const unsigned char* descriptorB) const {
    OKVIS_ASSERT_TRUE_DBG(
        Exception, descriptorA != NULL && descriptorB != NULL,
        "Trying to compare a descriptor with a null description vector");

    return brisk::Hamming::PopcntofXORed(descriptorA, descriptorB,
                                         3 /*48 / 16*/);
  }

 public:
  virtual void match();
};

}  // namespace okvis

#endif /* INCLUDE_OKVIS_VIOFRAMEMATCHINGALGORITHM_HPP_ */
