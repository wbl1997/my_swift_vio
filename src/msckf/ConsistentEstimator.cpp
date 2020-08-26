/**
 * @file ConsistentEstimator.cpp
 * @brief Source file for the ConsistentEstimator class.
 * @author Jianzhu Huai
 */

#include <msckf/EuclideanParamBlock.hpp>
#include <msckf/ConsistentEstimator.hpp>
#include <msckf/CameraTimeParamBlock.hpp>

#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/assert_macros.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

// Constructor if a ceres map is already available.
ConsistentEstimator::ConsistentEstimator(
    std::shared_ptr<okvis::ceres::Map> mapPtr)
    : Estimator(mapPtr)
{
}

// The default constructor.
ConsistentEstimator::ConsistentEstimator()
    : Estimator()
{
}

ConsistentEstimator::~ConsistentEstimator()
{
}

bool ConsistentEstimator::addReprojectionFactors() {
  okvis::cameras::NCameraSystem::DistortionType distortionType =
      camera_rig_.getDistortionType(0);
  for (PointMap::iterator pit = landmarksMap_.begin();
       pit != landmarksMap_.end(); ++pit) {
//    if (pit->second.residualizeCase == NotInState_NotTrackedNow) {
      // check the parallax of the landmark
      // if the landmark has sufficient parallax, retriangulate it,
      // and add all its observations to the estimator, and reset its status
      pit->second.residualizeCase = InState_TrackedNow;
      // or else do nothing

//    } else {
      // examine starting from the rear of a landmark's observations, add
      // reprojection factors for those with null residual pointers, terminate
      // until a valid residual pointer is hit.
      MapPoint& mp = pit->second;
      std::map<okvis::KeypointIdentifier, uint64_t>::reverse_iterator
          breakIter = mp.observations.rend();
      for (std::map<okvis::KeypointIdentifier, uint64_t>::reverse_iterator
               riter = mp.observations.rbegin();
           riter != mp.observations.rend(); ++riter) {
        ::ceres::ResidualBlockId retVal = 0u;
        if (riter->second == 0u) {
// TODO(jhuai): Placing the switch statement outside the double for loops saves
// most branchings of switch.
#define DISTORTION_MODEL_CASE(camera_geometry_t)                               \
  retVal = addPointFrameResidual<camera_geometry_t>(pit->first, riter->first); \
  riter->second = reinterpret_cast<uint64_t>(retVal);

          switch (distortionType) { DISTORTION_MODEL_SWITCH_CASES }

#undef DISTORTION_MODEL_CASE
        } else {
          breakIter = riter;
          break;
        }
      }

      for (std::map<okvis::KeypointIdentifier, uint64_t>::reverse_iterator
               riter = breakIter;
           riter != mp.observations.rend(); ++riter) {
        OKVIS_ASSERT_NE_DBG(
            Exception, riter->second, 0u,
            "Residuals should be contiguous unless epipolar factors are used!");
      }
//    }
  }
  return true;
}

// Start ceres optimization.
#ifdef USE_OPENMP
void ConsistentEstimator::optimize(size_t numIter, size_t numThreads,
                                 bool verbose)
#else
void ConsistentEstimator::optimize(size_t numIter, size_t /*numThreads*/,
                                 bool verbose) // avoid warning since numThreads unused
#warning openmp not detected, your system may be slower than expected
#endif
{
  // assemble options
  mapPtr_->options.linear_solver_type = ::ceres::SPARSE_SCHUR;
  //mapPtr_->options.initial_trust_region_radius = 1.0e4;
  //mapPtr_->options.initial_trust_region_radius = 2.0e6;
  //mapPtr_->options.preconditioner_type = ::ceres::IDENTITY;
  mapPtr_->options.trust_region_strategy_type = ::ceres::DOGLEG;
  //mapPtr_->options.trust_region_strategy_type = ::ceres::LEVENBERG_MARQUARDT;
  //mapPtr_->options.use_nonmonotonic_steps = true;
  //mapPtr_->options.max_consecutive_nonmonotonic_steps = 10;
  //mapPtr_->options.function_tolerance = 1e-12;
  //mapPtr_->options.gradient_tolerance = 1e-12;
  //mapPtr_->options.jacobi_scaling = false;
#ifdef USE_OPENMP
    mapPtr_->options.num_threads = numThreads;
#endif
  mapPtr_->options.max_num_iterations = numIter;

  if (verbose) {
    mapPtr_->options.minimizer_progress_to_stdout = true;
  } else {
    mapPtr_->options.minimizer_progress_to_stdout = false;
  }
  addReprojectionFactors();
  // call solver
  mapPtr_->solve();

  // update landmarks
  {
    for(auto it = landmarksMap_.begin(); it!=landmarksMap_.end(); ++it){
      Eigen::MatrixXd H(3,3);
      mapPtr_->getLhs(it->first,H);
      Eigen::SelfAdjointEigenSolver< Eigen::Matrix3d > saes(H);
      Eigen::Vector3d eigenvalues = saes.eigenvalues();
      const double smallest = (eigenvalues[0]);
      const double largest = (eigenvalues[2]);
      if(smallest<1.0e-12){
        // this means, it has a non-observable depth
        it->second.quality = 0.0;
      } else {
        // OK, well constrained
        it->second.quality = sqrt(smallest)/sqrt(largest);
      }

      // update coordinates
      it->second.pointHomog = std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(
          mapPtr_->parameterBlockPtr(it->first))->estimate();
    }
  }

  // summary output
  if (verbose) {
    LOG(INFO) << mapPtr_->summary.FullReport();
  }
}

}  // namespace okvis
