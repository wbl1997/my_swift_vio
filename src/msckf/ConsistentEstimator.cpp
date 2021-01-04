/**
 * @file ConsistentEstimator.cpp
 * @brief Source file for the ConsistentEstimator class.
 * @author Jianzhu Huai
 */


#include <msckf/ConsistentEstimator.hpp>

#include <msckf/CameraTimeParamBlock.hpp>
#include <msckf/EuclideanParamBlock.hpp>
#include <msckf/FeatureTriangulation.hpp>
#include <msckf/memory.h>

#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/assert_macros.hpp>

DECLARE_double(ray_sigma_scalar);

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

bool ConsistentEstimator::triangulateWithDisparityCheck(
    uint64_t lmkId, Eigen::Matrix<double, 4, 1>* hpW,
    double focalLength, double raySigmaScalar) const {
  const MapPoint& mp = landmarksMap_.at(lmkId);
  AlignedVector<Eigen::Vector3d> obsDirections;
  AlignedVector<okvis::kinematics::Transformation> T_CWs;
  std::vector<double> imageNoiseStd;
  size_t numObs = gatherMapPointObservations(mp, &obsDirections, &T_CWs, &imageNoiseStd);
  if (numObs < optimizationOptions_.minTrackLength) {
    return false;
  }
  if (msckf::hasLowDisparity(obsDirections, T_CWs, imageNoiseStd, focalLength, raySigmaScalar))
    return false;
  *hpW = msckf::triangulateHomogeneousDLT(obsDirections, T_CWs);
  *hpW /= hpW->w();
  return true;
}

bool ConsistentEstimator::addLandmarkToGraph(uint64_t lmkId,
                                             const Eigen::Vector4d& hpW) {
  okvis::cameras::NCameraSystem::DistortionType distortionType =
      camera_rig_.getDistortionType(0);
  ::ceres::ResidualBlockId retVal = 0u;
  uint64_t minValidStateId = statesMap_.begin()->first;
  okvis::MapPoint& mp = landmarksMap_.at(lmkId);

  std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock>
      pointParameterBlock(
          new okvis::ceres::HomogeneousPointParameterBlock(hpW, lmkId));
  if (!mapPtr_->addParameterBlock(pointParameterBlock,
                                  okvis::ceres::Map::HomogeneousPoint))
    return false;

  for (std::map<okvis::KeypointIdentifier, uint64_t>::iterator obsIter =
           mp.observations.begin();
       obsIter != mp.observations.end(); ++obsIter) {
    if (obsIter->first.frameId < minValidStateId) {
      // Some observations may be outside the horizon.
      continue;
    }

#define DISTORTION_MODEL_CASE(camera_geometry_t)                            \
  retVal = addPointFrameResidual<camera_geometry_t>(lmkId, obsIter->first); \
  obsIter->second = reinterpret_cast<uint64_t>(retVal);

    switch (distortionType) { DISTORTION_MODEL_SWITCH_CASES }

#undef DISTORTION_MODEL_CASE
  }

  mp.residualizeCase = InState_TrackedNow;
  return true;
}

bool ConsistentEstimator::addReprojectionFactors() {
  okvis::cameras::NCameraSystem::DistortionType distortionType =
      camera_rig_.getDistortionType(0);
  Eigen::VectorXd intrinsics;
  camera_rig_.getCameraGeometry(0)->getIntrinsics(intrinsics);
  double focalLength = intrinsics[0];
  uint64_t minValidStateId = statesMap_.begin()->first;
  for (PointMap::iterator pit = landmarksMap_.begin();
       pit != landmarksMap_.end(); ++pit) {
    if (pit->second.residualizeCase == NotInState_NotTrackedNow) {
      // remove observations outside the sliding window.
      // because in applyMarginalizationStrategy, we do not delete landmarks
      // that are in NotInState_NotTrackedNow status
      // and have zero residuals, so we have to delete these old observations
      // and landmarks of zero observations here.
      for (std::map<KeypointIdentifier, uint64_t>::iterator obsIter =
               pit->second.observations.begin();
           obsIter != pit->second.observations.end(); ) {
        if (obsIter->first.frameId < minValidStateId) {
          obsIter = pit->second.observations.erase(obsIter);
        } else {
          ++obsIter;
        }
      }

      Eigen::Vector4d hpW;
      bool triangulateOk = triangulateWithDisparityCheck(
          pit->first, &hpW, focalLength, FLAGS_ray_sigma_scalar);
      if (triangulateOk) {
        addLandmarkToGraph(pit->first, hpW);
      } // else do nothing
    } else {
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
    }
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
      if (it->second.residualizeCase == InState_TrackedNow) {
        Eigen::MatrixXd H(3, 3);
        mapPtr_->getLhs(it->first, H);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(H);
        Eigen::Vector3d eigenvalues = saes.eigenvalues();
        const double smallest = (eigenvalues[0]);
        const double largest = (eigenvalues[2]);
        if (smallest < 1.0e-12) {
          // this means, it has a non-observable depth
          it->second.quality = 0.0;
        } else {
          // OK, well constrained
          it->second.quality = sqrt(smallest) / sqrt(largest);
        }

        // update coordinates
        it->second.pointHomog =
            std::static_pointer_cast<
                okvis::ceres::HomogeneousPointParameterBlock>(
                mapPtr_->parameterBlockPtr(it->first))
                ->estimate();
      }
    }
  }

  // summary output
  if (verbose) {
    LOG(INFO) << mapPtr_->summary.FullReport();
  }
}

// Applies the dropping/marginalization strategy according to the RSS'13/IJRR'14 paper.
// The new number of frames in the window will be numKeyframes+numImuFrames.
bool ConsistentEstimator::applyMarginalizationStrategy(
    size_t numKeyframes, size_t numImuFrames,
    okvis::MapPointVector& removedLandmarks) {
  // keep the newest numImuFrames
  std::map<uint64_t, States>::reverse_iterator rit = statesMap_.rbegin();
  for(size_t k=0; k<numImuFrames; k++){
    rit++;
    if(rit==statesMap_.rend()){
      // nothing to do.
      return true;
    }
  }

  // remove linear marginalizationError, if existing
  if (marginalizationErrorPtr_ && marginalizationResidualId_) {
    bool success = mapPtr_->removeResidualBlock(marginalizationResidualId_);
    OKVIS_ASSERT_TRUE_DBG(Exception, success,
                       "could not remove marginalization error");
    marginalizationResidualId_ = 0;
    if (!success)
      return false;
  }

  // these will keep track of what we want to marginalize out.
  std::vector<uint64_t> paremeterBlocksToBeMarginalized;
  std::vector<bool> keepParameterBlocks;

  if (!marginalizationErrorPtr_) {
    marginalizationErrorPtr_.reset(
        new ceres::MarginalizationError(*mapPtr_.get()));
  }

  // distinguish if we marginalize everything or everything but pose
  std::vector<uint64_t> removeFrames;
  std::vector<uint64_t> removeAllButPose;
  std::vector<uint64_t> allLinearizedFrames;
  size_t countedKeyframes = 0;
  while (rit != statesMap_.rend()) {
    if (!rit->second.isKeyframe || countedKeyframes >= numKeyframes) {
      removeFrames.push_back(rit->second.id);
    } else {
      countedKeyframes++;
    }
    removeAllButPose.push_back(rit->second.id);
    allLinearizedFrames.push_back(rit->second.id);
    ++rit;// check the next frame
  }

  // marginalize everything but pose:
  for(size_t k = 0; k<removeAllButPose.size(); ++k){
    std::map<uint64_t, States>::iterator it = statesMap_.find(removeAllButPose[k]);
    for (size_t i = 0; i < it->second.global.size(); ++i) {
      if (i == GlobalStates::T_WS) {
        continue; // we do not remove the pose here.
      }
      if (!it->second.global[i].exists) {
        continue; // if it doesn't exist, we don't do anything.
      }
      if (mapPtr_->parameterBlockPtr(it->second.global[i].id)->fixed()) {
        continue;  // we never eliminate fixed blocks.
      }
      std::map<uint64_t, States>::iterator checkit = it;
      checkit++;
      // only get rid of it, if it's different
      if(checkit->second.global[i].exists &&
          checkit->second.global[i].id == it->second.global[i].id){
        continue;
      }
      it->second.global[i].exists = false; // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.global[i].id);
      keepParameterBlocks.push_back(false);
      ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(
          it->second.global[i].id);
      for (size_t r = 0; r < residuals.size(); ++r) {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
            residuals[r].errorInterfacePtr);
        if(!reprojectionError){   // we make sure no reprojection errors are yet included.
          marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
        }
      }
    }
    // add all error terms of the sensor states.
    for (size_t i = 0; i < it->second.sensors.size(); ++i) {
      for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
        for (size_t k = 0; k < it->second.sensors[i][j].size(); ++k) {
          if (i == SensorStates::Camera && k == CameraSensorStates::T_SCi) {
            continue; // we do not remove the extrinsics pose here.
          }
          if (!it->second.sensors[i][j][k].exists) {
            continue;
          }
          if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)
              ->fixed()) {
            continue;  // we never eliminate fixed blocks.
          }
          std::map<uint64_t, States>::iterator checkit = it;
          checkit++;
          // only get rid of it, if it's different
          if(checkit->second.sensors[i][j][k].exists &&
              checkit->second.sensors[i][j][k].id == it->second.sensors[i][j][k].id){
            continue;
          }
          it->second.sensors[i][j][k].exists = false; // remember we removed
          paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
          keepParameterBlocks.push_back(false);
          ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(
              it->second.sensors[i][j][k].id);
          for (size_t r = 0; r < residuals.size(); ++r) {
            std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
                std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                residuals[r].errorInterfacePtr);
            if(!reprojectionError){   // we make sure no reprojection errors are yet included.
              marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
            }
          }
        }
      }
    }
  }
  // marginalize ONLY pose now:
  bool reDoFixation = false;
  for(size_t k = 0; k<removeFrames.size(); ++k){
    std::map<uint64_t, States>::iterator it = statesMap_.find(removeFrames[k]);

    // schedule removal - but always keep the very first frame.
    //if(it != statesMap_.begin()){
    if(true){ /////DEBUG
      it->second.global[GlobalStates::T_WS].exists = false; // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.global[GlobalStates::T_WS].id);
      keepParameterBlocks.push_back(false);
    }

    // add remaing error terms
    ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(
        it->second.global[GlobalStates::T_WS].id);

    for (size_t r = 0; r < residuals.size(); ++r) {
      if(std::dynamic_pointer_cast<ceres::PoseError>(
           residuals[r].errorInterfacePtr)){ // avoids linearising initial pose error
        mapPtr_->removeResidualBlock(residuals[r].residualBlockId);
        reDoFixation = true;
        continue;
      }
      std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
          std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
          residuals[r].errorInterfacePtr);
      if(!reprojectionError){   // we make sure no reprojection errors are yet included.
        marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
      }
    }

    // add remaining error terms of the sensor states.
    size_t i = SensorStates::Camera;
    for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
      size_t k = CameraSensorStates::T_SCi;
      if (!it->second.sensors[i][j][k].exists) {
        continue;
      }
      if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)
          ->fixed()) {
        continue;  // we never eliminate fixed blocks.
      }
      std::map<uint64_t, States>::iterator checkit = it;
      checkit++;
      // only get rid of it, if it's different
      if(checkit->second.sensors[i][j][k].exists &&
          checkit->second.sensors[i][j][k].id == it->second.sensors[i][j][k].id){
        continue;
      }
      it->second.sensors[i][j][k].exists = false; // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
      keepParameterBlocks.push_back(false);
      ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(
          it->second.sensors[i][j][k].id);
      for (size_t r = 0; r < residuals.size(); ++r) {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
            residuals[r].errorInterfacePtr);
        if(!reprojectionError){   // we make sure no reprojection errors are yet included.
          marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
        }
      }
    }

    // now finally we treat all the observations.
    OKVIS_ASSERT_TRUE_DBG(Exception, allLinearizedFrames.size()>0, "bug");
    uint64_t currentKfId = allLinearizedFrames.at(0);

    {
      for(PointMap::iterator pit = landmarksMap_.begin();
          pit != landmarksMap_.end(); ) {
        if (pit->second.residualizeCase == NotInState_NotTrackedNow) {
          MapPoint& mp = pit->second;
          for (std::map<okvis::KeypointIdentifier, uint64_t>::iterator oit =
                   mp.observations.begin();
               oit != mp.observations.end();) {
            if (oit->first.frameId == it->second.id) {
              oit = mp.observations.erase(oit);
            } else {
              ++oit;
            }
          }
          pit++;
          continue;  // Factors of this landmark has not been added to the
                     // graph, so keep it.
        }
        ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(pit->first);

        // first check if we can skip
        bool skipLandmark = true;
        bool hasNewObservations = false;
        bool justDelete = false;
        bool marginalize = true;
        bool errorTermAdded = false;
        std::map<uint64_t,bool> visibleInFrame;
        size_t obsCount = 0;
        for (size_t r = 0; r < residuals.size(); ++r) {
          std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
              std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                  residuals[r].errorInterfacePtr);
          if (reprojectionError) {
            uint64_t poseId = mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
            // since we have implemented the linearisation to account for robustification,
            // we don't kick out bad measurements here any more like
            // if(vectorContains(allLinearizedFrames,poseId)){ ...
            //   if (error.transpose() * error > 6.0) { ... removeObservation ... }
            // }
            if(vectorContains(removeFrames,poseId)){
              skipLandmark = false;
            }
            if(poseId>=currentKfId){
              marginalize = false;
              hasNewObservations = true;
            }
            if(vectorContains(allLinearizedFrames, poseId)){
              visibleInFrame.insert(std::pair<uint64_t,bool>(poseId,true));
              obsCount++;
            }
          }
        }

        if(residuals.size()==0){
          mapPtr_->removeParameterBlock(pit->first);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }

        if(skipLandmark) {
          pit++;
          continue;
        }

        // so, we need to consider it.
        for (size_t r = 0; r < residuals.size(); ++r) {
          std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
              std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                  residuals[r].errorInterfacePtr);
          if (reprojectionError) {
            uint64_t poseId = mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
            if((vectorContains(removeFrames,poseId) && hasNewObservations) ||
                (!vectorContains(allLinearizedFrames,poseId) && marginalize)){
              // ok, let's ignore the observation.
              removeObservation(residuals[r].residualBlockId);
              residuals.erase(residuals.begin() + r);
              r--;
            } else if(marginalize && vectorContains(allLinearizedFrames,poseId)) {
              // TODO: consider only the sensible ones for marginalization
              if(obsCount<2){ //visibleInFrame.size()
                removeObservation(residuals[r].residualBlockId);
                residuals.erase(residuals.begin() + r);
                r--;
              } else {
                // add information to be considered in marginalization later.
                errorTermAdded = true;
                marginalizationErrorPtr_->addResidualBlock(
                    residuals[r].residualBlockId, false);
              }
            }
            // check anything left
            if (residuals.size() == 0) {
              justDelete = true;
              marginalize = false;
            }
          }
        }

        if(justDelete){
          mapPtr_->removeParameterBlock(pit->first);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }
        if(marginalize&&errorTermAdded){
          paremeterBlocksToBeMarginalized.push_back(pit->first);
          keepParameterBlocks.push_back(false);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }

        pit++;
      }
    }

    // update book-keeping and go to the next frame
    //if(it != statesMap_.begin()){ // let's remember that we kept the very first pose
    if(true) { ///// DEBUG
      multiFramePtrMap_.erase(it->second.id);
      statesMap_.erase(it->second.id);
    }
  }

  // now apply the actual marginalization
  if(paremeterBlocksToBeMarginalized.size()>0){
    std::vector< ::ceres::ResidualBlockId> addedPriors;
    marginalizationErrorPtr_->marginalizeOut(paremeterBlocksToBeMarginalized, keepParameterBlocks);
  }

  // update error computation
  if(paremeterBlocksToBeMarginalized.size()>0){
    marginalizationErrorPtr_->updateErrorComputation();
  }

  // add the marginalization term again
  if(marginalizationErrorPtr_->num_residuals()==0){
    marginalizationErrorPtr_.reset();
  }
  if (marginalizationErrorPtr_) {
  std::vector<std::shared_ptr<okvis::ceres::ParameterBlock> > parameterBlockPtrs;
  marginalizationErrorPtr_->getParameterBlockPtrs(parameterBlockPtrs);
  marginalizationResidualId_ = mapPtr_->addResidualBlock(
      marginalizationErrorPtr_, NULL, parameterBlockPtrs);
  OKVIS_ASSERT_TRUE_DBG(Exception, marginalizationResidualId_,
                     "could not add marginalization error");
  if (!marginalizationResidualId_)
    return false;
  }

	if(reDoFixation){
		// finally fix the first pose properly
		//mapPtr_->resetParameterization(statesMap_.begin()->first, ceres::Map::Pose3d);
		okvis::kinematics::Transformation T_WS_0;
		get_T_WS(statesMap_.begin()->first, T_WS_0);
		Eigen::Matrix<double, 6, 6> information;
		pvstd_.toInformation(&information);
		std::shared_ptr<ceres::PoseError > poseError(new ceres::PoseError(T_WS_0, information));
		mapPtr_->addResidualBlock(poseError,NULL,mapPtr_->parameterBlockPtr(statesMap_.begin()->first));
	}

	return true;
}

bool ConsistentEstimator::getStateStd(
    Eigen::Matrix<double, Eigen::Dynamic, 1>* stateStd) const {
  Eigen::MatrixXd covariance;
  bool status = computeCovariance(&covariance);
  *stateStd = covariance.diagonal().cwiseSqrt();
  return status;
}

bool hasMultipleObservationsInOneImage(const MapPoint& mapPoint) {
  uint64_t lastFrameId = 0u;
  size_t duplicates = 0u;
  for (std::map<okvis::KeypointIdentifier, uint64_t>::const_iterator
           obsIt = mapPoint.observations.begin();
       obsIt != mapPoint.observations.end(); ++obsIt) {
    if (obsIt->first.frameId == lastFrameId) {
      ++duplicates;
    }
    lastFrameId = obsIt->first.frameId;
  }
  return duplicates > 0u;
}

}  // namespace okvis
