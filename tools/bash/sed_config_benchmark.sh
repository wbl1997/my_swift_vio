#!/usr/bin/env bash
# change configuration file for benchmarking. Notice that 
# the camera and IMU parameters are not touched.

sed -i "/numImuFrames:/c\numImuFrames: $NUM_IMU_FRAMES" $SWIFT_VIO_TEMPLATE
sed -i "/down_scale:/c\        down_scale: 1," $SWIFT_VIO_TEMPLATE
sed -i "/projection_opt_mode:/c\        projection_opt_mode: FIXED," $SWIFT_VIO_TEMPLATE
sed -i "/extrinsic_opt_mode:/c\        extrinsic_opt_mode: FIXED," $SWIFT_VIO_TEMPLATE

sed -i "/algorithm/c\    algorithm: $ESTIMATOR_ALGORITHM" $SWIFT_VIO_TEMPLATE
sed -i "/useEpipolarConstraint/c\    useEpipolarConstraint: $useEpipolarConstraint" $SWIFT_VIO_TEMPLATE
sed -i "/cameraObservationModelId/c\    cameraObservationModelId: $cameraObservationModelId" $SWIFT_VIO_TEMPLATE
sed -i "/landmarkModelId/c\    landmarkModelId: $landmarkModelId" $SWIFT_VIO_TEMPLATE
