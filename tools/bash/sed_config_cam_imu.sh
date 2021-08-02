#!/usr/bin/env bash
# change the camera and IMU parameters inside configuration file.

sed -i "/camera_rate:/c\    camera_rate: $CAMERA_RATE" $SWIFT_VIO_TEMPLATE
sed -i "/distortion_type:/c\        distortion_type: $DISTORTION_TYPE," $SWIFT_VIO_TEMPLATE
sed -i "/distortion_coefficients:/c\        distortion_coefficients: $DISTORTION_COEFFS," $SWIFT_VIO_TEMPLATE
sed -i "/image_dimension:/c\        image_dimension: [${WIDTH[$j]}, ${HEIGHT[$j]}]," $SWIFT_VIO_TEMPLATE
sed -i "/  focal_length:/c\        focal_length: [${FXY[$j]}, ${FXY[$j]}]," $SWIFT_VIO_TEMPLATE
sed -i "/  principal_point:/c\        principal_point: [${CX[$j]}, ${CY[$j]}]," $SWIFT_VIO_TEMPLATE
sed -i "/image_readout_time:/c\        image_readout_time: ${SHUTTER_TR[$j]}}" $SWIFT_VIO_TEMPLATE

sed -i "/sigma_g_c:/c\    sigma_g_c: ${gwexpr[$j]}" $SWIFT_VIO_TEMPLATE
sed -i "/sigma_a_c:/c\    sigma_a_c: ${awexpr[$j]}" $SWIFT_VIO_TEMPLATE
sed -i "/sigma_gw_c:/c\    sigma_gw_c: ${gbwexpr[$j]}" $SWIFT_VIO_TEMPLATE
sed -i "/sigma_aw_c:/c\    sigma_aw_c: ${abwexpr[$j]}" $SWIFT_VIO_TEMPLATE

sed -i "/sigma_focal_length/c\    sigma_focal_length: $sigma_focal_length" $SWIFT_VIO_TEMPLATE
sed -i "/sigma_principal_point/c\    sigma_principal_point: $sigma_principal_point" $SWIFT_VIO_TEMPLATE
sed -i "/sigma_distortion/c\    sigma_distortion: $sigma_distortion" $SWIFT_VIO_TEMPLATE

sed -i "/minTrackLength/c\    minTrackLength: $min_track_len" $SWIFT_VIO_TEMPLATE
sed -i "/triangulationMaxDepth/c\    triangulationMaxDepth: $max_depth" $SWIFT_VIO_TEMPLATE
