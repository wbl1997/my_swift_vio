#!/usr/bin/env bash
# change the camera and IMU parameters inside configuration file.

sed -i "/camera_rate:/c\    camera_rate: $CAMERA_RATE" $MSCKF_TEMPLATE
sed -i "/distortion_type:/c\        distortion_type: $DISTORTION_TYPE," $MSCKF_TEMPLATE
sed -i "/distortion_coefficients:/c\        distortion_coefficients: $DISTORTION_COEFFS," $MSCKF_TEMPLATE
sed -i "/image_dimension:/c\        image_dimension: [${WIDTH[$j]}, ${HEIGHT[$j]}]," $MSCKF_TEMPLATE
sed -i "/  focal_length:/c\        focal_length: [${FXY[$j]}, ${FXY[$j]}]," $MSCKF_TEMPLATE
sed -i "/  principal_point:/c\        principal_point: [${CX[$j]}, ${CY[$j]}]," $MSCKF_TEMPLATE
sed -i "/image_readout_time:/c\        image_readout_time: ${SHUTTER_TR[$j]}}" $MSCKF_TEMPLATE

sed -i "/sigma_g_c:/c\    sigma_g_c: ${gwexpr[$j]}" $MSCKF_TEMPLATE
sed -i "/sigma_a_c:/c\    sigma_a_c: ${awexpr[$j]}" $MSCKF_TEMPLATE
sed -i "/sigma_gw_c:/c\    sigma_gw_c: ${gbwexpr[$j]}" $MSCKF_TEMPLATE
sed -i "/sigma_aw_c:/c\    sigma_aw_c: ${abwexpr[$j]}" $MSCKF_TEMPLATE

sed -i "/sigma_focal_length/c\    sigma_focal_length: $sigma_focal_length" $MSCKF_TEMPLATE
sed -i "/sigma_principal_point/c\    sigma_principal_point: $sigma_principal_point" $MSCKF_TEMPLATE
sed -i "/sigma_distortion/c\    sigma_distortion: $sigma_distortion" $MSCKF_TEMPLATE

sed -i "/minTrackLength/c\    minTrackLength: $min_track_len" $MSCKF_TEMPLATE
sed -i "/triangulationMaxDepth/c\    triangulationMaxDepth: $max_depth" $MSCKF_TEMPLATE
