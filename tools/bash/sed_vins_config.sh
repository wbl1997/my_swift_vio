#!/usr/bin/env bash
sed -i "/output_path/c\output_path: $VINS_OUTPUT_DIR" $VINS_TEMPLATE
sed -i "/image_width/c\image_width: ${WIDTH[$j]}" $VINS_TEMPLATE
sed -i "/image_height/c\image_height: ${HEIGHT[$j]}" $VINS_TEMPLATE

sed -i "/fx:/c\   fx: ${FXY[$j]}" $VINS_TEMPLATE
sed -i "/fy:/c\   fy: ${FXY[$j]}" $VINS_TEMPLATE
sed -i "/cx:/c\   cx: ${CX[$j]}" $VINS_TEMPLATE
sed -i "/cy:/c\   cy: ${CY[$j]}" $VINS_TEMPLATE

sed -i "/pose_graph_save_path/c\pose_graph_save_path: $VINS_OUTPUT_DIR" $VINS_TEMPLATE
sed -i "/rolling_shutter_tr:/c\rolling_shutter_tr: ${SHUTTER_TR[$j]}" $VINS_TEMPLATE

sed -i "/acc_n:/c\acc_n: ${awexpr[$j]}" $VINS_TEMPLATE
sed -i "/gyr_n:/c\gyr_n: ${gwexpr[$j]}" $VINS_TEMPLATE
sed -i "/acc_w:/c\acc_w: ${abwexpr[$j]}" $VINS_TEMPLATE
sed -i "/gyr_w:/c\gyr_w: ${gbwexpr[$j]}" $VINS_TEMPLATE
sed -i "/loop_closure:/c\loop_closure: 1" $VINS_TEMPLATE
sed -i "/freq:/c\freq: $OPTIMIZATION_FREQUENCY" $VINS_TEMPLATE
