# run vins mono on TUM VI data to see the effect of self-calibration.
datadir="/data/tumvi/calibrated/512_16/room"
datanames=("room1" "room2" "room3" "room4" "room5" "room6")
vinsmono_ws="/baselines/vinsmono_ws"
vins_rviz="$vinsmono_ws/src/VINS-Mono/config/vins_rviz_config.rviz"

vinsmono_process() {
cd $vinsmono_ws
source devel/setup.bash
cmd="roslaunch vins_estimator tum.launch \
  config_path:=$tum_config \
  bag_file:=$datadir/dataset-"$dataname"_512_16.bag \
  bag_start:=0 \
  bag_duration:=10000 \
  rviz_file:=$vins_rviz"
echo "Working on $dataname with vins mono"
echo "$cmd"
$cmd
}

tumvi_calibrated_vinsmono() {
for dataname in ${datanames[@]}; do
tum_config=$output_dir/$dataname/tum.yaml
vins_output_dir=$output_dir/$dataname
mkdir -p $vins_output_dir

cp $vinsmono_ws/src/VINS-Mono/config/tum/tum_config.yaml $tum_config

sed -i "/acc_n:/c\acc_n: $1" $tum_config
sed -i "/gyr_n:/c\gyr_n: $2" $tum_config
sed -i "/acc_w:/c\acc_w: $3" $tum_config
sed -i "/gyr_w:/c\gyr_w: $4" $tum_config
sed -i "/loop_closure:/c\loop_closure: 0" $tum_config

sed -i "/output_path/c\output_path: $vins_output_dir" $tum_config
sed -i "/estimate_extrinsic:/c\estimate_extrinsic: 0" $tum_config

vinsmono_process

done
}

acc_n=0.04
gyr_n=0.004
acc_w=0.0004
gyr_w=2.0e-5
tumvi_calibrated_vinsmono $acc_n $gyr_n $acc_w $gyr_w


