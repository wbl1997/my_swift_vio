Sensor calibration library

An implementation of Qiu et al. 2020 T-RO for real time time offset and rotation calibration 
of a target generic sensor relative to a central IMU.

## Install dependencies

### gtest
The program will download and compile it automatically.

### glog
```
sudo apt-get install libgoogle-glog-dev
```

### Eigen
```
sudo apt-get install libeigen3-dev
```

### ceres solver
The ceres auto diff header is used for differentiation.

```
cd sensor_calib
git clone --recursive https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver
git checkout a60136b7aa2c1ab97558a74b271eee16ca1365c4
INSTALL_DIR=$PWD/install
mkdir build
mkdir install
cd build
cmake .. -DBUILD_EXAMPLES:BOOL=OFF -DBUILD_TESTING:BOOL=OFF -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR
make -j4
make install
```

## Build
```
cd sensor_calib
mkdir build
cd build
cmake ..
make

```

## Test
```
$ cd sensor_calib/build
$ ./sensor_calib-test/sensor_calib-test
```


