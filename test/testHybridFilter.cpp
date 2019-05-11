
#include <gtest/gtest.h>
#ifdef USE_MSCKF2
#include <okvis/msckf2.hpp>
#else
#include <okvis/HybridFilter.hpp>
#endif
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>

#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/assert_macros.hpp>

#include <vio/rand_sampler.h>
#include <okvis/ceres/ShapeMatrixParamBlock.hpp>
#include <okvis/ceres/CameraDistortionParamBlock.hpp>
#include <okvis/ceres/CameraIntrinsicParamBlock.hpp>
#include <okvis/ceres/CameraTimeParamBlock.hpp>
#include "okvis/ImuSimulator.h"

#include "okvis/IMUOdometry.h"
#include "okvis/timing/Timer.hpp"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/density.hpp>
#include <boost/accumulators/statistics/stats.hpp>

using namespace boost;
using namespace boost::accumulators;

DECLARE_bool(use_mahalanobis);

typedef accumulator_set<double, features<tag::density> > MyAccumulator;
typedef iterator_range<std::vector<std::pair<double, double> >::iterator > histogram_type;

// Get a subset of the recorded IMU measurements from source imuMeasurements_.
okvis::ImuMeasurementDeque getImuMeasurments(
        const okvis::ImuMeasurementDeque & imuMeasurements_,
        const okvis::Time lastKFTime, const okvis::Time currentKFTime) {
    const okvis::Duration temporal_imu_data_overlap(0.01);
    okvis::Time imuDataEndTime = currentKFTime + temporal_imu_data_overlap;
    okvis::Time imuDataBeginTime = lastKFTime - temporal_imu_data_overlap;
    // sanity checks:
    // if end time is smaller than begin time, return empty queue.
    // if begin time is larger than newest imu time, return empty queue.
    if (imuDataEndTime < imuDataBeginTime
            || imuDataBeginTime > imuMeasurements_.back().timeStamp)
        return okvis::ImuMeasurementDeque();

    // get iterator to imu data before previous frame
    okvis::ImuMeasurementDeque::const_iterator first_imu_package = imuMeasurements_
            .begin();
    okvis::ImuMeasurementDeque::const_iterator last_imu_package =
            imuMeasurements_.end();
    // TODO go backwards through queue. Is probably faster.
    for (auto iter = imuMeasurements_.begin(); iter != imuMeasurements_.end();
         ++iter) {
        // move first_imu_package iterator back until iter->timeStamp is higher than requested begintime
        if (iter->timeStamp <= imuDataBeginTime)
            first_imu_package = iter;

        // set last_imu_package iterator as soon as we hit first timeStamp higher than requested endtime & break
        if (iter->timeStamp >= imuDataEndTime) {
            last_imu_package = iter;
            // since we want to include this last imu measurement in returned Deque we
            // increase last_imu_package iterator once.
            ++last_imu_package;
            break;
        }
    }

    // create copy of imu buffer
    return okvis::ImuMeasurementDeque(first_imu_package, last_imu_package);
}

void testHybridFilterCircle(){
    //srand((unsigned int) time(0)); // disabled: make unit tests deterministic...
    TestSetting cases[]= {TestSetting(false, false, false, false), // no noise, only imu
                          TestSetting(true, false, false, false), // only noisy imu
                          TestSetting(true, false, false, true), // noisy imu, and use true image measurements
                          TestSetting(true, true, true, true)}; // noisy data, vins integration
    // different cases
    for (size_t c = 3; c < sizeof(cases)/sizeof(cases[0]); ++c) {

        OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error);

        const double DURATION = 30.0;  // 10 seconds motion
        const double IMU_RATE = 100.0;  // Hz
        const double DT = 1.0 / IMU_RATE;  // time increments

        // set the imu parameters
        okvis::ImuParameters imuParameters;
        imuParameters.a0.setZero();
        imuParameters.g = 9.81;
        imuParameters.a_max = 1000.0;
        imuParameters.g_max = 1000.0;
        imuParameters.rate = 100;
        imuParameters.sigma_g_c = 6.0e-4;
        imuParameters.sigma_a_c = 2.0e-3;
        imuParameters.sigma_gw_c = 3.0e-6;
        imuParameters.sigma_aw_c = 2.0e-5;
        imuParameters.tau = 3600.0;

        imuParameters.sigma_bg = 1e-2;  ///< Initial gyroscope bias.
        imuParameters.sigma_ba = 5e-2;  ///< Initial accelerometer bias

        imuParameters.sigma_TGElement =1e-5; /// std for every element in shape matrix T_g
        imuParameters.sigma_TSElement =1e-5;
        imuParameters.sigma_TAElement =1e-5;

        std::cout << "case " << c << " "<< cases[c].print() <<std::endl;

        // let's generate a simple motion: constant angular rate and linear acceleration
        // the sensor rig is moving in a room with four walls of feature points
        // the world frame sits on the cube geometry center of the room
        // imu frame has z point up, x axis goes away from the world frame origin
        // camera frame has z point forward along the motion, x axis goes away from the world frame origin
        // the world frame has the same orientation as the imu frame at the starting point
        double angular_rate = 0.3; //rad/sec
        const double radius = 1.5;

        okvis::ImuMeasurementDeque imuMeasurements;
        okvis::ImuSensorReadings nominalImuSensorReadings(
                    Eigen::Vector3d(0,0, angular_rate),
                    Eigen::Vector3d( - radius*angular_rate*angular_rate, 0, imuParameters.g));
        okvis::Time t0 = okvis::Time::now();

        if(cases[c].addImuNoise){
            for (int i = -2; i <= DURATION * IMU_RATE + 2; ++i) {

                Eigen::Vector3d gyr = nominalImuSensorReadings.gyroscopes
                        + vio::Sample::gaussian(imuParameters.sigma_g_c / sqrt(DT), 3);
                Eigen::Vector3d acc = nominalImuSensorReadings.accelerometers
                        + vio::Sample::gaussian(imuParameters.sigma_a_c / sqrt(DT), 3);
                imuMeasurements.push_back(
                            okvis::ImuMeasurement(t0 + okvis::Duration(DT * i),
                                                  okvis::ImuSensorReadings(gyr, acc)));
            }
        }
        else
        {
            for (int i = -2; i <= DURATION * IMU_RATE + 2; ++i) {

                Eigen::Vector3d gyr = nominalImuSensorReadings.gyroscopes;
                Eigen::Vector3d acc = nominalImuSensorReadings.accelerometers;
                imuMeasurements.push_back(
                            okvis::ImuMeasurement(t0 + okvis::Duration(DT * i),
                                                  okvis::ImuSensorReadings(gyr, acc)));
            }
        }

        // create the map
        std::shared_ptr<okvis::ceres::Map> mapPtr(new okvis::ceres::Map);

        // camera extrinsics:
        Eigen::Matrix<double, 4,4 > matT_SC0;
        matT_SC0<< 1, 0, 0, 0,
                0, 0, 1, 0,
                0,-1, 0, 0,
                0, 0, 0, 1;
        std::shared_ptr<const okvis::kinematics::Transformation> T_SC_0(
                    new okvis::kinematics::Transformation(matT_SC0));

        // some parameters on how to do the online estimation:
        okvis::ExtrinsicsEstimationParameters extrinsicsEstimationParameters;
        extrinsicsEstimationParameters.sigma_absolute_translation = 2e-3;
        extrinsicsEstimationParameters.sigma_absolute_orientation = 0; // not used
        extrinsicsEstimationParameters.sigma_c_relative_translation = 0; //not used
        extrinsicsEstimationParameters.sigma_c_relative_orientation = 0; //not used in msckf2

        extrinsicsEstimationParameters.sigma_focal_length = 0.01;
        extrinsicsEstimationParameters.sigma_principal_point =0.01;
        extrinsicsEstimationParameters.sigma_distortion<< 1E-3, 1E-4, 1E-4, 1E-4, 1E-5; ///k1, k2, p1, p2, [k3]
        extrinsicsEstimationParameters.sigma_td = 1E-4;
        extrinsicsEstimationParameters.sigma_tr = 1E-4;

        // set up camera with intrinsics

        std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry0(
                    new okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> (640, 480, 350, 350, 322, 238,
                                                                                                   okvis::cameras::RadialTangentialDistortion(0, 0, 0, 0)));

        // create a 1-camera system:
        std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem(
                    new okvis::cameras::NCameraSystem);
        cameraSystem->addCamera(T_SC_0, cameraGeometry0,
                                okvis::cameras::NCameraSystem::DistortionType::RadialTangential);

        // create an Estimator
#ifdef USE_MSCKF2
        okvis::MSCKF2 estimator(mapPtr);
#else
        okvis::HybridFilter estimator(mapPtr);
#endif
        // create landmark grid
        std::vector<Eigen::Vector4d,
                Eigen::aligned_allocator<Eigen::Vector4d> > homogeneousPoints;
        std::vector<uint64_t> lmIds;
        // four walls
        double x(10), y(10), z(5);
        // wall at x == 10;
        for (y = -10.0; y <= 10.0; y += 0.5) {
            for (z = - 5; z <= 5.0; z += 0.5) {
                homogeneousPoints.push_back(Eigen::Vector4d(x + vio::gauss_rand(0, 0.1),
                                                            y + vio::gauss_rand(0, 0.1),
                                                            z + vio::gauss_rand(0, 0.1), 1));
                lmIds.push_back(okvis::IdProvider::instance().newId());

            }
        }
        //wall at y == 10
        for (y = -10.0; y <= 10.0; y += 0.5) {
            for (z = - 5; z <= 5.0; z += 0.5) {
                homogeneousPoints.push_back(Eigen::Vector4d(y + vio::gauss_rand(0, 0.1),
                                                            x + vio::gauss_rand(0, 0.1),
                                                            z + vio::gauss_rand(0, 0.1), 1));
                lmIds.push_back(okvis::IdProvider::instance().newId());

            }
        }

        x= -10;
        //wall at x == -10
        for (y = -10.0; y <= 10.0; y += 0.5) {
            for (z = - 5; z <= 5.0; z += 0.5) {
                homogeneousPoints.push_back(Eigen::Vector4d(x + vio::gauss_rand(0, 0.1),
                                                            y + vio::gauss_rand(0, 0.1),
                                                            z + vio::gauss_rand(0, 0.1), 1));
                lmIds.push_back(okvis::IdProvider::instance().newId());

            }
        }
        // wall at y == -10
        for (y = -10.0; y <= 10.0; y += 0.5) {
            for (z = - 5; z <= 5.0; z += 0.5) {
                homogeneousPoints.push_back(Eigen::Vector4d(y + vio::gauss_rand(0, 0.1),
                                                            x + vio::gauss_rand(0, 0.1),
                                                            z + vio::gauss_rand(0, 0.1), 1));
                lmIds.push_back(okvis::IdProvider::instance().newId());

            }
        }

        // add sensors
        estimator.addCamera(extrinsicsEstimationParameters);
        estimator.addImu(imuParameters);

        const size_t K = 15*DURATION; // total keyframes
        uint64_t id = -1;
        std::vector<uint64_t> multiFrameIds;
        okvis::kinematics::Transformation T_WS_est;
        okvis::SpeedAndBias speedAndBias_est;
        for (size_t k = 0; k < K + 1; ++k) {
            // calculate the ground truth motion
            double epoch = double(k) * DURATION / double(K);
            double theta = angular_rate*epoch;
            double ct = std::cos(theta), st= std::sin(theta);
            Eigen::Vector3d trans(radius* ct, radius* st, 0);
            Eigen::Matrix3d rot;
            rot<< ct, st, 0, -st, ct, 0, 0, 0,1;
            okvis::kinematics::Transformation T_WS( trans, Eigen::Quaterniond(rot.transpose()));

            // assemble a multi-frame
            std::shared_ptr<okvis::MultiFrame> mf(new okvis::MultiFrame);
            mf->setId(okvis::IdProvider::instance().newId());
            mf->setTimestamp(t0 + okvis::Duration(epoch));

            id = mf->id();
            multiFrameIds.push_back(id);

            // add frames
            mf->resetCameraSystemAndFrames(*cameraSystem);

            // add it in the window to create a new time instance

            okvis::Time lastKFTime(t0);

            okvis::Time currentKFTime = t0 + okvis::Duration(epoch);
            if(k!=0)
                lastKFTime = estimator.statesMap_.rbegin()->second.timestamp;
            okvis::ImuMeasurementDeque imuSegment = getImuMeasurments(
                        imuMeasurements, lastKFTime, currentKFTime);
            if(k==0){
                Eigen::Vector3d p_WS= Eigen::Vector3d(radius, 0, 0);
                Eigen::Vector3d v_WS= Eigen::Vector3d(0, angular_rate*radius, 0);
                Eigen::Matrix3d R_WS;
                R_WS<< 1, 0, 0, 0, 1, 0, 0, 0, 1; // the RungeKutta method assumes that the z direction of the world frame is negative gravity direction
                Eigen::Quaterniond q_WS = Eigen::Quaterniond(R_WS);
                if(cases[c].addPriorNoise){
                    // Eigen::Vector3d::Random() return -1, 1 random values
                    p_WS += 0.001*vio::Sample::gaussian(1,3);
                    v_WS += 0.001*vio::Sample::gaussian(1,3);
                    q_WS.normalize();
                }

                okvis::InitialPVandStd pvstd;
                pvstd.p_WS = p_WS;
                pvstd.q_WS = Eigen::Quaterniond(q_WS);
                pvstd.v_WS = v_WS;
                pvstd.std_p_WS = Eigen::Vector3d(1e-2, 1e-2, 1e-2);
                pvstd.std_v_WS = Eigen::Vector3d(1e-1, 1e-1, 1e-1);
                pvstd.std_q_WS = Eigen::Vector3d(5e-2, 5e-2, 5e-2);
                estimator.resetInitialPVandStd(pvstd, true);
                estimator.addStates(mf, imuSegment, true);
            }
            else
                estimator.addStates(mf, imuSegment, true);
            std::cout << "Frame " << k << " successfully added." << std::endl;

            // now let's add also landmark observations
            std::vector<cv::KeyPoint> keypoints;
            keypoints.reserve(160);
            std::vector<size_t> kpIds;
            kpIds.reserve(160);

            const size_t camId = 0;
            if(cases[c].useImageObservs){
                for (size_t j = 0; j < homogeneousPoints.size(); ++j) {

                    Eigen::Vector2d projection;
                    Eigen::Vector4d point_C = (T_WS*(*(mf->T_SC( camId )))).inverse() * homogeneousPoints[j];
                    okvis::cameras::CameraBase::ProjectionStatus status = mf
                            ->geometryAs<okvis::cameras::PinholeCamera<
                            okvis::cameras::RadialTangentialDistortion> >(camId)->projectHomogeneous(
                                point_C, &projection);
                    if (status == okvis::cameras::CameraBase::ProjectionStatus::Successful) {

                        Eigen::Vector2d measurement(projection);
                        if(cases[c].addImageNoise)
                            measurement += vio::Sample::gaussian(1,2);

                        keypoints.push_back(
                                    cv::KeyPoint(measurement[0], measurement[1], 8.0));
                        kpIds.push_back(j);
                    }
                }
                mf->resetKeypoints(camId, keypoints);
                for(size_t jack = 0; jack < kpIds.size(); ++jack){
                    if(!estimator.isLandmarkAdded(lmIds[kpIds[jack]]))
                        estimator.addLandmark(lmIds[ kpIds[jack] ], homogeneousPoints[kpIds[jack]]);

                    estimator.addObservation< okvis::cameras::PinholeCamera<
                            okvis::cameras::RadialTangentialDistortion> >(
                                lmIds[ kpIds[jack] ], id, camId, jack);

                }
            }
            // run the optimization
            estimator.optimize(false);

            estimator.applyMarginalizationStrategy();
        }

        std::cout << okvis::timing::Timing::print();
        //generate ground truth for the last keyframe pose
        double epoch = DURATION;
        double theta = angular_rate*epoch;
        double ct = std::cos(theta), st= std::sin(theta);
        Eigen::Vector3d trans(radius* ct, radius* st, 0);
        Eigen::Matrix3d rot;
        rot<< ct, st, 0, -st, ct, 0, 0, 0,1;
        okvis::kinematics::Transformation T_WS( trans, Eigen::Quaterniond(rot.transpose()));
        std::cout << "id and correct T_WS: " << std::endl <<
                     id<< " "<< T_WS.coeffs().transpose() << std::endl;

        okvis::SpeedAndBias speedAndBias;
        speedAndBias.setZero();
        speedAndBias.head<3>() = Eigen::Vector3d( - radius* st* angular_rate, radius*ct* angular_rate ,0);
        std::cout << "correct speed "<<std::endl << speedAndBias.transpose() << std::endl;

        // get the estimates
        estimator.get_T_WS(multiFrameIds.back(), T_WS_est);
        std::cout<< "id and T_WS estimated "<<std::endl;
        std::cout << multiFrameIds.back() <<" "<< T_WS_est.coeffs().transpose()<<std::endl;

        estimator.getSpeedAndBias(multiFrameIds.back(), 0, speedAndBias_est);
        std::cout<< "speed and bias estimated "<<std::endl;
        std::cout << speedAndBias_est.transpose()<<std::endl;

        Eigen::VectorXd intrinsics;
        cameraGeometry0->getIntrinsics(intrinsics);

        std::cout <<"corrent radial tangential distortion "<<std::endl << intrinsics.transpose() <<std::endl;
        const int nDistortionCoeffDim = okvis::cameras::RadialTangentialDistortion::NumDistortionIntrinsics;
        Eigen::VectorXd distIntrinsic = intrinsics.tail<nDistortionCoeffDim>();
        Eigen::Matrix<double,nDistortionCoeffDim, 1> cameraDistortion;
        estimator.getSensorStateEstimateAs<okvis::ceres::CameraDistortionParamBlock>(
                    multiFrameIds.back(), 0,
            #ifdef USE_MSCKF2
                    okvis::MSCKF2::SensorStates::Camera,
                    okvis::MSCKF2::CameraSensorStates::Distortion,
            #else
                    okvis::HybridFilter::SensorStates::Camera,
                    okvis::HybridFilter::CameraSensorStates::Distortion,
            #endif
                    cameraDistortion);

        std::cout<<"distortion deviation "<< (cameraDistortion - distIntrinsic).transpose()<<std::endl;

        estimator.get_T_WS(multiFrameIds.back(), T_WS_est);
        estimator.getSpeedAndBias(multiFrameIds.back(), 0, speedAndBias_est);
        OKVIS_ASSERT_LT(Exception, (speedAndBias_est - speedAndBias).norm(),
                        0.04, "speed and biases not close enough");
        OKVIS_ASSERT_LT(Exception, 2*(T_WS.q()*T_WS_est.q().inverse()).vec().norm(), 8e-2,
                        "quaternions not close enough");
        OKVIS_ASSERT_LT(Exception, (T_WS.r() - T_WS_est.r()).norm(), 1e-1,
                        "translation not close enough");


    }
}


// TODO: curiously, MSCKF2 or IEKF often diverges after 300 seconds.
// Note the std for noises used in covariance propagation should be slightly larger than the std used in sampling noises,
// becuase the process model involves many approximations other than these noise terms.
void testHybridFilterSinusoid(const size_t runs=100u) {
    FLAGS_use_mahalanobis = false; // set USE_MAHALANOBIS false in simulation if no outliers are added
    const double DURATION = 300.0;  // length of motion in seconds
    const double IMU_RATE = 100.0;  // Hz
    const double DT = 1.0 / IMU_RATE;  // time increments
    const double maxTrackLength = 60; // maximum length of a feature track
    double imageNoiseMag = 1.0; // pixel unit
    double imuNoiseFactor= 2.0; // divide the accelerometer and gyro scope noise root PSD by this factor in generating their noise to account for linearization uncertainty effect


    std::vector<std::pair<okvis::Time, Eigen::Vector3d> > nees, neesSum; //definition of NEES in Huang et al. 2007 Analysis and Improvement of the consistency of EKF-based SLAM
    // each entry, timestamp, nees in position, orientation, and pose, the expected NEES is 6 for pose error, see Li ijrr high precision

    std::vector<std::pair<okvis::Time, Eigen::Matrix<double, 55, 1> > > rmse, rmseSum;
    //each entry state timestamp, rmse in xyz, \alpha, v_WS, bg, ba, Tg, Ts, Ta, p_CB, fx, fy, cx, cy, k1, k2, p1, p2, td, tr

    std::string outputPath = "/home/jhuai/Desktop/temp";
    std::string neesFile =  outputPath+"/sinusoidNEES.txt";
    std::string rmseFile =  outputPath+"/sinusoidRMSE.txt";

    std::string truthFile = outputPath+"/sinusoidTruth.txt";
    std::ofstream truthStream;

    //create an accumulator for number of features distribution
    MyAccumulator myAccumulator( tag::density::num_bins = 20, tag::density::cache_size = 40);
    okvis::timing::Timer filterTimer("msckf2 timer",true);
    bool bVerbose= false; //only output the ground truth and data for the first successful trial
    int successRuns =0;
    for(size_t run=0; run<runs; ++run){    
        if(successRuns==0)
            bVerbose = true;
        else
            bVerbose= false;

        filterTimer.start();
        srand((unsigned int) time(0)); // if commented out: make unit tests deterministic...
        TestSetting cases[]= {TestSetting(true, true, true, true)}; // noisy data, vins integration
        OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error);
        size_t c=0;
        std::cout << "Run "<<run<<" "<< cases[c].print() <<std::endl;

        std::string pointFile = outputPath+"/sinusoidPoints.txt";
        std::ofstream pointStream;
        std::string imuSampleFile = outputPath+"/sinusoidInertial.txt";
        std::ofstream inertialStream;
        if(bVerbose){

            truthStream.open(truthFile, std::ofstream::out);
            truthStream << "%state timestamp, frameIdInSource, T_WS(xyz, xyzw), v_WS, bg, ba, Tg, Ts, Ta, "
                           "p_CB, fx, fy, cx, cy, k1, k2, p1, p2, td, tr"<< std::endl;
            pointStream.open(pointFile, std::ofstream::out);
            pointStream << "%id, x, y, z in the world frame "<<std::endl;
            inertialStream.open(imuSampleFile, std::ofstream::out);
            inertialStream << "% timestamp, gx, gy, gz[rad/sec], acc x, acc y, acc z[m/s^2], and noisy gxyz, acc xyz"<<std::endl;
        }

        std::stringstream ss;
        ss << run;
        std::string outputFile = outputPath+"/sinusoidMSCKF2_"+ss.str() +".txt";
        std::ofstream mDebug;  // record state history of a trial
        if(!mDebug.is_open())
        {
            mDebug.open (outputFile, std::ofstream::out);
            mDebug << "%state timestamp, frameIdInSource, T_WS(xyz, xyzw), v_WS, bg, ba, Tg, Ts, Ta, "
                      "p_CB, fx, fy, cx, cy, k1, k2, p1, p2, td, tr, and their stds"<< std::endl;
        }
        const double pCB_ba_Ta_std[]={2e-2, 2e-2, 5e-3};
//        const double pCB_ba_Ta_std[]={1e-2, 1e-2, 1e-4};


        okvis::ExtrinsicsEstimationParameters extrinsicsEstimationParameters;
        extrinsicsEstimationParameters.sigma_absolute_translation = pCB_ba_Ta_std[0];
        extrinsicsEstimationParameters.sigma_absolute_orientation = 0;
        extrinsicsEstimationParameters.sigma_c_relative_translation = 0;
        extrinsicsEstimationParameters.sigma_c_relative_orientation = 0;

        extrinsicsEstimationParameters.sigma_focal_length = 5;
        extrinsicsEstimationParameters.sigma_principal_point = 5;
        extrinsicsEstimationParameters.sigma_distortion<< 5E-2, 1E-2, 1E-3, 1E-3, 1E-3; ///k1, k2, p1, p2, [k3]
        extrinsicsEstimationParameters.sigma_td = 5E-3;
        extrinsicsEstimationParameters.sigma_tr = 5E-3;

        // set the imu parameters
        okvis::ImuParameters imuParameters;

        imuParameters.g = 9.81;
        imuParameters.a_max = 1000.0;
        imuParameters.g_max = 1000.0;
        imuParameters.rate = IMU_RATE;

//        imuParameters.sigma_g_c = 6e-4;
//        imuParameters.sigma_a_c = 3e-3;
//        imuParameters.sigma_gw_c = 4e-6;
//        imuParameters.sigma_aw_c = 2e-5;

        imuParameters.sigma_g_c = 1.2e-3;
        imuParameters.sigma_a_c = 8e-3;
        imuParameters.sigma_gw_c = 2e-5;
        imuParameters.sigma_aw_c = 5.5e-5;
        imuParameters.tau = 3600.0;

        imuParameters.sigma_bg = 5e-3;  ///< Initial gyroscope bias.
        imuParameters.sigma_ba = pCB_ba_Ta_std[1];  ///< Initial accelerometer bias

        imuParameters.sigma_TGElement =5e-3; /// std for every element in shape matrix T_g
        imuParameters.sigma_TSElement =1e-3;
        imuParameters.sigma_TAElement = pCB_ba_Ta_std[2];

        okvis::InitialPVandStd pvstd;
        pvstd.std_p_WS = Eigen::Vector3d(1e-8, 1e-8, 1e-8);
        pvstd.std_q_WS = Eigen::Vector3d(1e-8, 1e-8, 1e-8);
        pvstd.std_v_WS = Eigen::Vector3d(5e-2, 5e-2, 5e-2);

        Eigen::Matrix<double,9,1> eye;
        eye<< 1,0,0,0,1,0,0,0,1;

        if(cases[c].addPriorNoise){
            imuParameters.a0[0] = vio::gauss_rand(0,imuParameters.sigma_ba);
            imuParameters.a0[1] = vio::gauss_rand(0,imuParameters.sigma_ba);
            imuParameters.a0[2] = vio::gauss_rand(0,imuParameters.sigma_ba);
            imuParameters.g0[0] = vio::gauss_rand(0,imuParameters.sigma_bg);
            imuParameters.g0[1] = vio::gauss_rand(0,imuParameters.sigma_bg);
            imuParameters.g0[2] = vio::gauss_rand(0,imuParameters.sigma_bg);

            imuParameters.Tg0 = eye+ vio::Sample::gaussian(imuParameters.sigma_TGElement, 9);
            imuParameters.Ts0 = vio::Sample::gaussian(imuParameters.sigma_TSElement, 9);
            imuParameters.Ta0 = eye+vio::Sample::gaussian(imuParameters.sigma_TAElement, 9);
            imuParameters.td0 = vio::gauss_rand(0, extrinsicsEstimationParameters.sigma_td);
        }
        else
        {
            imuParameters.a0.setZero();
            imuParameters.g0.setZero();

            imuParameters.Tg0 = eye;
            imuParameters.Ts0.setZero();
            imuParameters.Ta0 = eye;
            imuParameters.td0 = 0;
        }
        // imu frame has z point up, x axis goes away from the world frame origin
        // camera frame has z point forward along the motion, x axis goes away from the world frame origin
        // the world frame has the same orientation as the imu frame at the starting point

        std::vector<okvis::kinematics::Transformation > qs2w;
        std::vector<okvis::Time> times;
        const okvis::Time tStart(20);
        const okvis::Time tEnd(20+DURATION);

        CircularSinusoidalTrajectory3 cst(IMU_RATE, Eigen::Vector3d(0,0, -imuParameters.g));
        cst.getTruePoses(tStart, tEnd, qs2w);

        cst.getSampleTimes(tStart, tEnd, times);
        OKVIS_ASSERT_EQ(Exception, qs2w.size(), times.size(), "timestamps and true poses should have the same size!");
        okvis::ImuMeasurementDeque imuMeasurements;
        cst.getTrueInertialMeasurements(tStart-okvis::Duration(1), tEnd+okvis::Duration(1), imuMeasurements);
        okvis::ImuMeasurementDeque trueBiases = imuMeasurements; // true biases used for computing RMSE

        if(cases[c].addImuNoise){
            Eigen::Vector3d bgk = Eigen::Vector3d::Zero();
            Eigen::Vector3d bak = Eigen::Vector3d::Zero();
            for (size_t i = 0; i < imuMeasurements.size(); ++i) {
                if(bVerbose)
                {
                    Eigen::Vector3d porterGyro = imuMeasurements[i].measurement.gyroscopes;
                    Eigen::Vector3d porterAcc = imuMeasurements[i].measurement.accelerometers;
                    inertialStream << imuMeasurements[i].timeStamp<<" "<<porterGyro[0]<<" "<<porterGyro[1]<<" "<<
                                      porterGyro[2]<<" "<<porterAcc[0]<<" "<<porterAcc[1]<<" "<< porterAcc[2];

                }

                trueBiases[i].measurement.gyroscopes = bgk;
                trueBiases[i].measurement.accelerometers = bak;
                imuMeasurements[i].measurement.gyroscopes += (bgk +
                        vio::Sample::gaussian(imuParameters.sigma_g_c / sqrt(DT)/imuNoiseFactor, 3));
                imuMeasurements[i].measurement.accelerometers += (bak +
                        vio::Sample::gaussian(imuParameters.sigma_a_c / sqrt(DT)/imuNoiseFactor, 3));
                bgk += vio::Sample::gaussian(imuParameters.sigma_gw_c * sqrt(DT), 3);
                bak += vio::Sample::gaussian(imuParameters.sigma_aw_c * sqrt(DT), 3);
                if(bVerbose)
                {
                    Eigen::Vector3d porterGyro = imuMeasurements[i].measurement.gyroscopes;
                    Eigen::Vector3d porterAcc = imuMeasurements[i].measurement.accelerometers;
                    inertialStream <<" "<<porterGyro[0]<<" "<<porterGyro[1]<<" "<<
                                      porterGyro[2]<<" "<<porterAcc[0]<<" "<<porterAcc[1]<<" "<< porterAcc[2]<<std::endl;

                }

            }
        }
        else
        {
            for (size_t i = 0; i < imuMeasurements.size(); ++i) {
                trueBiases[i].measurement.gyroscopes.setZero();
                trueBiases[i].measurement.accelerometers.setZero();
            }
        }
        // remove the padding part of trueBiases
        auto tempIter = trueBiases.begin();
        for(; tempIter!= trueBiases.end(); ++tempIter)
        {
            if(fabs((tempIter->timeStamp - times.front()).toSec())<1e-8)
                break;
        }
        trueBiases.erase(trueBiases.begin(), tempIter);
        // create the map
        std::shared_ptr<okvis::ceres::Map> mapPtr(new okvis::ceres::Map);

        // camera extrinsics:
        Eigen::Matrix<double, 4,4 > matT_SC0;
        matT_SC0<< 0, 0, 1, 0,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, 0, 1;
        std::shared_ptr<const okvis::kinematics::Transformation> T_SC_0(
                    new okvis::kinematics::Transformation(matT_SC0));

        // set up camera with intrinsics
        // true camera
        std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry0(
                    new okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion>(752, 480, 350, 360, 378, 238,
                                                                                                  okvis::cameras::RadialTangentialDistortion(0.00, 0.00, 0.000, 0.000)));

        // true camera system:
        std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem0(
                    new okvis::cameras::NCameraSystem);
        cameraSystem0->addCamera(T_SC_0, cameraGeometry0,
                                okvis::cameras::NCameraSystem::DistortionType::RadialTangential);

        // dummy camera only contains w, h
        std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry1(
                    new okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion>(752, 480, 0, 0, 0, 0,
                                                                                                  okvis::cameras::RadialTangentialDistortion(0.00, 0.00, 0.000, 0.000)));
        std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem1(
                    new okvis::cameras::NCameraSystem);
        //dummy camera system
        cameraSystem1->addCamera(T_SC_0, cameraGeometry1,
                                okvis::cameras::NCameraSystem::DistortionType::RadialTangential);

        // noisy camera
        Eigen::Matrix<double,4,1> fcNoise = vio::Sample::gaussian(1,4);
        fcNoise.head<2>()*=extrinsicsEstimationParameters.sigma_focal_length;
        fcNoise.tail<2>()*=extrinsicsEstimationParameters.sigma_principal_point;
        Eigen::Matrix<double,4,1> kpNoise = vio::Sample::gaussian(1,4);
        kpNoise[0]*= extrinsicsEstimationParameters.sigma_distortion[0];
        kpNoise[1]*= extrinsicsEstimationParameters.sigma_distortion[1];
        kpNoise[2]*= extrinsicsEstimationParameters.sigma_distortion[2];
        kpNoise[3]*= extrinsicsEstimationParameters.sigma_distortion[3];
        Eigen::Vector3d p_CBNoise;
        p_CBNoise[0] = vio::gauss_rand(0, extrinsicsEstimationParameters.sigma_absolute_translation);
        p_CBNoise[1] = vio::gauss_rand(0, extrinsicsEstimationParameters.sigma_absolute_translation);
        p_CBNoise[2] = vio::gauss_rand(0, extrinsicsEstimationParameters.sigma_absolute_translation);

        okvis::kinematics::Transformation tempT_SC(matT_SC0);
        std::shared_ptr<const okvis::kinematics::Transformation> T_SC_2(
                    new okvis::kinematics::Transformation(tempT_SC.r()-  tempT_SC.C()*p_CBNoise, tempT_SC.q()));

        std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry2(
                    new okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion>(752, 480, 350+fcNoise[0], 360+fcNoise[1], 378+fcNoise[2], 238+fcNoise[3],
                                                                                                  okvis::cameras::RadialTangentialDistortion(kpNoise[0], kpNoise[1], kpNoise[2], kpNoise[3])));
        // camera system used for initilizing the estimator
        std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem2(
                    new okvis::cameras::NCameraSystem);
        if(cases[c].addPriorNoise)
            cameraSystem2->addCamera(T_SC_2, cameraGeometry2,
                                okvis::cameras::NCameraSystem::DistortionType::RadialTangential);
        else
            cameraSystem2->addCamera(T_SC_0, cameraGeometry0,
                                    okvis::cameras::NCameraSystem::DistortionType::RadialTangential);

        // create an Estimator
        double trNoisy(0);
        if(cases[c].addPriorNoise)
            trNoisy = vio::gauss_rand(0, extrinsicsEstimationParameters.sigma_tr);

#ifdef USE_MSCKF2
        okvis::MSCKF2 estimator(mapPtr, trNoisy);
#else
        okvis::HybridFilter estimator(mapPtr, trNoisy);
#endif
        // create landmark grid
        std::vector<Eigen::Vector4d,
                Eigen::aligned_allocator<Eigen::Vector4d> > homogeneousPoints;
        std::vector<uint64_t> lmIds;
//        const double xyLimit = 10, zLimit = 5, xyzIncrement = 0.5, offsetNoiseMag = 0.1;
//        const double xyLimit = 5, zLimit = 2.5, xyzIncrement = 0.25, offsetNoiseMag = 0.05;
        const double xyLimit = 5, zLimit = 2.5, xyzIncrement = 0.5, offsetNoiseMag = 0.05;
        // four walls
        double x(xyLimit), y(xyLimit), z(zLimit);
        for (y = -xyLimit; y <= xyLimit; y += xyzIncrement) {
            for (z = - zLimit; z <= zLimit; z += xyzIncrement) {
                homogeneousPoints.push_back(Eigen::Vector4d(x + vio::gauss_rand(0, offsetNoiseMag),
                                                            y + vio::gauss_rand(0, offsetNoiseMag),
                                                            z + vio::gauss_rand(0, offsetNoiseMag), 1));
                lmIds.push_back(okvis::IdProvider::instance().newId());

            }
        }

        for (y = -xyLimit; y <= xyLimit; y += xyzIncrement) {
            for (z = - zLimit; z <= zLimit; z += xyzIncrement){
                homogeneousPoints.push_back(Eigen::Vector4d(y + vio::gauss_rand(0, offsetNoiseMag),
                                                            x + vio::gauss_rand(0, offsetNoiseMag),
                                                            z + vio::gauss_rand(0, offsetNoiseMag), 1));
                lmIds.push_back(okvis::IdProvider::instance().newId());

            }
        }

        x= - xyLimit;
        for (y = -xyLimit; y <= xyLimit; y += xyzIncrement) {
            for (z = - zLimit; z <= zLimit; z += xyzIncrement){
                homogeneousPoints.push_back(Eigen::Vector4d(x + vio::gauss_rand(0, offsetNoiseMag),
                                                            y + vio::gauss_rand(0, offsetNoiseMag),
                                                            z + vio::gauss_rand(0, offsetNoiseMag), 1));
                lmIds.push_back(okvis::IdProvider::instance().newId());
            }
        }

        for (y = -xyLimit; y <= xyLimit; y += xyzIncrement) {
            for (z = - zLimit; z <= zLimit; z += xyzIncrement){
                homogeneousPoints.push_back(Eigen::Vector4d(y + vio::gauss_rand(0, offsetNoiseMag),
                                                            x + vio::gauss_rand(0, offsetNoiseMag),
                                                            z + vio::gauss_rand(0, offsetNoiseMag), 1));
                lmIds.push_back(okvis::IdProvider::instance().newId());
            }
        }
        // top
        z = zLimit;
        for (y = -xyLimit; y <= xyLimit; y += xyzIncrement) {
            for (x = -xyLimit; x <= xyLimit; x += xyzIncrement) {
                homogeneousPoints.push_back(Eigen::Vector4d(x + vio::gauss_rand(0, offsetNoiseMag),
                                                            y + vio::gauss_rand(0, offsetNoiseMag),
                                                            z + vio::gauss_rand(0, offsetNoiseMag), 1));
                lmIds.push_back(okvis::IdProvider::instance().newId());
            }
        }
        // bottom
        z = -zLimit;
        for (y = -xyLimit; y <= xyLimit; y += xyzIncrement) {
            for (x = -xyLimit; x <= xyLimit; x += xyzIncrement) {
                homogeneousPoints.push_back(Eigen::Vector4d(x + vio::gauss_rand(0, offsetNoiseMag),
                                                            y + vio::gauss_rand(0, offsetNoiseMag),
                                                            z + vio::gauss_rand(0, offsetNoiseMag), 1));
                lmIds.push_back(okvis::IdProvider::instance().newId());
            }
        }

        //save these points into file
        if(bVerbose)
        {
            auto iter = homogeneousPoints.begin();
            for(auto it = lmIds.begin(); it!=lmIds.end(); ++it, ++iter)
                pointStream<< *it <<" "<< (*iter)[0] << " "<< (*iter)[1] <<" "<< (*iter)[2] <<std::endl;
            pointStream.close();
            assert(iter == homogeneousPoints.end());
        }

        // add sensors
        estimator.addCamera(extrinsicsEstimationParameters);
        estimator.addImu(imuParameters);

        std::vector<uint64_t> multiFrameIds;

        size_t kale =0; // imu data counter
        bool bStarted = false;
        int k =-1; //number of frames used in estimator
        int trackedFeatures=0; //feature tracks observed in a frame
        okvis::Time lastKFTime = times.front();
        okvis::ImuMeasurementDeque::const_iterator trueBiasIter = trueBiases.begin();
        nees.clear();
        rmse.clear();
        try{

        for (auto iter = times.begin(), iterEnd = times.end(); iter!=iterEnd;  iter+= 10, kale+=10, trueBiasIter+=10) {
            okvis::kinematics::Transformation T_WS( qs2w[kale]);

            // assemble a multi-frame
            std::shared_ptr<okvis::MultiFrame> mf(new okvis::MultiFrame);
            mf->setId(okvis::IdProvider::instance().newId());
            mf->setTimestamp(*iter);

            // reference ID will be and stay the first frame added.
            uint64_t id = mf->id();
            multiFrameIds.push_back(id);

            okvis::Time currentKFTime = *iter;
            okvis::ImuMeasurementDeque imuSegment = getImuMeasurments(
                        imuMeasurements, lastKFTime - okvis::Duration(1), currentKFTime+ okvis::Duration(1));

            // add it in the window to create a new time instance
            if(bStarted== false){
                bStarted=true;
                k=0;
                mf->resetCameraSystemAndFrames(*cameraSystem2);
                okvis::kinematics::Transformation truePose = cst.computeGlobalPose(*iter);
                Eigen::Vector3d p_WS= truePose.r();
                Eigen::Vector3d v_WS= cst.computeGlobalLinearVelocity(*iter);

                pvstd.p_WS = p_WS;
                pvstd.q_WS = truePose.q();
                pvstd.v_WS = v_WS;

                if(cases[c].addPriorNoise){
                    // Eigen::Vector3d::Random() return -1, 1 random values
                    //                p_WS += 0.1*Eigen::Vector3d::Random();
                    v_WS += vio::Sample::gaussian(1,3).cwiseProduct(pvstd.std_v_WS);
                }
                estimator.resetInitialPVandStd(pvstd, true);
                estimator.addStates(mf, imuSegment, true);
            }
            else{
                mf->resetCameraSystemAndFrames(*cameraSystem1);//dummy
                estimator.addStates(mf, imuSegment, true);
                ++k;
            }

            // now let's add also landmark observations
            trackedFeatures = 0;
            if(cases[c].useImageObservs){
                std::vector<cv::KeyPoint> keypoints;

                for (size_t j = 0; j < homogeneousPoints.size(); ++j) {
                    for (size_t i = 0; i < mf->numFrames(); ++i) {
                        Eigen::Vector2d projection;
                        Eigen::Vector4d point_C = cameraSystem0->T_SC(i)->inverse()
                                * T_WS.inverse() * homogeneousPoints[j];
                        okvis::cameras::CameraBase::ProjectionStatus status = cameraSystem0->cameraGeometry(i)->
                                projectHomogeneous(point_C, &projection);
                        if (status == okvis::cameras::CameraBase::ProjectionStatus::Successful) {

                            Eigen::Vector2d measurement(projection);
                            if(cases[c].addImageNoise){
                                measurement[0]+=vio::gauss_rand(0, imageNoiseMag);
                                measurement[1]+=vio::gauss_rand(0, imageNoiseMag);
                            }

                            keypoints.push_back(cv::KeyPoint(measurement[0], measurement[1], 8.0));
                            mf->resetKeypoints(i,keypoints);
                            size_t numObs = estimator.numObservations(lmIds[j]);
                            if(numObs==0){
                                estimator.addLandmark(lmIds[j], homogeneousPoints[j]);
                                estimator.addObservation< okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> >
                                        (lmIds[j], mf->id(), i, mf->numKeypoints(i) - 1);
                                ++trackedFeatures;
                            }
                            else
                            {
                                double sam = vio::Sample::uniform();
                                if (sam> numObs/maxTrackLength){
                                    estimator.addObservation< okvis::cameras::PinholeCamera<
                                            okvis::cameras::RadialTangentialDistortion> >(lmIds[j], mf->id(), i, mf->numKeypoints(i) - 1);
                                    ++trackedFeatures;
                                }
                            }
                        }
                    }
                }
            }
            myAccumulator(trackedFeatures);

            // run the optimization
            estimator.optimize(false);
            estimator.applyMarginalizationStrategy();

            Eigen::Vector3d v_WS_true = cst.computeGlobalLinearVelocity(*iter);

            estimator.print(mDebug);
            if(bVerbose){
                Eigen::VectorXd allIntrinsics;
                cameraGeometry0->getIntrinsics(allIntrinsics);
                Eigen::IOFormat SpaceInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", " ", "", "", "", "");
                truthStream <<*iter<<" "<< id<<" "<<std::setfill(' ')
                           << T_WS.parameters().transpose().format(SpaceInitFmt) <<" "
                           << v_WS_true.transpose().format(SpaceInitFmt) <<" 0 0 0 0 0 0 "
                           << "1 0 0 0 1 0 0 0 1 "
                           << "0 0 0 0 0 0 0 0 0 "<< "1 0 0 0 1 0 0 0 1 "
                           << T_SC_0->inverse().r().transpose().format(SpaceInitFmt)<<" "
                           << allIntrinsics.transpose().format(SpaceInitFmt)<<" 0 0"
                           << std::endl;
            }
            Eigen::Vector3d normalizedError;
            okvis::kinematics::Transformation T_WS_est;
            uint64_t currFrameId = estimator.currentFrameId();
            estimator.get_T_WS(currFrameId, T_WS_est);
            Eigen::Vector3d delta = T_WS.r() - T_WS_est.r();
            Eigen::Vector3d alpha = vio::unskew3d(T_WS.C()*T_WS_est.C().transpose()- Eigen::Matrix3d::Identity());
            Eigen::Matrix<double,6,1> deltaPose;
            deltaPose<< delta, alpha;

            normalizedError[0] = delta.transpose()*estimator.covariance_.topLeftCorner<3,3>().inverse()*delta;
            normalizedError[1] = alpha.transpose()*estimator.covariance_.block<3,3>(3,3).inverse()*alpha;
            Eigen::Matrix<double,6,1> tempPoseError= estimator.covariance_.topLeftCorner<6,6>().ldlt().solve(deltaPose);
            normalizedError[2] = deltaPose.transpose()*tempPoseError;

            //rmse in xyz, \alpha, v_WS, bg, ba, Tg, Ts, Ta, p_CB, (fx, fy), (cx, cy), k1, k2, p1, p2, td, tr
            Eigen::Matrix<double, 15+27+13, 1> rmsError;
            rmsError.head<3>() = delta.cwiseAbs2();
            rmsError.segment<3>(3) = alpha.cwiseAbs2();
            okvis::SpeedAndBias speedAndBias_est;
            estimator.getSpeedAndBias(currFrameId, 0, speedAndBias_est);
            Eigen::Vector3d deltaV= speedAndBias_est.head<3>() - v_WS_true;
            rmsError.segment<3>(6) = deltaV.cwiseAbs2();
            rmsError.segment<3>(9) = (speedAndBias_est.segment<3>(3) - trueBiasIter->measurement.gyroscopes).cwiseAbs2();
            rmsError.segment<3>(12) = (speedAndBias_est.tail<3>()- trueBiasIter->measurement.accelerometers).cwiseAbs2();

            Eigen::Matrix<double,9,1> Tg_est;
            estimator.getSensorStateEstimateAs<okvis::ceres::ShapeMatrixParamBlock>(
                        currFrameId, 0,
            #ifdef USE_MSCKF2
                        okvis::MSCKF2::SensorStates::Imu,
                        okvis::MSCKF2::ImuSensorStates::TG,
            #else
                        okvis::HybridFilter::SensorStates::Imu,
                        okvis::HybridFilter::ImuSensorStates::TG,
            #endif
                        Tg_est);

            rmsError.segment<9>(15) = (Tg_est - eye).cwiseAbs2();

            Eigen::Matrix<double,9,1> Ts_est;
            estimator.getSensorStateEstimateAs<okvis::ceres::ShapeMatrixParamBlock>(
                        currFrameId, 0,
            #ifdef USE_MSCKF2
                        okvis::MSCKF2::SensorStates::Imu,
                        okvis::MSCKF2::ImuSensorStates::TS,
            #else
                        okvis::HybridFilter::SensorStates::Imu,
                        okvis::HybridFilter::ImuSensorStates::TS,
            #endif
                        Ts_est);
            rmsError.segment<9>(24) = Ts_est.cwiseAbs2();

            Eigen::Matrix<double,9,1> Ta_est;
            estimator.getSensorStateEstimateAs<okvis::ceres::ShapeMatrixParamBlock>(
                        currFrameId, 0,
            #ifdef USE_MSCKF2
                        okvis::MSCKF2::SensorStates::Imu,
                        okvis::MSCKF2::ImuSensorStates::TA,
            #else
                        okvis::HybridFilter::SensorStates::Imu,
                        okvis::HybridFilter::ImuSensorStates::TA,
            #endif
                        Ta_est);
            rmsError.segment<9>(33) = (Ta_est - eye).cwiseAbs2();

            Eigen::Matrix<double, 3, 1> p_CB_est;
            okvis::kinematics::Transformation T_SC_est;
            estimator.getSensorStateEstimateAs<okvis::ceres::PoseParameterBlock>(currFrameId, 0,
                         #ifdef USE_MSCKF2
                                     okvis::MSCKF2::SensorStates::Camera,
                                     okvis::MSCKF2::CameraSensorStates::T_SCi,
                         #else
                                     okvis::HybridFilter::SensorStates::Camera,
                                     okvis::HybridFilter::CameraSensorStates::T_SCi,
                         #endif
                                     T_SC_est);
            p_CB_est = T_SC_est.inverse().r();
            rmsError.segment<3>(42) = p_CB_est.cwiseAbs2();

            Eigen::VectorXd intrinsics_true;
            cameraGeometry0->getIntrinsics(intrinsics_true);
            const int nDistortionCoeffDim = okvis::cameras::RadialTangentialDistortion::NumDistortionIntrinsics;
            Eigen::VectorXd distIntrinsic_true = intrinsics_true.tail<nDistortionCoeffDim>();

            Eigen::Matrix<double, 4, 1> cameraIntrinsics_est;
            estimator.getSensorStateEstimateAs<okvis::ceres::CameraIntrinsicParamBlock>(currFrameId, 0,
                         #ifdef USE_MSCKF2
                                     okvis::MSCKF2::SensorStates::Camera,
                                     okvis::MSCKF2::CameraSensorStates::Intrinsic,
                         #else
                                     okvis::HybridFilter::SensorStates::Camera,
                                     okvis::HybridFilter::CameraSensorStates::Intrinsic,
                         #endif
                                     cameraIntrinsics_est);
            rmsError.segment<2>(45) = (cameraIntrinsics_est.head<2>() - intrinsics_true.head<2>()).cwiseAbs2();
            rmsError.segment<2>(47) = (cameraIntrinsics_est.tail<2>() - intrinsics_true.segment<2>(2)).cwiseAbs2();

            Eigen::Matrix<double,nDistortionCoeffDim, 1> cameraDistortion_est;
            estimator.getSensorStateEstimateAs<okvis::ceres::CameraDistortionParamBlock>(
                        currFrameId, 0,
            #ifdef USE_MSCKF2
                        okvis::MSCKF2::SensorStates::Camera,
                        okvis::MSCKF2::CameraSensorStates::Distortion,
            #else
                        okvis::HybridFilter::SensorStates::Camera,
                        okvis::HybridFilter::CameraSensorStates::Distortion,
            #endif
                        cameraDistortion_est);
            rmsError.segment<4>(49) = (cameraDistortion_est - distIntrinsic_true).cwiseAbs2();

            double td_est, tr_est;
            estimator.getSensorStateEstimateAs<okvis::ceres::CameraTimeParamBlock>(
                        currFrameId, 0,
            #ifdef USE_MSCKF2
                        okvis::MSCKF2::SensorStates::Camera,
                        okvis::MSCKF2::CameraSensorStates::TD,
            #else
                        okvis::HybridFilter::SensorStates::Camera,
                        okvis::HybridFilter::CameraSensorStates::TD,
            #endif
                        td_est);
            rmsError[53] = td_est*td_est;

            estimator.getSensorStateEstimateAs<okvis::ceres::CameraTimeParamBlock>(
                        currFrameId, 0,
            #ifdef USE_MSCKF2
                        okvis::MSCKF2::SensorStates::Camera,
                        okvis::MSCKF2::CameraSensorStates::TR,
            #else
                        okvis::HybridFilter::SensorStates::Camera,
                        okvis::HybridFilter::CameraSensorStates::TR,
            #endif
                        tr_est);
            rmsError[54] = tr_est*tr_est;

            nees.push_back(std::make_pair(*iter, normalizedError));
            rmse.push_back(std::make_pair(*iter, rmsError));

            lastKFTime = currentKFTime;
        }//every keyframe

        if(neesSum.empty()){
            neesSum = nees;
            rmseSum = rmse;
        }
        else{
            for(size_t jack=0; jack< neesSum.size(); ++jack){
                neesSum[jack].second += nees[jack].second;
                rmseSum[jack].second += rmse[jack].second;
            }
        }


        std::cout << "Run "<< run<<" finishes with last added frame " << k << " of tracked features " << trackedFeatures <<std::endl;

        // output track length distribution
        if(!truthStream.is_open()){
            truthStream.open (truthFile, std::ios_base::app);
        }
        estimator.printTrackLengthHistogram(truthStream);
        if(truthStream.is_open())
            truthStream.close();
        //end output track length distribution

        ++successRuns;
        }//try
        catch(std::exception & e)
        {
            if(truthStream.is_open())
                truthStream.close();
            std::cout <<"Run and last added frame "<<run<<" "<< k<<" "<< e.what()<<std::endl;
            //revert the accumulated errors and delete the corresponding file
            if(mDebug.is_open())
                mDebug.close();
            unlink(outputFile.c_str());
        }
        double elapsedTime = filterTimer.stop();
        std::cout << "Run "<< run<<" using time [sec] " << elapsedTime <<std::endl;
    }//next run
    std::cout << "number of successful runs "<<successRuns<< " out of runs "<< runs<<std::endl;

    //output the feature distribution
    if(!truthStream.is_open()){
        truthStream.open (truthFile, std::ios_base::app);
    }
    histogram_type hist = density(myAccumulator);

    double total = 0.0;
    truthStream<<"histogram of number of features in images (bin lower bound, value)"<< std::endl;
    for( size_t i = 0; i < hist.size(); i++ )
    {
      truthStream << hist[i].first << " " << hist[i].second << std::endl;
      total += hist[i].second;
    }
    truthStream.close();
    std::cout << "Total of densities: " << total << " should be 1."<<std::endl;
    //end output feature distribution

    for(auto it= neesSum.begin(); it!= neesSum.end();++it)
        (it->second)/=successRuns;
    std::ofstream neesStream;
    neesStream.open(neesFile, std::ofstream::out);
    neesStream << "%state timestamp, NEES of p_WS, \alpha_WS, T_WS "<< std::endl;
    for(auto it= neesSum.begin(); it!= neesSum.end();++it)
        neesStream << it->first<<" "<< it->second.transpose() << std::endl;
    neesStream.close();

    for(auto it= rmseSum.begin(); it!= rmseSum.end();++it)
        (it->second) = ((it->second)/successRuns).cwiseSqrt();

    std::ofstream rmseStream;
    rmseStream.open(rmseFile, std::ofstream::out);
    rmseStream << "%state timestamp, rmse in xyz, \alpha, v_WS, bg, ba, Tg, Ts, Ta, p_CB, (fx, fy), (cx, cy), k1, k2, p1, p2, td, tr "<< std::endl;
    for(auto it= rmseSum.begin(); it!= rmseSum.end();++it)
        rmseStream << it->first<<" "<< it->second.transpose() << std::endl;
    rmseStream.close();
}

void testEigenLDLT()
{
    Eigen::Matrix3d A, Ainv;
    A<<     3,     1,     2,
            1,     4,     1,
            2,     1,     5;
    Ainv <<  0.4750,-0.0750,-0.1750,
            -0.0750, 0.2750,-0.0250,
            -0.1750,-0.0250, 0.2750;
    Eigen::Vector3d b= Eigen::Vector3d::Random();
    Eigen::Vector3d error = A.ldlt().solve(b) - Ainv*b;
    std::cout<<"ldlt diff with true inv "<< error.transpose()<< std::endl;
    assert(error.lpNorm<Eigen::Infinity>() < 1e-8);
}
