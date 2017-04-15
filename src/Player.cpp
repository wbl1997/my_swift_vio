/**
 * @file Player.cpp
 * @brief Source file for the Player class.
 * @author Jianzhu Huai
 */

#include <glog/logging.h>
#include <okvis/Player.hpp>
#include <functional>

#include <thread>

#define THRESHOLD_DATA_DELAY_WARNING 0.1 // in seconds

/// \brief okvis Main namespace of this package.
namespace okvis {

Player::~Player()
{
}

Player::Player(okvis::VioInterface* vioInterfacePtr,
               const okvis::VioParameters & params)
    : vioInterface_(vioInterfacePtr),
      vioParameters_(params),
      mVideoFile(params.input.videoFile),
      mImuFile(params.input.imuFile),
      mIG(mImuFile, vio::SensorStreamCSV,0.005),
      mFG(mVideoFile, params.input.startIndex, params.input.finishIndex),
      mSG(NULL)
{

}

Player::Player(okvis::VioInterface* vioInterfacePtr,
               const okvis::VioParameters & params, std::string imageFolder)
    : vioInterface_(vioInterfacePtr),
      vioParameters_(params),
      mImageFolder(params.input.imageFolder),
      mTimeFile(params.input.timeFile),
      mImuFile(params.input.imuFile),
      mIG(mImuFile, vio::PlainText,0.01),
      mFG(mImageFolder, mTimeFile, params.input.startIndex, params.input.finishIndex),
      mSG(NULL)
{

}

void Player::Run()
{
    //for the first frame
    ros::Rate rate(vioParameters_.sensors_information.cameraRate);
    std::cout <<"camera frame rate "<<vioParameters_.sensors_information.cameraRate<<std::endl;
    const double advance = 0.5; //+ advance to retrieve a little more imu data, thus,
            // avoid waiting in processing frames which causes false warning of delayed frames
    cv::Mat frame;
    double frameTime;
    int frameCounter(0);

    while(mFG.grabFrame(frame, frameTime)){

        cv::Mat filtered;
        if (vioParameters_.optimization.useMedianFilter) {
            cv::medianBlur(frame, filtered, 3);
        } else {
            filtered = frame;
        }
        okvis::Time t(frameTime);
        t -= okvis::Duration(vioParameters_.sensors_information.imageDelay);
        std::cout <<"read in frame at "<<std::setprecision(12)<<t.toSec()<<std::endl;
        vioInterface_->addImage(t, 0, filtered, NULL, mFG.getCurrentId());

        // add corresponding imu data
        if(frameCounter ==0){
            mIG.getObservation(t.toSec() - 0.1); // 0.1 to avoid reading the first entry that may be useful for later processing
        }

        bool isMeasGood = mIG.getObservation(t.toSec() + advance);
        if(!isMeasGood) // the measurements can be bad when appraoching the end of a file
        {
            ++frameCounter;
            rate.sleep();
            continue;
        }
        std::vector<Eigen::Matrix<double, 7,1> > imuObservations = mIG.measurement;

        imuObservations.pop_back(); //remove the first entry which was the last in the previous observations

//        std::cout <<" start and finish timestamp " <<std::setprecision(12)<< imuObservations.front()[0] <<" "<< imuObservations.back()[0]<<std::endl;
        for(const Eigen::Matrix<double, 7,1> & obs: imuObservations)
        {            
            vioInterface_->addImuMeasurement(okvis::Time(obs[0]),
                                             obs.segment<3>(1), obs.segment<3>(4));
        }
        ++frameCounter;
        rate.sleep();
    }
}

enum VIOStage{NotInitialized= 0, Initializing, Initialized};

// In this case, feature associations, i.e. tracks are from the external VO input. The pose records by the external VO is
// also required because they are used to prune frames that do not have recorded pose and hence recorded feature observations
// the pose records is simply a byproduct of visual odometry in generating the feature tracks.
// As a result, the image frames are only used for visualization,
void Player::RunWithSavedTracks()
{

    std::string orbvo_output = vioParameters_.input.voPosesFile;
    if(orbvo_output.empty() || vioParameters_.input.voFeatureTracksFile.empty())
    {
        std::cerr<<"Please specify the vo output files for both poses and feature tracks!"<< std::endl;
        return;
    }
    else
    {
        std::cout <<"Reading in poses and feature tracks from "<< orbvo_output<< std::endl<<
                    vioParameters_.input.voFeatureTracksFile << std::endl;
    }

    //for the first frame
    ros::Rate rate(vioParameters_.sensors_information.cameraRate);
    std::cout <<"camera frame rate "<<vioParameters_.sensors_information.cameraRate<<std::endl;
    const double advance = 0.5; //+ advance to retrieve a little more imu data, thus,
            // avoid waiting in processing frames which causes false warning of delayed frames
    cv::Mat frame;
    double frameTime;
    int frameCounter(0);

    bool bUseExternalInitState = vioParameters_.initialState.bUseExternalInitState;
    VIOStage stage(NotInitialized);
    okvis::Time initStateTime = vioParameters_.initialState.stateTime;

    mSG= new vio::StatesGrabber(orbvo_output, 17);

    while(mFG.grabFrame(frame, frameTime)){

        cv::Mat filtered;
        if (vioParameters_.optimization.useMedianFilter) {
            cv::medianBlur(frame, filtered, 3);
        } else {
            filtered = frame;
        }
        okvis::Time t(frameTime);
        t -= okvis::Duration(vioParameters_.sensors_information.imageDelay);
        if(!mSG->getObservation(t.toSec()))
        {
            std::cerr <<"unable to locate orbvo entry for frame at "<<std::setprecision(12)<< t.toSec()<<std::endl;
            continue;
        }

        // wait until the time of the external initial state
        if( bUseExternalInitState)
        {
            double deltaTime = initStateTime.toSec() - t.toSec();
            if(deltaTime > 0.1)
            {
                continue;
            }
            else if( deltaTime < -0.1 )
            {
                assert(stage == Initialized);
            }
            else//( initStateTime - t < okvis::Duration(0.1) &&  t - initStateTime < okvis::Duration(0.1))
            {
                std::cout <<" Coming to initial state at "<<std::setprecision(12)<<initStateTime.toSec()<<std::endl;
                stage = Initialized;
            }
        }

        std::cout <<"read in frame at "<<std::setprecision(12)<<t.toSec()<<std::endl;

        // add corresponding imu data
        if(frameCounter ==0){
            mIG.getObservation(t.toSec() - 0.1); // 0.1 to avoid reading the first entry that may be useful for later processing
        }
        ++frameCounter;

        bool isMeasGood = mIG.getObservation(t.toSec() + advance);
        if(!isMeasGood) // the measurements can be bad when appraoching the end of a file
        {
            std::cerr<<"Warn: failed to get inertial observation up to "<<std::setprecision(12)<< t.toSec() +advance <<std::endl;
            rate.sleep();
            continue;
        }

        std::vector<Eigen::Matrix<double, 7,1> > imuObservations = mIG.measurement;
        imuObservations.pop_back(); //remove the first entry which was the last in the previous observations

        std::cout <<" start and finish timestamp " <<std::setprecision(12)<< imuObservations.front()[0] <<" "<< imuObservations.back()[0]<<std::endl;
        for(const Eigen::Matrix<double, 7,1> & obs: imuObservations)
        {
            vioInterface_->addImuMeasurement(okvis::Time(obs[0]),
                                             obs.segment<3>(1), obs.segment<3>(4));
        }
        vioInterface_->addImage(t, 0, filtered, NULL, mFG.getCurrentId());

        rate.sleep();
    }
    std::cout <<"frame grabber finishes "<<std::endl;
    ros::shutdown();
}

} // namespace okvis
