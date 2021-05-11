
#include "swift_vio/BaseFilter.h"
#include "swift_vio/EkfUpdater.h"

#include <Eigen/Dense>

namespace swift_vio {
BaseFilter::BaseFilter()
    : numResiduals_(0),
      computeKalmanGainTimer("3.1.2 computeKalmanGain", true),
      updateStatesTimer("3.1.3 updateStates", true),
      updateCovarianceTimer("3.1.4 updateCovariance", true) {}

BaseFilter::~BaseFilter() {

}

int BaseFilter::computeStackedJacobianAndResidual(
    Eigen::MatrixXd */*T_H*/, Eigen::Matrix<double, Eigen::Dynamic, 1> */*r_q*/,
    Eigen::MatrixXd */*R_q*/) {
  return 0;
}

void BaseFilter::updateIekf(int variableStartIndex, int variableDim,
                            int maxNumIteration,
                            double updateVecNormTermination) {
  // (1) Iterated extended Kalman filter based visual-inertial odometry using
  // direct photometric feedback on:
  // https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/263423/ROVIO.pdf?sequence=1&isAllowed=y
  // (2) Performance evaluation of iterated extended Kalman filter with variable
  // step-length on:
  // https://iopscience.iop.org/article/10.1088/1742-6596/659/1/012022/pdf (3)
  // Faraz Mirzaei, a Kalman filter based algorithm for IMU-Camera calibration
  // on:
  // http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.6717&rep=rep1&type=pdf

  // Initial condition: $x_k^0 = x_{k|k-1}$
  // in each iteration,
  // $\Delta x_i = K_k^i(z - h(x_k^i) - H_k^i(x_{k|k-1}\boxminus x_k^i)) +
  // x_{k|k-1}\boxminus x_k^i$ $x_k^{i+1} =  x_k^i\boxplus \Delta x_i$

  // We record the initial states, and update the estimator states in each
  // iteration which are used in computing Jacobians, and initializing
  // landmarks.
  StatePointerAndEstimateList initialStates;
  cloneFilterStates(&initialStates);

  int numIteration = 0;
  DefaultEkfUpdater updater(covariance_, variableStartIndex, variableDim);
  while (numIteration < maxNumIteration) {
    Eigen::MatrixXd T_H, R_q;
    Eigen::Matrix<double, Eigen::Dynamic, 1> r_q;
    numResiduals_ = computeStackedJacobianAndResidual(&T_H, &r_q, &R_q);
    if (numResiduals_ == 0) {
      return;
    }
    computeKalmanGainTimer.start();
    Eigen::VectorXd totalCorrection;
    boxminusFromInput(initialStates, &totalCorrection);
    Eigen::VectorXd deltax =
        updater.computeCorrection(T_H, r_q, R_q, &totalCorrection);
    computeKalmanGainTimer.stop();
    updateStatesTimer.start();
    updateStates(deltax);
    updateStatesTimer.stop();
    double lpNorm = deltax.lpNorm<Eigen::Infinity>();
    //      LOG(INFO) << "num iteration " << numIteration << " deltax norm " <<
    //      lpNorm;
    if (lpNorm < updateVecNormTermination)
      break;
    ++numIteration;
  }
  updateCovarianceTimer.start();
  updater.updateCovariance(&covariance_);
  updateCovarianceTimer.stop();
}

void BaseFilter::updateEkf(int variableStartIndex, int variableDim) {
  Eigen::MatrixXd T_H, R_q;
  Eigen::Matrix<double, Eigen::Dynamic, 1> r_q;
  numResiduals_ = computeStackedJacobianAndResidual(&T_H, &r_q, &R_q);
  if (numResiduals_ == 0) {
    return;
  }
  DefaultEkfUpdater updater(covariance_, variableStartIndex, variableDim);
  computeKalmanGainTimer.start();
  Eigen::Matrix<double, Eigen::Dynamic, 1> deltaX =
      updater.computeCorrection(T_H, r_q, R_q);
  computeKalmanGainTimer.stop();
  updateStatesTimer.start();
  updateStates(deltaX);
  updateStatesTimer.stop();
  updateCovarianceTimer.start();
  updater.updateCovariance(&covariance_);
  updateCovarianceTimer.stop();
}

} // namespace swift_vio
