#include "kalman_filter.h"
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
 * Please note that the Eigen library does not initialize
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in)
{
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict()
{
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z)
{
  VectorXd y = z - H_ * x_;
  Calculate(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z)
{
  float px = x_[0];
  float py = x_[1];
  float vx = x_[2];
  float vy = x_[3];

  float rho = sqrt(px * px + py * py);
  float phi = atan2(py, px);
  float rho_dot = (rho > 0.0000001) ? ((px * vx + py * vy) / rho) : 0;

  VectorXd h_(3);
  h_ << rho, phi, rho_dot;

  VectorXd y = z - h_;
  y[1] = fmod(y[1] + M_PI, 2 * M_PI) - M_PI;

  Calculate(y);
}

void KalmanFilter::Calculate(const VectorXd &y)
{
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();
  x_ = x_ + K * y;
  int size = x_.size();
  MatrixXd I = MatrixXd::Identity(size, size);
  P_ = (I - K * H_) * P_;
}
