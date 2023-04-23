#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
   VectorXd rmse(4);
   rmse << 0, 0, 0, 0;

   if (estimations.size() != ground_truth.size() || estimations.size() == 0)
   {
      std::cout << "Invalid estimation or ground_truth data" << std::endl;
      return rmse;
   }

   for (int i = 0; i < estimations.size(); ++i)
   {
      VectorXd c = estimations[i] - ground_truth[i];
      rmse = rmse.array() + c.array() * c.array();
   }

   // mean calculation
   rmse = rmse.array() / estimations.size();

   // squared root calculated
   rmse = rmse.array().sqrt();

   std::cout << rmse << std::endl;

   return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state)
{
   MatrixXd Hj = MatrixXd::Zero(3, 4);

   float px = x_state(0);
   float py = x_state(1);
   float vx = x_state(2);
   float vy = x_state(3);

   // calculating constants
   float c1 = px * px + py * py;
   float c2 = sqrt(c1);
   float c3 = c1 * c2;

   // check for zero
   if (fabs(c1) < 0.0001)
   {
      std::cout << "Error: CalculateJacobian - Division by Zero" << std::endl;
      return Hj;
   }

   // compute the Jacobian matrix
   Hj << px / c2, py / c2, 0, 0,
       -py / c1, px / c1, 0, 0,
       py * (vx * py - vy * px) / c3, px * (px * vy - py * vx) / c3, px / c2, py / c2;

   return Hj;
}
