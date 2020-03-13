#include "base/ExpLoss.hh"

namespace dd {
namespace opt {

ExpLoss::ExpLoss(const double r, const double s) : s(s), sR2Inv(s / (r * r)) {}

void ExpLoss::Evaluate(double x, double *rho) const {
    rho[0] = std::exp(sR2Inv * x - s);
    rho[1] = sR2Inv * rho[0];
    rho[2] = sR2Inv * rho[1];
}

}  // namespace opt
}  // namespace dd
