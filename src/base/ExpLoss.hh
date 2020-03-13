#pragma once

#include <ceres/ceres.h>

namespace dd {
namespace opt {

/**
 * Loss function: e^(s * ((x / r^2) - 1))
 * where
 * - x is the squared 2-norm of residual vector
 * - r defines the residual range [-r,r] where the function flat,
 *   out of this range it increases rapidly
 * - s controls how flat it is in [-r,r] and how rapidly increases outside,
 *   high means flat inside, increases fast outside,
 *   s corresponds to the slope of the loss function at r^2 (which is residual at +/-r)
 */
class ExpLoss : public ceres::LossFunction {
    const double s;
    const double sR2Inv;

public:
    explicit ExpLoss(const double r, const double s = 10);

    /**
     * For a residual vector with squared 2-norm 'sq_norm', this method
     * is required to fill in the value and derivatives of the loss
     * function (rho in this example):
     *
     *   out[0] = rho(sq_norm),
     *   out[1] = rho'(sq_norm),
     *   out[2] = rho''(sq_norm),
     *
     * Here the convention is that the contribution of a term to the
     * cost function is given by 1/2 rho(s),  where
     *
     *   s = ||residuals||^2.
     *
     * Calling the method with a negative value of 's' is an error and
     * the implementations are not required to handle that case.
     *
     * Most sane choices of rho() satisfy:
     *
     *   rho(0) = 0,
     *   rho'(0) = 1,
     *   rho'(s) < 1 in outlier region,
     *   rho''(s) < 0 in outlier region,
     *
     * so that they mimic the least squares cost for small residuals.
     *
     * (https://github.com/kashif/ceres-solver/blob/master/include/ceres/loss_function.h)
     */
    virtual void Evaluate(double x, double *rho) const;
};

}  // namespace opt
}  // namespace dd
