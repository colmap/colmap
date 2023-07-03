// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2019 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: keir@google.com (Keir Mierle)
//         sameeragarwal@google.com (Sameer Agarwal)

#ifndef CERES_PUBLIC_LOCAL_PARAMETERIZATION_H_
#define CERES_PUBLIC_LOCAL_PARAMETERIZATION_H_

#include <array>
#include <memory>
#include <vector>

#include "ceres/internal/disable_warnings.h"
#include "ceres/internal/export.h"
#include "ceres/internal/port.h"

namespace ceres {

// WARNING: LocalParameterizations are deprecated. They will be removed from
// Ceres Solver in version 2.2.0. Please use Manifolds instead.

// Purpose: Sometimes parameter blocks x can overparameterize a problem
//
//   min f(x)
//    x
//
// In that case it is desirable to choose a parameterization for the
// block itself to remove the null directions of the cost. More
// generally, if x lies on a manifold of a smaller dimension than the
// ambient space that it is embedded in, then it is numerically and
// computationally more effective to optimize it using a
// parameterization that lives in the tangent space of that manifold
// at each point.
//
// For example, a sphere in three dimensions is a 2 dimensional
// manifold, embedded in a three dimensional space. At each point on
// the sphere, the plane tangent to it defines a two dimensional
// tangent space. For a cost function defined on this sphere, given a
// point x, moving in the direction normal to the sphere at that point
// is not useful. Thus a better way to do a local optimization is to
// optimize over two dimensional vector delta in the tangent space at
// that point and then "move" to the point x + delta, where the move
// operation involves projecting back onto the sphere. Doing so
// removes a redundant dimension from the optimization, making it
// numerically more robust and efficient.
//
// More generally we can define a function
//
//   x_plus_delta = Plus(x, delta),
//
// where x_plus_delta has the same size as x, and delta is of size
// less than or equal to x. The function Plus, generalizes the
// definition of vector addition. Thus it satisfies the identify
//
//   Plus(x, 0) = x, for all x.
//
// A trivial version of Plus is when delta is of the same size as x
// and
//
//   Plus(x, delta) = x + delta
//
// A more interesting case if x is two dimensional vector, and the
// user wishes to hold the first coordinate constant. Then, delta is a
// scalar and Plus is defined as
//
//   Plus(x, delta) = x + [0] * delta
//                        [1]
//
// An example that occurs commonly in Structure from Motion problems
// is when camera rotations are parameterized using Quaternion. There,
// it is useful to only make updates orthogonal to that 4-vector
// defining the quaternion. One way to do this is to let delta be a 3
// dimensional vector and define Plus to be
//
//   Plus(x, delta) = [cos(|delta|), sin(|delta|) delta / |delta|] * x
//
// The multiplication between the two 4-vectors on the RHS is the
// standard quaternion product.
//
// Given f and a point x, optimizing f can now be restated as
//
//     min  f(Plus(x, delta))
//    delta
//
// Given a solution delta to this problem, the optimal value is then
// given by
//
//   x* = Plus(x, delta)
//
// The class LocalParameterization defines the function Plus and its
// Jacobian which is needed to compute the Jacobian of f w.r.t delta.
class CERES_DEPRECATED_WITH_MSG(
    "LocalParameterizations will be removed from the Ceres Solver API in "
    "version 2.2.0. Use Manifolds instead.")
    CERES_EXPORT LocalParameterization {
 public:
  virtual ~LocalParameterization();

  // Generalization of the addition operation,
  //
  //   x_plus_delta = Plus(x, delta)
  //
  // with the condition that Plus(x, 0) = x.
  //
  virtual bool Plus(const double* x,
                    const double* delta,
                    double* x_plus_delta) const = 0;

  // The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
  //
  // jacobian is a row-major GlobalSize() x LocalSize() matrix.
  virtual bool ComputeJacobian(const double* x, double* jacobian) const = 0;

  // local_matrix = global_matrix * jacobian
  //
  // global_matrix is a num_rows x GlobalSize  row major matrix.
  // local_matrix is a num_rows x LocalSize row major matrix.
  // jacobian(x) is the matrix returned by ComputeJacobian at x.
  //
  // This is only used by GradientProblem. For most normal uses, it is
  // okay to use the default implementation.
  virtual bool MultiplyByJacobian(const double* x,
                                  const int num_rows,
                                  const double* global_matrix,
                                  double* local_matrix) const;

  // Size of x.
  virtual int GlobalSize() const = 0;

  // Size of delta.
  virtual int LocalSize() const = 0;
};

// Some basic parameterizations

// Identity Parameterization: Plus(x, delta) = x + delta
class CERES_DEPRECATED_WITH_MSG("Use EuclideanManifold instead.")
    CERES_EXPORT IdentityParameterization : public LocalParameterization {
 public:
  explicit IdentityParameterization(int size);
  bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override;
  bool ComputeJacobian(const double* x, double* jacobian) const override;
  bool MultiplyByJacobian(const double* x,
                          const int num_cols,
                          const double* global_matrix,
                          double* local_matrix) const override;
  int GlobalSize() const override { return size_; }
  int LocalSize() const override { return size_; }

 private:
  const int size_;
};

// Hold a subset of the parameters inside a parameter block constant.
class CERES_DEPRECATED_WITH_MSG("Use SubsetManifold instead.")
    CERES_EXPORT SubsetParameterization : public LocalParameterization {
 public:
  explicit SubsetParameterization(int size,
                                  const std::vector<int>& constant_parameters);
  bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override;
  bool ComputeJacobian(const double* x, double* jacobian) const override;
  bool MultiplyByJacobian(const double* x,
                          const int num_cols,
                          const double* global_matrix,
                          double* local_matrix) const override;
  int GlobalSize() const override {
    return static_cast<int>(constancy_mask_.size());
  }
  int LocalSize() const override { return local_size_; }

 private:
  const int local_size_;
  std::vector<char> constancy_mask_;
};

// Plus(x, delta) = [cos(|delta|), sin(|delta|) delta / |delta|] * x
// with * being the quaternion multiplication operator. Here we assume
// that the first element of the quaternion vector is the real (cos
// theta) part.
class CERES_DEPRECATED_WITH_MSG("Use QuaternionManifold instead.")
    CERES_EXPORT QuaternionParameterization : public LocalParameterization {
 public:
  bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override;
  bool ComputeJacobian(const double* x, double* jacobian) const override;
  int GlobalSize() const override { return 4; }
  int LocalSize() const override { return 3; }
};

// Implements the quaternion local parameterization for Eigen's representation
// of the quaternion. Eigen uses a different internal memory layout for the
// elements of the quaternion than what is commonly used. Specifically, Eigen
// stores the elements in memory as [x, y, z, w] where the real part is last
// whereas it is typically stored first. Note, when creating an Eigen quaternion
// through the constructor the elements are accepted in w, x, y, z order. Since
// Ceres operates on parameter blocks which are raw double pointers this
// difference is important and requires a different parameterization.
//
// Plus(x, delta) = [sin(|delta|) delta / |delta|, cos(|delta|)] * x
// with * being the quaternion multiplication operator.
class CERES_DEPRECATED_WITH_MSG("Use EigenQuaternionManifold instead.")
    CERES_EXPORT EigenQuaternionParameterization
    : public ceres::LocalParameterization {
 public:
  bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override;
  bool ComputeJacobian(const double* x, double* jacobian) const override;
  int GlobalSize() const override { return 4; }
  int LocalSize() const override { return 3; }
};

// This provides a parameterization for homogeneous vectors which are commonly
// used in Structure from Motion problems.  One example where they are used is
// in representing points whose triangulation is ill-conditioned. Here it is
// advantageous to use an over-parameterization since homogeneous vectors can
// represent points at infinity.
//
// The plus operator is defined as
// Plus(x, delta) =
//    [sin(0.5 * |delta|) * delta / |delta|, cos(0.5 * |delta|)] * x
//
// with * defined as an operator which applies the update orthogonal to x to
// remain on the sphere. We assume that the last element of x is the scalar
// component. The size of the homogeneous vector is required to be greater than
// 1.
class CERES_DEPRECATED_WITH_MSG("Use SphereManifold instead.") CERES_EXPORT
    HomogeneousVectorParameterization : public LocalParameterization {
 public:
  explicit HomogeneousVectorParameterization(int size);
  bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override;
  bool ComputeJacobian(const double* x, double* jacobian) const override;
  int GlobalSize() const override { return size_; }
  int LocalSize() const override { return size_ - 1; }

 private:
  const int size_;
};

// This provides a parameterization for lines, where the line is
// over-parameterized by an origin point and a direction vector. So the
// parameter vector size needs to be two times the ambient space dimension,
// where the first half is interpreted as the origin point and the second half
// as the direction.
//
// The plus operator for the line direction is the same as for the
// HomogeneousVectorParameterization. The update of the origin point is
// perpendicular to the line direction before the update.
//
// This local parameterization is a special case of the affine Grassmannian
// manifold (see https://en.wikipedia.org/wiki/Affine_Grassmannian_(manifold))
// for the case Graff_1(R^n).
template <int AmbientSpaceDimension>
class CERES_DEPRECATED_WITH_MSG("Use LineManifold instead.")
    LineParameterization : public LocalParameterization {
 public:
  static_assert(AmbientSpaceDimension >= 2,
                "The ambient space must be at least 2");

  bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override;
  bool ComputeJacobian(const double* x, double* jacobian) const override;
  int GlobalSize() const override { return 2 * AmbientSpaceDimension; }
  int LocalSize() const override { return 2 * (AmbientSpaceDimension - 1); }
};

// Construct a local parameterization by taking the Cartesian product
// of a number of other local parameterizations. This is useful, when
// a parameter block is the cartesian product of two or more
// manifolds. For example the parameters of a camera consist of a
// rotation and a translation, i.e., SO(3) x R^3.
//
// Example usage:
//
// ProductParameterization product_param(new QuaterionionParameterization(),
//                                       new IdentityParameterization(3));
//
// is the local parameterization for a rigid transformation, where the
// rotation is represented using a quaternion.
//
class CERES_DEPRECATED_WITH_MSG("Use ProductManifold instead.")
    CERES_EXPORT ProductParameterization : public LocalParameterization {
 public:
  ProductParameterization(const ProductParameterization&) = delete;
  ProductParameterization& operator=(const ProductParameterization&) = delete;
  //
  // NOTE: The constructor takes ownership of the input local
  // parameterizations.
  //
  template <typename... LocalParams>
  explicit ProductParameterization(LocalParams*... local_params)
      : local_params_(sizeof...(LocalParams)) {
    constexpr int kNumLocalParams = sizeof...(LocalParams);
    static_assert(kNumLocalParams >= 2,
                  "At least two local parameterizations must be specified.");

    using LocalParameterizationPtr = std::unique_ptr<LocalParameterization>;

    // Wrap all raw pointers into std::unique_ptr for exception safety.
    std::array<LocalParameterizationPtr, kNumLocalParams> local_params_array{
        LocalParameterizationPtr(local_params)...};

    // Initialize internal state.
    for (int i = 0; i < kNumLocalParams; ++i) {
      LocalParameterizationPtr& param = local_params_[i];
      param = std::move(local_params_array[i]);

      buffer_size_ =
          std::max(buffer_size_, param->LocalSize() * param->GlobalSize());
      global_size_ += param->GlobalSize();
      local_size_ += param->LocalSize();
    }
  }

  bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override;
  bool ComputeJacobian(const double* x, double* jacobian) const override;
  int GlobalSize() const override { return global_size_; }
  int LocalSize() const override { return local_size_; }

 private:
  std::vector<std::unique_ptr<LocalParameterization>> local_params_;
  int local_size_{0};
  int global_size_{0};
  int buffer_size_{0};
};

}  // namespace ceres

// clang-format off
#include "ceres/internal/reenable_warnings.h"
// clang-format on

#include "ceres/internal/line_parameterization.h"

#endif  // CERES_PUBLIC_LOCAL_PARAMETERIZATION_H_
