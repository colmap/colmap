#pragma once

#include <cuda_runtime.h>

namespace caspar {

cudaError_t ConstPinholeFocalStackedToCaspar(const double* stacked_data,
                                             double* cas_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t ConstPinholeFocalCasparToStacked(const double* cas_data,
                                             double* stacked_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t ConstPinholePoseStackedToCaspar(const double* stacked_data,
                                            double* cas_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t ConstPinholePoseCasparToStacked(const double* cas_data,
                                            double* stacked_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t ConstPinholePrincipalPointStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholePrincipalPointCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholeSensorFromRigStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholeSensorFromRigCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPixelStackedToCaspar(const double* stacked_data,
                                      double* cas_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstPixelCasparToStacked(const double* cas_data,
                                      double* stacked_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstPointStackedToCaspar(const double* stacked_data,
                                      double* cas_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstPointCasparToStacked(const double* cas_data,
                                      double* stacked_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstSimpleRadialFocalAndExtraStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialFocalAndExtraCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPoseStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPoseCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPrincipalPointStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPrincipalPointCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialSensorFromRigStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialSensorFromRigCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t PinholeCalibStackedToCaspar(const double* stacked_data,
                                        double* cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t PinholeCalibCasparToStacked(const double* cas_data,
                                        double* stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t PinholeFocalStackedToCaspar(const double* stacked_data,
                                        double* cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t PinholeFocalCasparToStacked(const double* cas_data,
                                        double* stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t PinholePoseStackedToCaspar(const double* stacked_data,
                                       double* cas_data,
                                       const unsigned int cas_stride,
                                       const unsigned int cas_offset,
                                       const unsigned int num_objects);

cudaError_t PinholePoseCasparToStacked(const double* cas_data,
                                       double* stacked_data,
                                       const unsigned int cas_stride,
                                       const unsigned int cas_offset,
                                       const unsigned int num_objects);

cudaError_t PinholePrincipalPointStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t PinholePrincipalPointCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t PointStackedToCaspar(const double* stacked_data,
                                 double* cas_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects);

cudaError_t PointCasparToStacked(const double* cas_data,
                                 double* stacked_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects);

cudaError_t SimpleRadialCalibStackedToCaspar(const double* stacked_data,
                                             double* cas_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t SimpleRadialCalibCasparToStacked(const double* cas_data,
                                             double* stacked_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t SimpleRadialFocalAndExtraStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SimpleRadialFocalAndExtraCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SimpleRadialPoseStackedToCaspar(const double* stacked_data,
                                            double* cas_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t SimpleRadialPoseCasparToStacked(const double* cas_data,
                                            double* stacked_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t SimpleRadialPrincipalPointStackedToCaspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SimpleRadialPrincipalPointCasparToStacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

}  // namespace caspar