#pragma once

#include "cuda_to_hip.h"

namespace caspar {

cudaError_t ConstOpenCVFocalAndExtraStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t ConstOpenCVFocalAndExtraCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t ConstOpenCVPoseStackedToCaspar(const float *stacked_data,
                                           float *cas_data,
                                           const unsigned int cas_stride,
                                           const unsigned int cas_offset,
                                           const unsigned int num_objects);

cudaError_t ConstOpenCVPoseCasparToStacked(const float *cas_data,
                                           float *stacked_data,
                                           const unsigned int cas_stride,
                                           const unsigned int cas_offset,
                                           const unsigned int num_objects);

cudaError_t ConstOpenCVPrincipalPointStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t ConstOpenCVPrincipalPointCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t ConstOpenCVSensorFromRigStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t ConstOpenCVSensorFromRigCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t ConstPinholeFocalStackedToCaspar(const float *stacked_data,
                                             float *cas_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t ConstPinholeFocalCasparToStacked(const float *cas_data,
                                             float *stacked_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t ConstPinholePoseStackedToCaspar(const float *stacked_data,
                                            float *cas_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t ConstPinholePoseCasparToStacked(const float *cas_data,
                                            float *stacked_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t ConstPinholePrincipalPointStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t ConstPinholePrincipalPointCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t ConstPinholeSensorFromRigStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t ConstPinholeSensorFromRigCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t ConstPixelStackedToCaspar(const float *stacked_data,
                                      float *cas_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstPixelCasparToStacked(const float *cas_data,
                                      float *stacked_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstPointStackedToCaspar(const float *stacked_data,
                                      float *cas_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstPointCasparToStacked(const float *cas_data,
                                      float *stacked_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstSimpleRadialFocalAndExtraStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t ConstSimpleRadialFocalAndExtraCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t ConstSimpleRadialPoseStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t ConstSimpleRadialPoseCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t ConstSimpleRadialPrincipalPointStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t ConstSimpleRadialPrincipalPointCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t ConstSimpleRadialSensorFromRigStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t ConstSimpleRadialSensorFromRigCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t OpenCVCalibStackedToCaspar(const float *stacked_data,
                                       float *cas_data,
                                       const unsigned int cas_stride,
                                       const unsigned int cas_offset,
                                       const unsigned int num_objects);

cudaError_t OpenCVCalibCasparToStacked(const float *cas_data,
                                       float *stacked_data,
                                       const unsigned int cas_stride,
                                       const unsigned int cas_offset,
                                       const unsigned int num_objects);

cudaError_t OpenCVFocalAndExtraStackedToCaspar(const float *stacked_data,
                                               float *cas_data,
                                               const unsigned int cas_stride,
                                               const unsigned int cas_offset,
                                               const unsigned int num_objects);

cudaError_t OpenCVFocalAndExtraCasparToStacked(const float *cas_data,
                                               float *stacked_data,
                                               const unsigned int cas_stride,
                                               const unsigned int cas_offset,
                                               const unsigned int num_objects);

cudaError_t OpenCVPoseStackedToCaspar(const float *stacked_data,
                                      float *cas_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t OpenCVPoseCasparToStacked(const float *cas_data,
                                      float *stacked_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t OpenCVPrincipalPointStackedToCaspar(const float *stacked_data,
                                                float *cas_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

cudaError_t OpenCVPrincipalPointCasparToStacked(const float *cas_data,
                                                float *stacked_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

cudaError_t PinholeCalibStackedToCaspar(const float *stacked_data,
                                        float *cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t PinholeCalibCasparToStacked(const float *cas_data,
                                        float *stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t PinholeFocalStackedToCaspar(const float *stacked_data,
                                        float *cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t PinholeFocalCasparToStacked(const float *cas_data,
                                        float *stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t PinholePoseStackedToCaspar(const float *stacked_data,
                                       float *cas_data,
                                       const unsigned int cas_stride,
                                       const unsigned int cas_offset,
                                       const unsigned int num_objects);

cudaError_t PinholePoseCasparToStacked(const float *cas_data,
                                       float *stacked_data,
                                       const unsigned int cas_stride,
                                       const unsigned int cas_offset,
                                       const unsigned int num_objects);

cudaError_t PinholePrincipalPointStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t PinholePrincipalPointCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t PointStackedToCaspar(const float *stacked_data, float *cas_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects);

cudaError_t PointCasparToStacked(const float *cas_data, float *stacked_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects);

cudaError_t SimpleRadialCalibStackedToCaspar(const float *stacked_data,
                                             float *cas_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t SimpleRadialCalibCasparToStacked(const float *cas_data,
                                             float *stacked_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t SimpleRadialFocalAndExtraStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t SimpleRadialFocalAndExtraCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t SimpleRadialPoseStackedToCaspar(const float *stacked_data,
                                            float *cas_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t SimpleRadialPoseCasparToStacked(const float *cas_data,
                                            float *stacked_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t SimpleRadialPrincipalPointStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

cudaError_t SimpleRadialPrincipalPointCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects);

} // namespace caspar