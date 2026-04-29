#pragma once

#include <cuda_runtime.h>

namespace caspar {

cudaError_t ConstPinholeFocalAndExtraStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholeFocalAndExtraCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholePoseStackedToCaspar(const float* stacked_data,
                                            float* cas_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t ConstPinholePoseCasparToStacked(const float* cas_data,
                                            float* stacked_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t ConstPinholePrincipalPointStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholePrincipalPointCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPixelStackedToCaspar(const float* stacked_data,
                                      float* cas_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstPixelCasparToStacked(const float* cas_data,
                                      float* stacked_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstPointStackedToCaspar(const float* stacked_data,
                                      float* cas_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstPointCasparToStacked(const float* cas_data,
                                      float* stacked_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstSimpleRadialFocalAndExtraStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialFocalAndExtraCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPoseStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPoseCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPrincipalPointStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPrincipalPointCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t PinholeCalibStackedToCaspar(const float* stacked_data,
                                        float* cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t PinholeCalibCasparToStacked(const float* cas_data,
                                        float* stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t PinholeFocalAndExtraStackedToCaspar(const float* stacked_data,
                                                float* cas_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

cudaError_t PinholeFocalAndExtraCasparToStacked(const float* cas_data,
                                                float* stacked_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

cudaError_t PinholePoseStackedToCaspar(const float* stacked_data,
                                       float* cas_data,
                                       const unsigned int cas_stride,
                                       const unsigned int cas_offset,
                                       const unsigned int num_objects);

cudaError_t PinholePoseCasparToStacked(const float* cas_data,
                                       float* stacked_data,
                                       const unsigned int cas_stride,
                                       const unsigned int cas_offset,
                                       const unsigned int num_objects);

cudaError_t PinholePrincipalPointStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t PinholePrincipalPointCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t PointStackedToCaspar(const float* stacked_data,
                                 float* cas_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects);

cudaError_t PointCasparToStacked(const float* cas_data,
                                 float* stacked_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects);

cudaError_t SimpleRadialCalibStackedToCaspar(const float* stacked_data,
                                             float* cas_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t SimpleRadialCalibCasparToStacked(const float* cas_data,
                                             float* stacked_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t SimpleRadialFocalAndExtraStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SimpleRadialFocalAndExtraCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SimpleRadialPoseStackedToCaspar(const float* stacked_data,
                                            float* cas_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t SimpleRadialPoseCasparToStacked(const float* cas_data,
                                            float* stacked_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t SimpleRadialPrincipalPointStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SimpleRadialPrincipalPointCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

}  // namespace caspar