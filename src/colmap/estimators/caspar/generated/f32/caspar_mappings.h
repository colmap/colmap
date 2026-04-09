#pragma once

#include <cuda_runtime.h>

namespace caspar {

cudaError_t ConstPinholeCalib_stacked_to_caspar(const float* stacked_data,
                                                float* cas_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

cudaError_t ConstPinholeCalib_caspar_to_stacked(const float* cas_data,
                                                float* stacked_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

cudaError_t ConstPixel_stacked_to_caspar(const float* stacked_data,
                                         float* cas_data,
                                         const unsigned int cas_stride,
                                         const unsigned int cas_offset,
                                         const unsigned int num_objects);

cudaError_t ConstPixel_caspar_to_stacked(const float* cas_data,
                                         float* stacked_data,
                                         const unsigned int cas_stride,
                                         const unsigned int cas_offset,
                                         const unsigned int num_objects);

cudaError_t ConstPoint_stacked_to_caspar(const float* stacked_data,
                                         float* cas_data,
                                         const unsigned int cas_stride,
                                         const unsigned int cas_offset,
                                         const unsigned int num_objects);

cudaError_t ConstPoint_caspar_to_stacked(const float* cas_data,
                                         float* stacked_data,
                                         const unsigned int cas_stride,
                                         const unsigned int cas_offset,
                                         const unsigned int num_objects);

cudaError_t ConstPose_stacked_to_caspar(const float* stacked_data,
                                        float* cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t ConstPose_caspar_to_stacked(const float* cas_data,
                                        float* stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t ConstSimpleRadialCalib_stacked_to_caspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialCalib_caspar_to_stacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t PinholeCalib_stacked_to_caspar(const float* stacked_data,
                                           float* cas_data,
                                           const unsigned int cas_stride,
                                           const unsigned int cas_offset,
                                           const unsigned int num_objects);

cudaError_t PinholeCalib_caspar_to_stacked(const float* cas_data,
                                           float* stacked_data,
                                           const unsigned int cas_stride,
                                           const unsigned int cas_offset,
                                           const unsigned int num_objects);

cudaError_t Point_stacked_to_caspar(const float* stacked_data,
                                    float* cas_data,
                                    const unsigned int cas_stride,
                                    const unsigned int cas_offset,
                                    const unsigned int num_objects);

cudaError_t Point_caspar_to_stacked(const float* cas_data,
                                    float* stacked_data,
                                    const unsigned int cas_stride,
                                    const unsigned int cas_offset,
                                    const unsigned int num_objects);

cudaError_t Pose_stacked_to_caspar(const float* stacked_data,
                                   float* cas_data,
                                   const unsigned int cas_stride,
                                   const unsigned int cas_offset,
                                   const unsigned int num_objects);

cudaError_t Pose_caspar_to_stacked(const float* cas_data,
                                   float* stacked_data,
                                   const unsigned int cas_stride,
                                   const unsigned int cas_offset,
                                   const unsigned int num_objects);

cudaError_t SimpleRadialCalib_stacked_to_caspar(const float* stacked_data,
                                                float* cas_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

cudaError_t SimpleRadialCalib_caspar_to_stacked(const float* cas_data,
                                                float* stacked_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

}  // namespace caspar