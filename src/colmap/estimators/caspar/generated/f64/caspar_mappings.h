#pragma once

#include <cuda_runtime.h>

namespace caspar {

cudaError_t ConstPinholeExtraCalib_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholeExtraCalib_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholeFocal_stacked_to_caspar(const double* stacked_data,
                                                double* cas_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

cudaError_t ConstPinholeFocal_caspar_to_stacked(const double* cas_data,
                                                double* stacked_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

cudaError_t ConstPixel_stacked_to_caspar(const double* stacked_data,
                                         double* cas_data,
                                         const unsigned int cas_stride,
                                         const unsigned int cas_offset,
                                         const unsigned int num_objects);

cudaError_t ConstPixel_caspar_to_stacked(const double* cas_data,
                                         double* stacked_data,
                                         const unsigned int cas_stride,
                                         const unsigned int cas_offset,
                                         const unsigned int num_objects);

cudaError_t ConstPoint_stacked_to_caspar(const double* stacked_data,
                                         double* cas_data,
                                         const unsigned int cas_stride,
                                         const unsigned int cas_offset,
                                         const unsigned int num_objects);

cudaError_t ConstPoint_caspar_to_stacked(const double* cas_data,
                                         double* stacked_data,
                                         const unsigned int cas_stride,
                                         const unsigned int cas_offset,
                                         const unsigned int num_objects);

cudaError_t ConstPose_stacked_to_caspar(const double* stacked_data,
                                        double* cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t ConstPose_caspar_to_stacked(const double* cas_data,
                                        double* stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t ConstSimpleRadialExtraCalib_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialExtraCalib_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialFocal_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialFocal_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t PinholeExtraCalib_stacked_to_caspar(const double* stacked_data,
                                                double* cas_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

cudaError_t PinholeExtraCalib_caspar_to_stacked(const double* cas_data,
                                                double* stacked_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

cudaError_t PinholeFocal_stacked_to_caspar(const double* stacked_data,
                                           double* cas_data,
                                           const unsigned int cas_stride,
                                           const unsigned int cas_offset,
                                           const unsigned int num_objects);

cudaError_t PinholeFocal_caspar_to_stacked(const double* cas_data,
                                           double* stacked_data,
                                           const unsigned int cas_stride,
                                           const unsigned int cas_offset,
                                           const unsigned int num_objects);

cudaError_t Point_stacked_to_caspar(const double* stacked_data,
                                    double* cas_data,
                                    const unsigned int cas_stride,
                                    const unsigned int cas_offset,
                                    const unsigned int num_objects);

cudaError_t Point_caspar_to_stacked(const double* cas_data,
                                    double* stacked_data,
                                    const unsigned int cas_stride,
                                    const unsigned int cas_offset,
                                    const unsigned int num_objects);

cudaError_t Pose_stacked_to_caspar(const double* stacked_data,
                                   double* cas_data,
                                   const unsigned int cas_stride,
                                   const unsigned int cas_offset,
                                   const unsigned int num_objects);

cudaError_t Pose_caspar_to_stacked(const double* cas_data,
                                   double* stacked_data,
                                   const unsigned int cas_stride,
                                   const unsigned int cas_offset,
                                   const unsigned int num_objects);

cudaError_t SimpleRadialExtraCalib_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SimpleRadialExtraCalib_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SimpleRadialFocal_stacked_to_caspar(const double* stacked_data,
                                                double* cas_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

cudaError_t SimpleRadialFocal_caspar_to_stacked(const double* cas_data,
                                                double* stacked_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

}  // namespace caspar