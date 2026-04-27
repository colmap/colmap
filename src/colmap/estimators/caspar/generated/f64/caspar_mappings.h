#pragma once

#include <cuda_runtime.h>

namespace caspar {

cudaError_t ConstPinholeFocalAndExtra_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholeFocalAndExtra_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholePose_stacked_to_caspar(const double* stacked_data,
                                               double* cas_data,
                                               const unsigned int cas_stride,
                                               const unsigned int cas_offset,
                                               const unsigned int num_objects);

cudaError_t ConstPinholePose_caspar_to_stacked(const double* cas_data,
                                               double* stacked_data,
                                               const unsigned int cas_stride,
                                               const unsigned int cas_offset,
                                               const unsigned int num_objects);

cudaError_t ConstPinholePrincipalPoint_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholePrincipalPoint_caspar_to_stacked(
    const double* cas_data,
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

cudaError_t ConstSimpleRadialFocalAndExtra_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialFocalAndExtra_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPose_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPose_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPrincipalPoint_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPrincipalPoint_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t PinholeCalib_stacked_to_caspar(const double* stacked_data,
                                           double* cas_data,
                                           const unsigned int cas_stride,
                                           const unsigned int cas_offset,
                                           const unsigned int num_objects);

cudaError_t PinholeCalib_caspar_to_stacked(const double* cas_data,
                                           double* stacked_data,
                                           const unsigned int cas_stride,
                                           const unsigned int cas_offset,
                                           const unsigned int num_objects);

cudaError_t PinholeFocalAndExtra_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t PinholeFocalAndExtra_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t PinholePose_stacked_to_caspar(const double* stacked_data,
                                          double* cas_data,
                                          const unsigned int cas_stride,
                                          const unsigned int cas_offset,
                                          const unsigned int num_objects);

cudaError_t PinholePose_caspar_to_stacked(const double* cas_data,
                                          double* stacked_data,
                                          const unsigned int cas_stride,
                                          const unsigned int cas_offset,
                                          const unsigned int num_objects);

cudaError_t PinholePrincipalPoint_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t PinholePrincipalPoint_caspar_to_stacked(
    const double* cas_data,
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

cudaError_t SimpleRadialCalib_stacked_to_caspar(const double* stacked_data,
                                                double* cas_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

cudaError_t SimpleRadialCalib_caspar_to_stacked(const double* cas_data,
                                                double* stacked_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

cudaError_t SimpleRadialFocalAndExtra_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SimpleRadialFocalAndExtra_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SimpleRadialPose_stacked_to_caspar(const double* stacked_data,
                                               double* cas_data,
                                               const unsigned int cas_stride,
                                               const unsigned int cas_offset,
                                               const unsigned int num_objects);

cudaError_t SimpleRadialPose_caspar_to_stacked(const double* cas_data,
                                               double* stacked_data,
                                               const unsigned int cas_stride,
                                               const unsigned int cas_offset,
                                               const unsigned int num_objects);

cudaError_t SimpleRadialPrincipalPoint_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SimpleRadialPrincipalPoint_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

}  // namespace caspar