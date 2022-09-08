/** @file slic.h
 ** @brief SLIC superpixels (@ref slic)
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_SLIC_H
#define VL_SLIC_H

#include "generic.h"

VL_EXPORT void
vl_slic_segment (vl_uint32 * segmentation,
                 float const * image,
                 vl_size width,
                 vl_size height,
                 vl_size numChannels,
                 vl_size regionSize,
                 float regularization,
                 vl_size minRegionSize) ;

/* VL_SLIC_H */
#endif
