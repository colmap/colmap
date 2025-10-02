// Copied from glog/logging.h and renamed prefix from GLOG to COLMAP. For
// backwards compatibility with older versions of glog without these macros.
//
// Original glog copyright notice:
//
// Copyright (c) 2024, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: Ray Sidney

#pragma once

#if defined(__has_builtin)
#if __has_builtin(__builtin_expect)
#define COLMAP_BUILTIN_EXPECT_PRESENT
#endif
#endif

#if !defined(COLMAP_BUILTIN_EXPECT_PRESENT) && defined(__GNUG__)
// __has_builtin is not available prior to GCC 10
#define COLMAP_BUILTIN_EXPECT_PRESENT
#endif

#if defined(COLMAP_BUILTIN_EXPECT_PRESENT)

#ifndef COLMAP_PREDICT_BRANCH_NOT_TAKEN
#define COLMAP_PREDICT_BRANCH_NOT_TAKEN(x) (__builtin_expect(x, 0))
#endif

#ifndef COLMAP_PREDICT_FALSE
#define COLMAP_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#endif

#ifndef COLMAP_PREDICT_TRUE
#define COLMAP_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#endif

#else

#ifndef COLMAP_PREDICT_BRANCH_NOT_TAKEN
#define COLMAP_PREDICT_BRANCH_NOT_TAKEN(x) x
#endif

#ifndef COLMAP_PREDICT_TRUE
#define COLMAP_PREDICT_FALSE(x) x
#endif

#ifndef COLMAP_PREDICT_TRUE
#define COLMAP_PREDICT_TRUE(x) x
#endif

#endif

#undef COLMAP_BUILTIN_EXPECT_PRESENT
