// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#define TEST_NAME "base/reconstruction_manager"
#include "util/testing.h"

#include "base/reconstruction_manager.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestEmpty) {
  ReconstructionManager reconstruction_manager;
  BOOST_CHECK_EQUAL(reconstruction_manager.Size(), 0);
}

BOOST_AUTO_TEST_CASE(TestAddGet) {
  ReconstructionManager reconstruction_manager;
  BOOST_CHECK_EQUAL(reconstruction_manager.Size(), 0);
  for (size_t i = 0; i < 10; ++i) {
    const size_t idx = reconstruction_manager.Add();
    BOOST_CHECK_EQUAL(reconstruction_manager.Size(), i + 1);
    BOOST_CHECK_EQUAL(idx, i);
    BOOST_CHECK_EQUAL(reconstruction_manager.Get(idx).NumCameras(), 0);
    BOOST_CHECK_EQUAL(reconstruction_manager.Get(idx).NumImages(), 0);
    BOOST_CHECK_EQUAL(reconstruction_manager.Get(idx).NumPoints3D(), 0);
  }
}

BOOST_AUTO_TEST_CASE(TestDelete) {
  ReconstructionManager reconstruction_manager;
  BOOST_CHECK_EQUAL(reconstruction_manager.Size(), 0);
  for (size_t i = 0; i < 10; ++i) {
    reconstruction_manager.Add();
  }

  BOOST_CHECK_EQUAL(reconstruction_manager.Size(), 10);
  for (size_t i = 0; i < 10; ++i) {
    reconstruction_manager.Delete(0);
    BOOST_CHECK_EQUAL(reconstruction_manager.Size(), 9 - i);
  }
}

BOOST_AUTO_TEST_CASE(TestClear) {
  ReconstructionManager reconstruction_manager;
  BOOST_CHECK_EQUAL(reconstruction_manager.Size(), 0);
  for (size_t i = 0; i < 10; ++i) {
    reconstruction_manager.Add();
  }

  BOOST_CHECK_EQUAL(reconstruction_manager.Size(), 10);
  reconstruction_manager.Clear();
  BOOST_CHECK_EQUAL(reconstruction_manager.Size(), 0);
}
