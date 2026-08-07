// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#pragma once

#include <string>

namespace colmap {

#ifdef COLMAP_DOWNLOAD_ENABLED
inline const std::string kDefaultAlikedN16RotFeatureExtractorUri =
    "https://github.com/colmap/colmap/releases/download/3.13.0/"
    "aliked-n16rot.onnx;"
    "aliked-n16rot.onnx;"
    "39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547";
inline const std::string kDefaultAlikedN32FeatureExtractorUri =
    "https://github.com/colmap/colmap/releases/download/3.13.0/"
    "aliked-n32.onnx;"
    "aliked-n32.onnx;"
    "a077728a02d2de1a775c66df6de8cfeb7c6b51ca57572c64c680131c988c8b3c";
inline const std::string kDefaultAlikedLightGlueFeatureMatcherUri =
    "https://github.com/colmap/colmap/releases/download/3.13.0/"
    "aliked-lightglue.onnx;"
    "aliked-lightglue.onnx;"
    "b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d";
inline const std::string kDefaultBruteForceONNXMatcherUri =
    "https://github.com/colmap/colmap/releases/download/3.13.0/"
    "bruteforce-matcher.onnx;"
    "bruteforce-matcher.onnx;"
    "3c1282f96d83f5ffc861a873298d08bbe5219f59af59223f5ceab5c41a182a47";
inline const std::string kDefaultSiftLightGlueFeatureMatcherUri =
    "https://github.com/colmap/colmap/releases/download/3.13.0/"
    "sift-lightglue.onnx;"
    "sift-lightglue.onnx;"
    "e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e";
// TODO: Release these as official colmap releases instead of hosting in
// davnords/storage
inline const std::string kDefaultLomaBDetectorUri =
    "https://github.com/davnords/storage/releases/download/loma/"
    "loma_detector.onnx;"
    "loma_detector.onnx;"
    "b6af99c5e730034ac9b675d1ebe05d0679af4569a3c26f10a6a50f91e02dc512";
inline const std::string kDefaultLomaBDescriptorUri =
    "https://github.com/davnords/storage/releases/download/loma/"
    "loma_descriptor_dedode_g.onnx;"
    "loma_descriptor_dedode_g.onnx;"
    "5a7b9eaf7425d4513c5d7feae86080bae7ed3aceae7fb1b9f059d0752e2ad564";
inline const std::string kDefaultLomaBMatcherUri =
    "https://github.com/davnords/storage/releases/download/loma/"
    "loma_matcher_B.onnx;"
    "loma_matcher_B.onnx;"
    "ba5a2773b29cace19f1240e14e5a080cca3eaf9f69a7adb829a1d470557001c7";
inline const std::string kDefaultLomaBDescriptorBf16Uri =
    "https://github.com/davnords/storage/releases/download/loma/"
    "loma_descriptor_dedode_g_bf16.onnx;"
    "loma_descriptor_dedode_g_bf16.onnx;"
    "679b3ead5385b2bf0424548ed027edd48f96eb8b7cddcb42344e899cf0d97e7b";
inline const std::string kDefaultLomaB128DescriptorUri =
    "https://github.com/davnords/storage/releases/download/loma/"
    "loma_descriptor_dedode_b.onnx;"
    "loma_descriptor_dedode_b.onnx;"
    "82660a364299013618fe649092ebc4f617559f6a77e1ab5a3412be62a47ddc2d";
inline const std::string kDefaultLomaBMatcherBf16Uri =
    "https://github.com/davnords/storage/releases/download/loma/"
    "loma_matcher_B_bf16.onnx;"
    "loma_matcher_B_bf16.onnx;"
    "e61751cdbac6f305efa0d5b2ddc8f8c8e095b1b7196bfac3ed8d4e29b007f625";
inline const std::string kDefaultLomaB128MatcherUri =
    "https://github.com/davnords/storage/releases/download/loma/"
    "loma_matcher_B128.onnx;"
    "loma_matcher_B128.onnx;"
    "e71ad490d13713374433a7ef99a7b4f4877d09338e40f347b7e64cc90150ee16";
inline const std::string kDefaultLomaB128MatcherBf16Uri =
    "https://github.com/davnords/storage/releases/download/loma/"
    "loma_matcher_B128_bf16.onnx;"
    "loma_matcher_B128_bf16.onnx;"
    "7dec0563c0955400a7bef18ba0c62a70772c9c5c3bf2734372c32de5f1876c32";
inline const std::string kDefaultLomaRMatcherUri =
    "https://github.com/davnords/storage/releases/download/loma/"
    "loma_matcher_R.onnx;"
    "loma_matcher_R.onnx;"
    "6c55247068568861d983b6005d9c401a4c62b8f8ea75d1a6925f13b0211407b5";
inline const std::string kDefaultLomaRMatcherBf16Uri =
    "https://github.com/davnords/storage/releases/download/loma/"
    "loma_matcher_R_bf16.onnx;"
    "loma_matcher_R_bf16.onnx;"
    "172d214a91e5c12126acc7c7c66804c22ec47b9a8fc572b10ba9711714386e43";
inline const std::string kDefaultLomaLMatcherUri =
    "https://github.com/davnords/storage/releases/download/loma/"
    "loma_matcher_L.onnx;"
    "loma_matcher_L.onnx;"
    "918da5427f37c3d9e05264a58a4d05729f87d459c3d5d69add7ce72eeacea66d";
inline const std::string kDefaultLomaLMatcherBf16Uri =
    "https://github.com/davnords/storage/releases/download/loma/"
    "loma_matcher_L_bf16.onnx;"
    "loma_matcher_L_bf16.onnx;"
    "bab859300bbec88af30705021356f526a374d48ac7917b135d979686ade0ed23";
inline const std::string kDefaultLomaGMatcherUri =
    "https://github.com/davnords/storage/releases/download/loma/"
    "loma_matcher_G.onnx;"
    "loma_matcher_G.onnx;"
    "62134cde779ed1031982c13c2c1952ed14856d715e97072364821a91f6276cf9";
inline const std::string kDefaultLomaGMatcherBf16Uri =
    "https://github.com/davnords/storage/releases/download/loma/"
    "loma_matcher_G_bf16.onnx;"
    "loma_matcher_G_bf16.onnx;"
    "2d21420e1e0e3bcd085cd9a7d33be1f990e820d6b13ffbbddc73d9d04713b6c6";
#else
inline const std::string kDefaultAlikedN16RotFeatureExtractorUri = "";
inline const std::string kDefaultAlikedN32FeatureExtractorUri = "";
inline const std::string kDefaultAlikedLightGlueFeatureMatcherUri = "";
inline const std::string kDefaultBruteForceONNXMatcherUri = "";
inline const std::string kDefaultSiftLightGlueFeatureMatcherUri = "";
inline const std::string kDefaultLomaBDetectorUri = "";
inline const std::string kDefaultLomaBDescriptorUri = "";
inline const std::string kDefaultLomaBMatcherUri = "";
inline const std::string kDefaultLomaBDescriptorBf16Uri = "";
inline const std::string kDefaultLomaB128DescriptorUri = "";
inline const std::string kDefaultLomaBMatcherBf16Uri = "";
inline const std::string kDefaultLomaB128MatcherUri = "";
inline const std::string kDefaultLomaB128MatcherBf16Uri = "";
inline const std::string kDefaultLomaRMatcherUri = "";
inline const std::string kDefaultLomaRMatcherBf16Uri = "";
inline const std::string kDefaultLomaLMatcherUri = "";
inline const std::string kDefaultLomaLMatcherBf16Uri = "";
inline const std::string kDefaultLomaGMatcherUri = "";
inline const std::string kDefaultLomaGMatcherBf16Uri = "";
#endif

}  // namespace colmap
