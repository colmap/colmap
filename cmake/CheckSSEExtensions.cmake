include(CheckCXXSourceRuns)

################################################################################
# SSE
################################################################################

if(IS_GNU OR IS_CLANG)
    set(CMAKE_REQUIRED_FLAGS "-msse")
endif()

CHECK_CXX_SOURCE_RUNS("
        #include <xmmintrin.h>
        int main() {
            __m128 a, b;
            float vals[4] = {0};
            a = _mm_loadu_ps(vals);
            b = a;
            b = _mm_add_ps(a, b);
            _mm_storeu_ps(vals, b);
            return 0;
        }"
        HAS_SSE_EXTENSION)

################################################################################
# SSE 2
################################################################################

if(IS_GNU OR IS_CLANG)
    set(CMAKE_REQUIRED_FLAGS "-msse2")
endif()

CHECK_CXX_SOURCE_RUNS("
        #include <emmintrin.h>
        int main() {
            __m128d a, b;
            double vals[2] = {0};
            a = _mm_loadu_pd(vals);
            b = _mm_add_pd(a, a);
            _mm_storeu_pd(vals, b);
            return 0;
        }"
        HAS_SSE2_EXTENSION)

################################################################################
# SSE 3
################################################################################

if(IS_GNU OR IS_CLANG)
    set(CMAKE_REQUIRED_FLAGS "-msse3")
endif()

CHECK_CXX_SOURCE_RUNS("
        #include <pmmintrin.h>
        int main() {
            __m128d a, b;
            double vals[2] = {0};
            a = _mm_loadu_pd(vals);
            b = _mm_hadd_pd(a, a);
            _mm_storeu_pd(vals, b);
            return 0;
        }"
        HAS_SSE3_EXTENSION)

################################################################################
# SSE 4.1
################################################################################

if(IS_GNU OR IS_CLANG)
    set(CMAKE_REQUIRED_FLAGS "-msse4.1")
endif()

CHECK_CXX_SOURCE_RUNS("
        #include <smmintrin.h>
        int main() {
          __m128 a, b;
          float vals[4] = {1, 2, 3, 4};
          const int mask = 123;
          a = _mm_loadu_ps(vals);
          b = a;
          b = _mm_dp_ps(a, a, mask);
          _mm_storeu_ps(vals, b);
          return 0;
        }"
        HAS_SSE41_EXTENSION)

################################################################################
# SSE 4.2
################################################################################

if(IS_GNU OR IS_CLANG)
    set(CMAKE_REQUIRED_FLAGS "-msse4.2")
endif()

CHECK_CXX_SOURCE_RUNS("
        #include <emmintrin.h>
        #include <nmmintrin.h>
        int main() {
          long long a[2] = { 1, 2};
          long long b[2] = {-1, 3};
          long long c[2];
          __m128i va = _mm_loadu_si128((__m128i*)a);
          __m128i vb = _mm_loadu_si128((__m128i*)b);
          __m128i vc = _mm_cmpgt_epi64(va, vb);
          _mm_storeu_si128((__m128i*)c, vc);
          if (c[0] == -1LL && c[1] == 0LL)
            return 0;
          else
            return 1;
        }"
        HAS_SSE42_EXTENSION)

################################################################################
# AVX
################################################################################

if(IS_GNU OR IS_CLANG)
    set(CMAKE_REQUIRED_FLAGS "-mavx")
endif()

CHECK_CXX_SOURCE_RUNS("
        #include <immintrin.h>
        #include <stdio.h>
        int main() {
          __m256 a = _mm256_set_ps(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);
          __m256 b = _mm256_set_ps(1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0);
          __m256 c = _mm256_sub_ps(a, b);
          float* f = (float*)&c;
          return 0;
        }"
        HAS_AVX_EXTENSION)

################################################################################
# Setup the compile flags
################################################################################

# Clear the set flags from above.
set(CMAKE_REQUIRED_FLAGS)

# Set the compile flags for the supported extensions.
set(SSE_FLAGS)

if(IS_GNU OR IS_CLANG)
    if(HAS_SSE_EXTENSION)
        set(SSE_FLAGS "${SSE_FLAGS} -msse")
    endif()

    if(HAS_SSE2_EXTENSION)
        set(SSE_FLAGS "${SSE_FLAGS} -msse2")
    endif()

    if(HAS_SSE3_EXTENSION)
        set(SSE_FLAGS "${SSE_FLAGS} -msse3")
    endif()

    if(HAS_SSE41_EXTENSION)
        set(SSE_FLAGS "${SSE_FLAGS} -msse4.1")
    endif()

    if(HAS_SSE41_EXTENSION)
        set(SSE_FLAGS "${SSE_FLAGS} -msse4.2")
    endif()

    if(HAS_AVX_EXTENSION)
        set(SSE_FLAGS "${SSE_FLAGS} -mavx")
    endif()
endif()
