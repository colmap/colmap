# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


set(FOLDER_NAME "retrieval")

COLMAP_ADD_LIBRARY(
    NAME colmap_retrieval
    SRCS
        geometry.h geometry.cc
        inverted_file.h
        inverted_file_entry.h
        inverted_index.h
        resources.h
        utils.h
        visual_index.h visual_index.cc
        vote_and_verify.h vote_and_verify.cc
    PUBLIC_LINK_LIBS
        Boost::boost
        Eigen3::Eigen
    PRIVATE_LINK_LIBS
        colmap_math
        colmap_estimators
        colmap_optim
        faiss
)
if(OPENMP_FOUND)
    target_link_libraries(colmap_retrieval PUBLIC OpenMP::OpenMP_CXX)
endif()

COLMAP_ADD_TEST(
    NAME geometry_test
    SRCS geometry_test.cc
    LINK_LIBS colmap_retrieval
)
COLMAP_ADD_TEST(
    NAME inverted_file_entry_test
    SRCS inverted_file_entry_test.cc
    LINK_LIBS colmap_retrieval
)
COLMAP_ADD_TEST(
    NAME visual_index_test
    SRCS visual_index_test.cc
    LINK_LIBS colmap_retrieval
)
COLMAP_ADD_TEST(
    NAME vote_and_verify_test
    SRCS vote_and_verify_test.cc
    LINK_LIBS colmap_retrieval
)
