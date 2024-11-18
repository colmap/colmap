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

#pragma once

#include <iostream>
#include <vector>
#include <boost/preprocessor.hpp>

namespace colmap {

#define ENUM_TO_STRING_PROCESS_ELEMENT(r, unused, idx, elem) \
    BOOST_PP_COMMA_IF(idx) BOOST_PP_STRINGIZE(elem)
   
#define DEFINE_ENUM_TO_STRING(name, ...)\
    const std::vector<std::string> name##Strings = { BOOST_PP_SEQ_FOR_EACH_I(ENUM_TO_STRING_PROCESS_ELEMENT, %%, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))  };\
    template<typename T>\
    constexpr const std::string& name##ToString(T value) { return name##Strings[static_cast<int>(value)]; }

#define MAGIC_MAKE_ENUM(name, ...)\
    enum class name { __VA_ARGS__  };\
    DEFINE_ENUM_TO_STRING(name, __VA_ARGS__)

// Note: this only works for non-nested enum classes.
#define MAGIC_MAKE_ENUM_OVERLOAD_STREAM(name, ...)\
    MAGIC_MAKE_ENUM(name, __VA_ARGS__);\
    inline std::ostream& operator<<(std::ostream& os, name value) {\
      return os << name##ToString(static_cast<int>(value));\
    }

}  // namespace colmap
