
// Copyright (c) 2010-2017 niXman (i dot nixman dog gmail dot com). All
// rights reserved.
//
// This file is part of YAS(https://github.com/niXman/yas) project.
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//
//
// Boost Software License - Version 1.0 - August 17th, 2003
//
// Permission is hereby granted, free of charge, to any person or organization
// obtaining a copy of the software and accompanying documentation covered by
// this license (the "Software") to use, reproduce, display, distribute,
// execute, and transmit the Software, and to prepare derivative works of the
// Software, and to permit third-parties to whom the Software is furnished to
// do so, all subject to the following:
//
// The copyright notices in the Software and this entire statement, including
// the above license grant, this restriction and the following disclaimer,
// must be included in all copies of the Software, in whole or in part, and
// all derivative works of the Software, unless such copies or derivative
// works are solely in the form of machine-executable object code generated by
// a source language processor.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
// SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
// FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#ifndef __yas__types__std__std_optional_serializers_hpp
#define __yas__types__std__std_optional_serializers_hpp

#if __cplusplus > 201103L

#include <yas/detail/type_traits/type_traits.hpp>
#include <yas/detail/type_traits/serializer.hpp>

#ifdef __has_include
#	if __has_include(<optional>)
#		include <optional>
#		define _YAS_HAVE_STD_OPTIONAL 1
#	elif __has_include(<experimental/optional>)
#		include <experimental/optional>
#		define _YAS_HAVE_STD_OPTIONAL 1
#		define _YAS_HAVE_STD_EXPERIMENTAL_OPTIONAL
#	else
#		define _YAS_HAVE_STD_OPTIONAL 0
#	endif
#endif

namespace yas {
namespace detail {

#if _YAS_HAVE_STD_OPTIONAL

#ifdef _YAS_HAVE_STD_EXPERIMENTAL_OPTIONAL
#	define _YAS_STD_OPTIONAL_NS std::experimental
#else
#	define _YAS_STD_OPTIONAL_NS std
#endif // _YAS_HAVE_STD_EXPERIMENTAL_OPTIONAL

/***************************************************************************/

template<std::size_t F, typename T>
struct serializer<
	type_prop::not_a_fundamental,
	ser_method::use_internal_serializer,
	F,
	_YAS_STD_OPTIONAL_NS::optional<T>
> {
	template<typename Archive>
	static Archive& save(Archive& ar, const _YAS_STD_OPTIONAL_NS::optional<T> &t) {
		const bool initialized = static_cast<bool>(t);
		ar.write(initialized);
		if ( initialized )
			ar & t.value();

		return ar;
	}

	template<typename Archive>
	static Archive& load(Archive& ar, _YAS_STD_OPTIONAL_NS::optional<T> &t) {
		bool initialized = false;
		ar.read(initialized);
		if ( initialized ) {
			T val{};
			ar & val;
			t = std::move(val);
		} else {
			t = _YAS_STD_OPTIONAL_NS::optional<T>();
		}

		return ar;
	}
};

/***************************************************************************/

#endif // _YAS_HAVE_STD_OPTIONAL

#undef _YAS_HAVE_STD_OPTIONAL
#undef _YAS_HAVE_STD_EXPERIMENTAL_OPTIONAL
#undef _YAS_STD_OPTIONAL_NS

} // namespace detail
} // namespace yas

#endif // __cplusplus > 201103L

#endif // __yas__types__std__std_optional_serializers_hpp
