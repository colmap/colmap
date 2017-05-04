
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

#ifndef __yas__types__utility__object_hpp
#define __yas__types__utility__object_hpp

#include <yas/detail/preprocessor/preprocessor.hpp>

#include <yas/detail/tools/cast.hpp>

#include <tuple>
#include <type_traits>
#include <cstring>

namespace yas {

/***************************************************************************/

template<typename T>
struct value {
	using type = typename std::conditional<
		 std::is_array<typename std::remove_reference<T>::type>::value
		,typename std::remove_cv<T>::type
		,typename std::conditional<
			 std::is_lvalue_reference<T>::value
			,T
			,typename std::decay<T>::type
		>::type
	>::type;

	value(const value &) = delete;
	value& operator=(const value &) = delete;

	constexpr value(const char *k, std::size_t klen, T &&v)
		:key(k)
		,klen(klen)
		,val(std::forward<T>(v))
	{}
	constexpr value(value &&r)
		:key(r.key)
		,klen(r.klen)
		,val(std::forward<type>(r.val))
	{}

	const char *key;
	const std::size_t klen;
	type val;
};

template<std::size_t N, typename T>
constexpr value<T> make_val(const char (&key)[N], T &&val) {
	return {key, N-1, std::forward<T>(val)};
}

template<
	 typename ConstCharPtr
	,typename T
>
constexpr typename std::enable_if<
     std::is_same<ConstCharPtr, const char*>::value
    ,value<T>
>::type
make_val(ConstCharPtr key, T &&val) {
	return {key, std::strlen(key), std::forward<T>(val)};
}

/***************************************************************************/

template<typename... Pairs>
struct object_t {
	using type = std::tuple<Pairs...>;

	object_t(const object_t &) = delete;
	object_t& operator=(const object_t &) = delete;

	constexpr object_t(const char *k, std::size_t klen, Pairs&&... pairs)
		:key(k)
		,klen(klen)
		,pairs(std::forward<Pairs>(pairs)...)
	{}
	constexpr object_t(object_t &&r)
		:key(r.key)
		,klen(r.klen)
		,pairs(std::move(r.pairs))
	{}

	const char *key;
	const std::size_t klen;
	type pairs;
};

template<std::size_t N, typename... Pairs>
constexpr object_t<Pairs...>
make_object(const char (&key)[N], Pairs &&... pairs) {
	return {key, N-1, std::forward<Pairs>(pairs)...};
}

template<
	 typename ConstCharPtr
	,typename... Pairs
>
constexpr typename std::enable_if<
     std::is_same<ConstCharPtr, const char*>::value
    ,object_t<Pairs...>
>::type
make_object(ConstCharPtr key, Pairs &&... pairs) {
	return {key, std::strlen(key), std::forward<Pairs>(pairs)...};
}

/**************************************************************************/

#define _YAS_ARG16(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, ...) _15
#define _YAS_HAS_COMMA(...) _YAS_ARG16(__VA_ARGS__, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0)
#define _YAS__TRIGGER_PARENTHESIS_(...) ,

#define _YAS_ISEMPTY(...) \
	_YAS__ISEMPTY( \
		_YAS_HAS_COMMA(__VA_ARGS__), \
		_YAS_HAS_COMMA(_YAS__TRIGGER_PARENTHESIS_ __VA_ARGS__),                 \
		_YAS_HAS_COMMA(__VA_ARGS__ (/*empty*/)), \
		_YAS_HAS_COMMA(_YAS__TRIGGER_PARENTHESIS_ __VA_ARGS__ (/*empty*/)) \
	)

#define _YAS_PASTE5(_0, _1, _2, _3, _4) _0 ## _1 ## _2 ## _3 ## _4
#define _YAS__ISEMPTY(_0, _1, _2, _3) _YAS_HAS_COMMA(_YAS_PASTE5(_YAS__IS_EMPTY_CASE_, _0, _1, _2, _3))
#define _YAS__IS_EMPTY_CASE_0001 ,
#define _YAS_TUPLE_TO_ARGS(...) __VA_ARGS__
#define _YAS_TUPLE_IS_EMPTY(tuple) _YAS_ISEMPTY(_YAS_TUPLE_TO_ARGS tuple)

/**************************************************************************/

#define _YAS_OBJECT_GEN_PAIRS(unused, size, idx, elem) \
	::yas::make_val(YAS_PP_STRINGIZE(elem), elem) \
		YAS_PP_COMMA_IF(YAS_PP_NOT_EQUAL(YAS_PP_INC(idx), size))

#define _YAS_OBJECT_IMPL(seq) \
	YAS_PP_SEQ_FOR_EACH_I( \
		 _YAS_OBJECT_GEN_PAIRS \
		,YAS_PP_SEQ_SIZE(seq) \
		,seq \
	)

#define _YAS_OBJECT_EMPTY(name, ...) \
	::yas::make_object(name)

#define _YAS_OBJECT_NONEMPTY(name, ...) \
	::yas::make_object( \
		 name \
		,_YAS_OBJECT_IMPL(__VA_ARGS__) \
	)

#define YAS_OBJECT(name, ...) \
	YAS_PP_IF( \
		 _YAS_TUPLE_IS_EMPTY((__VA_ARGS__)) \
		,_YAS_OBJECT_EMPTY \
		,_YAS_OBJECT_NONEMPTY \
	)(name, YAS_PP_TUPLE_TO_SEQ((__VA_ARGS__)))

/***************************************************************************/

} // ns yas

#endif // __yas__types__utility__object_hpp
