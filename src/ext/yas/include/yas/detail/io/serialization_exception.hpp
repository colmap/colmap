
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

#ifndef __yas__detail__io__serialization_exception_hpp
#define __yas__detail__io__serialization_exception_hpp

#include <yas/detail/io/exception_base.hpp>

namespace yas {

/***************************************************************************/

YAS_DECLARE_EXCEPTION_TYPE(serialization_exception)

/***************************************************************************/

#define YAS_THROW_BAD_ARRAY_SIZE() \
	YAS_THROW_EXCEPTION(serialization_exception, "bad array size");

#define YAS_THROW_SPACE_IS_EXPECTED() \
	YAS_THROW_EXCEPTION(serialization_exception, "space symbol is expected");

#define YAS_THROW_BAD_SIZE_OF_ENUM() \
	YAS_THROW_EXCEPTION(serialization_exception, "bad size of enum");

#define YAS_THROW_BAD_BITSET_SIZE() \
	YAS_THROW_EXCEPTION(serialization_exception, "bad bitset size");

#define YAS_THROW_BAD_BITSET_STORAGE_SIZE() \
	YAS_THROW_EXCEPTION(serialization_exception, "bad bitset storage size");

#define YAS_THROW_BAD_SIZE_ON_DESERIALIZE(type) \
	YAS_THROW_EXCEPTION(serialization_exception, "bad size on deserialize " type);

/***************************************************************************/

} // ns yas

#endif // __yas__detail__io__serialization_exception_hpp
