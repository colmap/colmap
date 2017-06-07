
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

#ifndef __yas__tests__base__include__chrono_hpp
#define __yas__tests__base__include__chrono_hpp

/***************************************************************************/

template<typename archive_traits>
bool chrono_test(std::ostream &log, const char* archive_type) {
	{
		std::chrono::duration<int, std::ratio<1>> w0{32}, r0;
		std::chrono::duration<double, std::ratio<1>> w1{23}, r1;

		typename archive_traits::oarchive oa;
		archive_traits::ocreate(oa, archive_type);
		oa & YAS_OBJECT("o0", w0, w1);
		;

		typename archive_traits::iarchive ia;
		archive_traits::icreate(ia, oa, archive_type);
		ia & YAS_OBJECT("o1", r0, r1);
		;

		if ( r0 != w0 || r1 != w1 ) {
			YAS_TEST_REPORT(log, "CHRONO serialization error!");
			return false;
		}
	}
	{
		auto w0 = std::chrono::system_clock::now();
		auto r0 = std::chrono::system_clock::now();

		typename archive_traits::oarchive oa;
		archive_traits::ocreate(oa, archive_type);
		oa & YAS_OBJECT("w0", w0);

		typename archive_traits::iarchive ia;
		archive_traits::icreate(ia, oa, archive_type);
		ia & YAS_OBJECT("r0", r0);

		if ( r0 != w0 ) {
			YAS_TEST_REPORT(log, "CHRONO serialization error!");
			return false;
		}
	}
#if defined(YAS_SERIALIZE_BOOST_CHRONO)
	{
		boost::chrono::duration<int, boost::ratio<1>> w0{32}, r0;
		boost::chrono::duration<double, boost::ratio<1>> w1{23}, r1;

		typename archive_traits::oarchive oa;
		archive_traits::ocreate(oa, archive_type);
		oa & YAS_OBJECT("o2", w0, w1);
		;

		typename archive_traits::iarchive ia;
		archive_traits::icreate(ia, oa, archive_type);
		ia & YAS_OBJECT("o3", r0, r1);
		;

		if ( r0 != w0 || r1 != w1 ) {
			YAS_TEST_REPORT(log, "CHRONO serialization error!");
			return false;
		}
	}
	{
		decltype(boost::chrono::system_clock::now()) w0 = boost::chrono::system_clock::now(), r0;

		typename archive_traits::oarchive oa;
		archive_traits::ocreate(oa, archive_type);
		oa & YAS_OBJECT("w0", w0);

		typename archive_traits::iarchive ia;
		archive_traits::icreate(ia, oa, archive_type);
		ia & YAS_OBJECT("r0", r0);

		if ( r0 != w0 ) {
			YAS_TEST_REPORT(log, "CHRONO serialization error!");
			return false;
		}
	}
#endif // defined(YAS_SERIALIZE_BOOST_TYPES)

	return true;
}

/***************************************************************************/

#endif // __yas__tests__base__include__chrono_hpp
