
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

#ifndef __yas__tests__base__include__array_hpp
#define __yas__tests__base__include__array_hpp

/***************************************************************************/

template<typename archive_traits>
bool array_test(std::ostream &log, const char* archive_type) {
	std::array<int, 10> arr1 = {
		{0,1,2,3,4,5,6,7,8,9}
	}, arr2;

	typename archive_traits::oarchive oa;
	archive_traits::ocreate(oa, archive_type);
	oa & YAS_OBJECT("arr1", arr1);

	typename archive_traits::iarchive ia;
	archive_traits::icreate(ia, oa, archive_type);
	ia & YAS_OBJECT("arr2", arr2);

	if ( arr1 != arr2 ) {
        YAS_TEST_REPORT(log, "ARRAY deserialization error!");
		return false;
	}

	std::array<std::string, 10> arr3 = {
		{"0","1","2","3","4","5","6","7","8","9"}
	}, arr4;

	typename archive_traits::oarchive oa2;
	archive_traits::ocreate(oa2, archive_type);
	oa2 & YAS_OBJECT("arr3", arr3);

	typename archive_traits::iarchive ia2;
	archive_traits::icreate(ia2, oa2, archive_type);
	ia2 & YAS_OBJECT("arr4", arr4);

	if ( arr3 != arr4 ) {
        YAS_TEST_REPORT(log, "ARRAY deserialization error! [2]");
		return false;
	}

#if defined(YAS_SERIALIZE_BOOST_ARRAY)
	boost::array<int, 10> arr5 = {
		{0,1,2,3,4,5,6,7,8,9}
	}, arr6;

	typename archive_traits::oarchive oa3;
	archive_traits::ocreate(oa3, archive_type);
	oa3 & YAS_OBJECT("arr5", arr5);

	typename archive_traits::iarchive ia3;
	archive_traits::icreate(ia3, oa3, archive_type);
	ia3 & YAS_OBJECT("arr6", arr6);

	if ( arr5 != arr6 ) {
        YAS_TEST_REPORT(log, "ARRAY deserialization error! [3]");
		return false;
	}

	boost::array<std::string, 10> arr7 = {
		{"0","1","2","3","4","5","6","7","8","9"}
	}, arr8;

	typename archive_traits::oarchive oa4;
	archive_traits::ocreate(oa4, archive_type);
	oa4 & YAS_OBJECT("arr7", arr7);

	typename archive_traits::iarchive ia4;
	archive_traits::icreate(ia4, oa4, archive_type);
	ia4 & YAS_OBJECT("arr8", arr8);

	if ( arr7 != arr8 ) {
        YAS_TEST_REPORT(log, "ARRAY deserialization error! [4]");
		return false;
	}
#endif // defined(YAS_SERIALIZE_BOOST_TYPES)
	return true;
}

/***************************************************************************/

#endif // __yas__tests__base__include__array_hpp
