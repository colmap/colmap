
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

#ifndef __yas__tests__base__include__unordered_set_hpp
#define __yas__tests__base__include__unordered_set_hpp

#include <algorithm>
#include <numeric>

/***************************************************************************/

template<typename archive_traits>
bool unordered_set_test(std::ostream &log, const char* archive_type) {
	std::unordered_set<int> set1, set2;
	set1.insert(0);
	set1.insert(1);
	set1.insert(2);
	set1.insert(3);
	set1.insert(5);
	set1.insert(5);
	set1.insert(6);
	set1.insert(7);
	set1.insert(8);
	set1.insert(8);
	int expected1 = std::accumulate(set1.begin(), set1.end(), 0);

	typename archive_traits::oarchive oa;
	archive_traits::ocreate(oa, archive_type);
	oa & YAS_OBJECT("set1", set1);

	typename archive_traits::iarchive ia;
	archive_traits::icreate(ia, oa, archive_type);
	ia & YAS_OBJECT("set2", set2);

	if ( expected1 != std::accumulate(set2.begin(), set2.end(), 0) ) {
		YAS_TEST_REPORT(log, "UNORDERED_SET deserialization error!");
		return false;
	}

	std::unordered_set<std::string> set3, set4;
	set3.insert(std::string("1"));
	set3.insert(std::string("2"));
	set3.insert(std::string("3"));
	set3.insert(std::string("4"));
	set3.insert(std::string("5"));
	set3.insert(std::string("4"));

	std::string expected2 = std::accumulate(set3.begin(), set3.end(), std::string());
	std::sort(expected2.begin(), expected2.end());

	typename archive_traits::oarchive oa2;
	archive_traits::ocreate(oa2, archive_type);
	oa2 & YAS_OBJECT("set3", set3);

	typename archive_traits::iarchive ia2;
	archive_traits::icreate(ia2, oa2, archive_type);
	ia2 & YAS_OBJECT("set4", set4);

	std::string res2 = std::accumulate(set4.begin(), set4.end(), std::string());
	std::sort(res2.begin(), res2.end());
	if ( expected2 != res2 ) {
		YAS_TEST_REPORT(log, "UNORDERED_SET deserialization error!");
		return false;
	}

#if defined(YAS_SERIALIZE_BOOST_TYPES)
	boost::unordered_set<int> set5, set6;
	set5.insert(0);
	set5.insert(1);
	set5.insert(2);
	set5.insert(3);
	set5.insert(3);
	set5.insert(5);
	set5.insert(6);
	set5.insert(7);
	set5.insert(5);
	set5.insert(9);
	int expected3 = std::accumulate(set5.begin(), set5.end(), 0);

	typename archive_traits::oarchive oa3;
	archive_traits::ocreate(oa3, archive_type);
	oa3 & YAS_OBJECT("set5", set5);

	typename archive_traits::iarchive ia3;
	archive_traits::icreate(ia3, oa3, archive_type);
	ia3 & YAS_OBJECT("set6", set6);

	if ( expected3 != std::accumulate(set6.begin(), set6.end(), 0) ) {
	YAS_TEST_REPORT(log, "UNORDERED_SET deserialization error!");
		return false;
	}

	boost::unordered_set<std::string> set7, set8;
	set7.insert("1");
	set7.insert("1");
	set7.insert("2");
	set7.insert("5");
	set7.insert("2");
	set7.insert("5");
	std::string expected4 = std::accumulate(set7.begin(), set7.end(), std::string());
	std::sort(expected4.begin(), expected4.end());

	typename archive_traits::oarchive oa4;
	archive_traits::ocreate(oa4, archive_type);
	oa4 & YAS_OBJECT("set7", set7);

	typename archive_traits::iarchive ia4;
	archive_traits::icreate(ia4, oa4, archive_type);
	ia4 & YAS_OBJECT("set8", set8);

	std::string res4 = std::accumulate(set8.begin(), set8.end(), std::string());
	std::sort(res4.begin(), res4.end());
	if ( expected4 != res4 ) {
		YAS_TEST_REPORT(log, "UNORDERED_SET deserialization error!");
		return false;
	}
#endif // defined(YAS_SERIALIZE_BOOST_TYPES)
	return true;
}

/***************************************************************************/

#endif // __yas__tests__base__include__unordered_set_hpp
