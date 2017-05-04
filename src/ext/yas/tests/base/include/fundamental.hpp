
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

#ifndef __yas__tests__base__include__pod_hpp
#define __yas__tests__base__include__pod_hpp

/***************************************************************************/

template<typename archive_traits>
bool fundamental_test(std::ostream &log, const char *archive_type) {
    typename archive_traits::oarchive oa;
    archive_traits::ocreate(oa, archive_type);

    bool b = true, bb;
    std::int8_t c = '1', cc;
    std::uint8_t uc = '2', uc2;
    std::int16_t s = 3, ss;
    std::uint16_t us = 4, us2;
    std::int32_t i = 5, ii;
    std::uint32_t l = 6, ll;
    std::int64_t i64 = 7, ii64;
    std::uint64_t ui64 = 8, uui64;

    std::int64_t i64max = std::numeric_limits<std::int64_t>::max(), ii64max;
    std::uint64_t u64max = std::numeric_limits<std::uint64_t>::max(), iu64max;

    float f = 3.14f, ff;
    double d = 3.14, dd;

    enum {
        binary_expected_size =
             sizeof(b)
            +sizeof(c)
            +sizeof(uc)
            +sizeof(s)
            +sizeof(us)
            +sizeof(i)
            +sizeof(l)
            +sizeof(i64)
            +sizeof(ui64)
            +sizeof(f)
            +sizeof(d)
            +sizeof(i64max)
            +sizeof(u64max)
        ,binary_compacted_expected_size = 45
        ,text_expected_size = 78
        ,object_expected_size = 28
    };

    oa & YAS_OBJECT("pod", b, c, uc, s, us, i, l, i64, ui64, f, d, i64max, u64max);

    switch (archive_traits::oarchive_type::type()) {
        case yas::binary: {
            std::size_t exp_size = archive_traits::oarchive_type::header_size() + binary_expected_size;
            std::size_t comp_exp_size = archive_traits::oarchive_type::header_size() + binary_compacted_expected_size;
            if (oa.size() != (archive_traits::oarchive_type::compacted() ? comp_exp_size : exp_size)) {
                YAS_TEST_REPORT(log, "POD serialization error!");
                return false;
            }
        } break;
        case yas::text: {
            if (oa.size() != archive_traits::oarchive_type::header_size() + text_expected_size) {
                YAS_TEST_REPORT(log, "POD serialization error!");
                return false;
            }
        } break;
        case yas::object: {
            if (oa.size() != archive_traits::oarchive_type::header_size() + object_expected_size) {
                YAS_TEST_REPORT(log, "POD serialization error!");
                return false;
            }
        } break;
        default:
            YAS_TEST_REPORT(log, "POD serialization bad archive type!");
            return false;
    }

    typename archive_traits::iarchive ia;
    archive_traits::icreate(ia, oa, archive_type);
    auto o = YAS_OBJECT("pod", bb, cc, uc2, ss, us2, ii, ll, ii64, uui64, ff, dd, ii64max, iu64max);
    ia & o;

    if (b != bb || c != cc || uc != uc2 || s != ss || us != us2 || i != ii
        || l != ll || i64 != ii64 || ui64 != uui64 || f != ff || d != dd || i64max != ii64max || u64max != iu64max)
    {
        YAS_TEST_REPORT(log, "POD deserialization error!");
        return false;
    }

    return true;
}

/***************************************************************************/

#endif // __yas__tests__base__include__pod_hpp
