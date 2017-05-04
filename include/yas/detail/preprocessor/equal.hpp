# /* Copyright (C) 2001
#  * Housemarque Oy
#  * http://www.housemarque.com
#  *
#  * Distributed under the Boost Software License, Version 1.0. (See
#  * accompanying file LICENSE_1_0.txt or copy at
#  * http://www.boost.org/LICENSE_1_0.txt)
#  */
#
# /* Revised by Paul Mensonides (2002) */
#
# /* See http://www.boost.org for most recent version. */
#
#ifndef __yas__detail__preprocessor__equal_hpp
#define __yas__detail__preprocessor__equal_hpp
#
# include <yas/detail/preprocessor/not_equal.hpp>
# include <yas/detail/preprocessor/config.hpp>
# include <yas/detail/preprocessor/compl.hpp>
#
# /* YAS_PP_EQUAL */
#
# if ~YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_EDG()
#    define YAS_PP_EQUAL(x, y) YAS_PP_COMPL(YAS_PP_NOT_EQUAL(x, y))
# else
#    define YAS_PP_EQUAL(x, y) YAS_PP_EQUAL_I(x, y)
#    define YAS_PP_EQUAL_I(x, y) YAS_PP_COMPL(YAS_PP_NOT_EQUAL(x, y))
# endif
#
# /* YAS_PP_EQUAL_D */
#
# define YAS_PP_EQUAL_D(d, x, y) YAS_PP_EQUAL(x, y)
#
#endif // __yas__detail__preprocessor__equal_hpp
