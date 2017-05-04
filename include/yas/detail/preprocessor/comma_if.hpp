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
#ifndef __yas__detail__preprocessor__comma_if_hpp
#define __yas__detail__preprocessor__comma_if_hpp
#
# include <yas/detail/preprocessor/config.hpp>
# include <yas/detail/preprocessor/if.hpp>
# include <yas/detail/preprocessor/empty.hpp>
# include <yas/detail/preprocessor/comma.hpp>
#
# /* YAS_PP_COMMA_IF */
#
# if ~YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_EDG()
#    define YAS_PP_COMMA_IF(cond) YAS_PP_IF(cond, YAS_PP_COMMA, YAS_PP_EMPTY)()
# else
#    define YAS_PP_COMMA_IF(cond) YAS_PP_COMMA_IF_I(cond)
#    define YAS_PP_COMMA_IF_I(cond) YAS_PP_IF(cond, YAS_PP_COMMA, YAS_PP_EMPTY)()
# endif
#
#endif // __yas__detail__preprocessor__comma_if_hpp
