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
#ifndef __yas__detail__preprocessor__if_hpp
#define __yas__detail__preprocessor__if_hpp
#
# include <yas/detail/preprocessor/config.hpp>
# include <yas/detail/preprocessor/iif.hpp>
# include <yas/detail/preprocessor/bool.hpp>
#
# /* YAS_PP_IF */
#
# if ~YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_EDG()
#    define YAS_PP_IF(cond, t, f) YAS_PP_IIF(YAS_PP_BOOL(cond), t, f)
# else
#    define YAS_PP_IF(cond, t, f) YAS_PP_IF_I(cond, t, f)
#    define YAS_PP_IF_I(cond, t, f) YAS_PP_IIF(YAS_PP_BOOL(cond), t, f)
# endif
#
#endif // __yas__detail__preprocessor__if_hpp
