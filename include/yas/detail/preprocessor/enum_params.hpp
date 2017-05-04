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
#ifndef __yas__detail__preprocessor__enum_params_hpp
#define __yas__detail__preprocessor__enum_params_hpp
#
# include <yas/detail/preprocessor/config.hpp>
# include <yas/detail/preprocessor/comma_if.hpp>
# include <yas/detail/preprocessor/repeat.hpp>
#
# /* YAS_PP_ENUM_PARAMS */
#
# if ~YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_EDG()
#    define YAS_PP_ENUM_PARAMS(count, param) YAS_PP_REPEAT(count, YAS_PP_ENUM_PARAMS_M, param)
# else
#    define YAS_PP_ENUM_PARAMS(count, param) YAS_PP_ENUM_PARAMS_I(count, param)
#    define YAS_PP_ENUM_PARAMS_I(count, param) YAS_PP_REPEAT(count, YAS_PP_ENUM_PARAMS_M, param)
# endif
#
# define YAS_PP_ENUM_PARAMS_M(z, n, param) YAS_PP_COMMA_IF(n) param ## n
#
# /* YAS_PP_ENUM_PARAMS_Z */
#
# if ~YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_EDG()
#    define YAS_PP_ENUM_PARAMS_Z(z, count, param) YAS_PP_REPEAT_ ## z(count, YAS_PP_ENUM_PARAMS_M, param)
# else
#    define YAS_PP_ENUM_PARAMS_Z(z, count, param) YAS_PP_ENUM_PARAMS_Z_I(z, count, param)
#    define YAS_PP_ENUM_PARAMS_Z_I(z, count, param) YAS_PP_REPEAT_ ## z(count, YAS_PP_ENUM_PARAMS_M, param)
# endif
#
#endif // __yas__detail__preprocessor__enum_params_hpp
