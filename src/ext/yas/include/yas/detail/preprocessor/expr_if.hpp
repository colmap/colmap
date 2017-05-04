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
#ifndef __yas__detail__preprocessor__expr_if_hpp
#define __yas__detail__preprocessor__expr_if_hpp
#
# include <yas/detail/preprocessor/config.hpp>
# include <yas/detail/preprocessor/expr_iif.hpp>
# include <yas/detail/preprocessor/bool.hpp>
#
# /* YAS_PP_EXPR_IF */
#
# if ~YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_EDG()
#    define YAS_PP_EXPR_IF(cond, expr) YAS_PP_EXPR_IIF(YAS_PP_BOOL(cond), expr)
# else
#    define YAS_PP_EXPR_IF(cond, expr) YAS_PP_EXPR_IF_I(cond, expr)
#    define YAS_PP_EXPR_IF_I(cond, expr) YAS_PP_EXPR_IIF(YAS_PP_BOOL(cond), expr)
# endif
#
#endif // __yas__detail__preprocessor__expr_if_hpp
