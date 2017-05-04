# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Paul Mensonides 2002.
#  *     Distributed under the Boost Software License, Version 1.0. (See
#  *     accompanying file LICENSE_1_0.txt or copy at
#  *     http://www.boost.org/LICENSE_1_0.txt)
#  *                                                                          *
#  ************************************************************************** */
#
# /* See http://www.boost.org for most recent version. */
#
#ifndef __yas__detail__preprocessor__expr_iif_hpp
#define __yas__detail__preprocessor__expr_iif_hpp
#
# include <yas/detail/preprocessor/config.hpp>
#
# /* YAS_PP_EXPR_IIF */
#
# if ~YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_MWCC()
#    define YAS_PP_EXPR_IIF(bit, expr) YAS_PP_EXPR_IIF_I(bit, expr)
# else
#    define YAS_PP_EXPR_IIF(bit, expr) YAS_PP_EXPR_IIF_OO((bit, expr))
#    define YAS_PP_EXPR_IIF_OO(par) YAS_PP_EXPR_IIF_I ## par
# endif
#
# define YAS_PP_EXPR_IIF_I(bit, expr) YAS_PP_EXPR_IIF_ ## bit(expr)
#
# define YAS_PP_EXPR_IIF_0(expr)
# define YAS_PP_EXPR_IIF_1(expr) expr
#
#endif // __yas__detail__preprocessor__expr_iif_hpp
