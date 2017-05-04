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
#ifndef __yas__detail__preprocessor__bitand_hpp
#define __yas__detail__preprocessor__bitand_hpp
#
# include <yas/detail/preprocessor/config.hpp>
#
# /* YAS_PP_BITAND */
#
# if ~YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_MWCC()
#    define YAS_PP_BITAND(x, y) YAS_PP_BITAND_I(x, y)
# else
#    define YAS_PP_BITAND(x, y) YAS_PP_BITAND_OO((x, y))
#    define YAS_PP_BITAND_OO(par) YAS_PP_BITAND_I ## par
# endif
#
# if ~YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_MSVC()
#    define YAS_PP_BITAND_I(x, y) YAS_PP_BITAND_ ## x ## y
# else
#    define YAS_PP_BITAND_I(x, y) YAS_PP_BITAND_ID(YAS_PP_BITAND_ ## x ## y)
#    define YAS_PP_BITAND_ID(res) res
# endif
#
# define YAS_PP_BITAND_00 0
# define YAS_PP_BITAND_01 0
# define YAS_PP_BITAND_10 0
# define YAS_PP_BITAND_11 1
#
#endif // __yas__detail__preprocessor__bitand_hpp
