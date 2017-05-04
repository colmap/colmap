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
#ifndef __yas__detail__preprocessor__compl_hpp
#define __yas__detail__preprocessor__compl_hpp
#
# include <yas/detail/preprocessor/config.hpp>
#
# /* YAS_PP_COMPL */
#
# if ~YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_MWCC()
#    define YAS_PP_COMPL(x) YAS_PP_COMPL_I(x)
# else
#    define YAS_PP_COMPL(x) YAS_PP_COMPL_OO((x))
#    define YAS_PP_COMPL_OO(par) YAS_PP_COMPL_I ## par
# endif
#
# if ~YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_MSVC()
#    define YAS_PP_COMPL_I(x) YAS_PP_COMPL_ ## x
# else
#    define YAS_PP_COMPL_I(x) YAS_PP_COMPL_ID(YAS_PP_COMPL_ ## x)
#    define YAS_PP_COMPL_ID(id) id
# endif
#
# define YAS_PP_COMPL_0 1
# define YAS_PP_COMPL_1 0
#
#endif // __yas__detail__preprocessor__compl_hpp
