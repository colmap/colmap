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
#ifndef __yas__detail__preprocessor__iif_hpp
#define __yas__detail__preprocessor__iif_hpp
#
# include <yas/detail/preprocessor/config.hpp>
#
# if ~YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_MWCC()
#    define YAS_PP_IIF(bit, t, f) YAS_PP_IIF_I(bit, t, f)
# else
#    define YAS_PP_IIF(bit, t, f) YAS_PP_IIF_OO((bit, t, f))
#    define YAS_PP_IIF_OO(par) YAS_PP_IIF_I ## par
# endif
#
# if ~YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_MSVC()
#    define YAS_PP_IIF_I(bit, t, f) YAS_PP_IIF_ ## bit(t, f)
# else
#    define YAS_PP_IIF_I(bit, t, f) YAS_PP_IIF_II(YAS_PP_IIF_ ## bit(t, f))
#    define YAS_PP_IIF_II(id) id
# endif
#
# define YAS_PP_IIF_0(t, f) f
# define YAS_PP_IIF_1(t, f) t
#
#endif // __yas__detail__preprocessor__iif_hpp
