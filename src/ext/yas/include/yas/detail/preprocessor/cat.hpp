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
#ifndef __yas__detail__preprocessor__cat_hpp
#define __yas__detail__preprocessor__cat_hpp
#
# include <yas/detail/preprocessor/config.hpp>
#
# /* YAS_PP_CAT */
#
# if ~YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_MWCC()
#    define YAS_PP_CAT(a, b) YAS_PP_CAT_I(a, b)
# else
#    define YAS_PP_CAT(a, b) YAS_PP_CAT_OO((a, b))
#    define YAS_PP_CAT_OO(par) YAS_PP_CAT_I ## par
# endif
#
# if ~YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_MSVC()
#    define YAS_PP_CAT_I(a, b) a ## b
# else
#    define YAS_PP_CAT_I(a, b) YAS_PP_CAT_II(~, a ## b)
#    define YAS_PP_CAT_II(p, res) res
# endif
#
#endif // __yas__detail__preprocessor__cat_hpp
