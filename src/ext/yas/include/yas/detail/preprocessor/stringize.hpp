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
#ifndef __yas__detail__preprocessor__stringize_hpp
#define __yas__detail__preprocessor__stringize_hpp
#
# include <yas/detail/preprocessor/config.hpp>
#
# /* YAS_PP_STRINGIZE */
#
# if YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_MSVC()
#    define YAS_PP_STRINGIZE(text) YAS_PP_STRINGIZE_A((text))
#    define YAS_PP_STRINGIZE_A(arg) YAS_PP_STRINGIZE_I arg
# elif YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_MWCC()
#    define YAS_PP_STRINGIZE(text) YAS_PP_STRINGIZE_OO((text))
#    define YAS_PP_STRINGIZE_OO(par) YAS_PP_STRINGIZE_I ## par
# else
#    define YAS_PP_STRINGIZE(text) YAS_PP_STRINGIZE_I(text)
# endif
#
# define YAS_PP_STRINGIZE_I(text) #text
#
#endif // __yas__detail__preprocessor__stringize_hpp
