# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Edward Diener 2011.                                    *
#  *     (C) Copyright Paul Mensonides 2011.                                  *
#  *     Distributed under the Boost Software License, Version 1.0. (See      *
#  *     accompanying file LICENSE_1_0.txt or copy at                         *
#  *     http://www.boost.org/LICENSE_1_0.txt)                                *
#  *                                                                          *
#  ************************************************************************** */
#
# /* See http://www.boost.org for most recent version. */
#
#ifndef __yas__detail__preprocessor__tuple_size_hpp
#define __yas__detail__preprocessor__tuple_size_hpp
#
# include <yas/detail/preprocessor/cat.hpp>
# include <yas/detail/preprocessor/config.hpp>
# include <yas/detail/preprocessor/variadic_size.hpp>
#
# if YAS_PP_VARIADICS
#    if YAS_PP_VARIADICS_MSVC
#        define YAS_PP_TUPLE_SIZE(tuple) YAS_PP_CAT(YAS_PP_VARIADIC_SIZE tuple,)
#    else
#        define YAS_PP_TUPLE_SIZE(tuple) YAS_PP_VARIADIC_SIZE tuple
#    endif
# endif
#
#endif // __yas__detail__preprocessor__tuple_size_hpp
