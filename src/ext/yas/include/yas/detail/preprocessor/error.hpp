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
#ifndef __yas__detail__preprocessor__error_hpp
#define __yas__detail__preprocessor__error_hpp
#
# include <yas/detail/preprocessor/cat.hpp>
# include <yas/detail/preprocessor/config.hpp>
#
# /* YAS_PP_ERROR */
#
# if YAS_PP_CONFIG_ERRORS
#    define YAS_PP_ERROR(code) YAS_PP_CAT(YAS_PP_ERROR_, code)
# endif
#
# define YAS_PP_ERROR_0x0000 YAS_PP_ERROR(0x0000, YAS_PP_INDEX_OUT_OF_BOUNDS)
# define YAS_PP_ERROR_0x0001 YAS_PP_ERROR(0x0001, YAS_PP_WHILE_OVERFLOW)
# define YAS_PP_ERROR_0x0002 YAS_PP_ERROR(0x0002, YAS_PP_FOR_OVERFLOW)
# define YAS_PP_ERROR_0x0003 YAS_PP_ERROR(0x0003, YAS_PP_REPEAT_OVERFLOW)
# define YAS_PP_ERROR_0x0004 YAS_PP_ERROR(0x0004, YAS_PP_LIST_FOLD_OVERFLOW)
# define YAS_PP_ERROR_0x0005 YAS_PP_ERROR(0x0005, YAS_PP_SEQ_FOLD_OVERFLOW)
# define YAS_PP_ERROR_0x0006 YAS_PP_ERROR(0x0006, YAS_PP_ARITHMETIC_OVERFLOW)
# define YAS_PP_ERROR_0x0007 YAS_PP_ERROR(0x0007, YAS_PP_DIVISION_BY_ZERO)
#
#endif // __yas__detail__preprocessor__error_hpp
