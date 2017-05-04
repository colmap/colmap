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
#ifndef __yas__detail__preprocessor__seq_for_each_hpp
#define __yas__detail__preprocessor__seq_for_each_hpp
#
# include <yas/detail/preprocessor/dec.hpp>
# include <yas/detail/preprocessor/config.hpp>
# include <yas/detail/preprocessor/rep_for.hpp>
# include <yas/detail/preprocessor/seq_seq.hpp>
# include <yas/detail/preprocessor/seq_size.hpp>
# include <yas/detail/preprocessor/tuple_elem.hpp>
# include <yas/detail/preprocessor/tuple_rem.hpp>
#
# /* YAS_PP_SEQ_FOR_EACH */
#
# if ~YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_EDG()
#    define YAS_PP_SEQ_FOR_EACH(macro, data, seq) YAS_PP_FOR((macro, data, seq (nil)), YAS_PP_SEQ_FOR_EACH_P, YAS_PP_SEQ_FOR_EACH_O, YAS_PP_SEQ_FOR_EACH_M)
# else
#    define YAS_PP_SEQ_FOR_EACH(macro, data, seq) YAS_PP_SEQ_FOR_EACH_D(macro, data, seq)
#    define YAS_PP_SEQ_FOR_EACH_D(macro, data, seq) YAS_PP_FOR((macro, data, seq (nil)), YAS_PP_SEQ_FOR_EACH_P, YAS_PP_SEQ_FOR_EACH_O, YAS_PP_SEQ_FOR_EACH_M)
# endif
#
# define YAS_PP_SEQ_FOR_EACH_P(r, x) YAS_PP_DEC(YAS_PP_SEQ_SIZE(YAS_PP_TUPLE_ELEM(3, 2, x)))
#
# if YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_STRICT()
#    define YAS_PP_SEQ_FOR_EACH_O(r, x) YAS_PP_SEQ_FOR_EACH_O_I x
# else
#    define YAS_PP_SEQ_FOR_EACH_O(r, x) YAS_PP_SEQ_FOR_EACH_O_I(YAS_PP_TUPLE_ELEM(3, 0, x), YAS_PP_TUPLE_ELEM(3, 1, x), YAS_PP_TUPLE_ELEM(3, 2, x))
# endif
#
# define YAS_PP_SEQ_FOR_EACH_O_I(macro, data, seq) (macro, data, YAS_PP_SEQ_TAIL(seq))
#
# if YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_STRICT()
#    define YAS_PP_SEQ_FOR_EACH_M(r, x) YAS_PP_SEQ_FOR_EACH_M_IM(r, YAS_PP_TUPLE_REM_3 x)
#    define YAS_PP_SEQ_FOR_EACH_M_IM(r, im) YAS_PP_SEQ_FOR_EACH_M_I(r, im)
# else
#    define YAS_PP_SEQ_FOR_EACH_M(r, x) YAS_PP_SEQ_FOR_EACH_M_I(r, YAS_PP_TUPLE_ELEM(3, 0, x), YAS_PP_TUPLE_ELEM(3, 1, x), YAS_PP_TUPLE_ELEM(3, 2, x))
# endif
#
# define YAS_PP_SEQ_FOR_EACH_M_I(r, macro, data, seq) macro(r, data, YAS_PP_SEQ_HEAD(seq))
#
# /* YAS_PP_SEQ_FOR_EACH_R */
#
# if ~YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_EDG()
#    define YAS_PP_SEQ_FOR_EACH_R(r, macro, data, seq) YAS_PP_FOR_ ## r((macro, data, seq (nil)), YAS_PP_SEQ_FOR_EACH_P, YAS_PP_SEQ_FOR_EACH_O, YAS_PP_SEQ_FOR_EACH_M)
# else
#    define YAS_PP_SEQ_FOR_EACH_R(r, macro, data, seq) YAS_PP_SEQ_FOR_EACH_R_I(r, macro, data, seq)
#    define YAS_PP_SEQ_FOR_EACH_R_I(r, macro, data, seq) YAS_PP_FOR_ ## r((macro, data, seq (nil)), YAS_PP_SEQ_FOR_EACH_P, YAS_PP_SEQ_FOR_EACH_O, YAS_PP_SEQ_FOR_EACH_M)
# endif
#
#endif // __yas__detail__preprocessor__seq_for_each_hpp
