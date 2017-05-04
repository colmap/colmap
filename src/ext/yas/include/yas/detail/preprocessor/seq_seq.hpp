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
#ifndef __yas__detail__preprocessor__seq_seq_hpp
#define __yas__detail__preprocessor__seq_seq_hpp
#
# include <yas/detail/preprocessor/config.hpp>
# include <yas/detail/preprocessor/seq_elem.hpp>
#
# /* YAS_PP_SEQ_HEAD */
#
# define YAS_PP_SEQ_HEAD(seq) YAS_PP_SEQ_ELEM(0, seq)
#
# /* YAS_PP_SEQ_TAIL */
#
# if YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_MWCC()
#    define YAS_PP_SEQ_TAIL(seq) YAS_PP_SEQ_TAIL_1((seq))
#    define YAS_PP_SEQ_TAIL_1(par) YAS_PP_SEQ_TAIL_2 ## par
#    define YAS_PP_SEQ_TAIL_2(seq) YAS_PP_SEQ_TAIL_I ## seq
# elif YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_MSVC()
#    define YAS_PP_SEQ_TAIL(seq) YAS_PP_SEQ_TAIL_ID(YAS_PP_SEQ_TAIL_I seq)
#    define YAS_PP_SEQ_TAIL_ID(id) id
# elif YAS_PP_CONFIG_FLAGS() & YAS_PP_CONFIG_EDG()
#    define YAS_PP_SEQ_TAIL(seq) YAS_PP_SEQ_TAIL_D(seq)
#    define YAS_PP_SEQ_TAIL_D(seq) YAS_PP_SEQ_TAIL_I seq
# else
#    define YAS_PP_SEQ_TAIL(seq) YAS_PP_SEQ_TAIL_I seq
# endif
#
# define YAS_PP_SEQ_TAIL_I(x)
#
# /* YAS_PP_SEQ_NIL */
#
# define YAS_PP_SEQ_NIL(x) (x)
#
#endif // __yas__detail__preprocessor__seq_seq_hpp
