#ifndef __yas__tests__base__include__base_hpp
#define __yas__tests__base__include__base_hpp

#define YAS_TEST_REPORT(log, msg) \
    log << __FILE__ << "(" << __LINE__ << "): " << msg << std::endl;

#include "./array.hpp"
#include "./auto_array.hpp"
#include "./base_object.hpp"
#include "./bitset.hpp"
#include "./chrono.hpp"
#include "./complex.hpp"
#include "./buffer.hpp"
#include "./endian.hpp"
#include "./enum.hpp"
#include "./forward_list.hpp"
#include "./fundamental.hpp"
#include "./header.hpp"

#if defined(YAS_SERIALIZE_BOOST_TYPES)
#include "./boost_cont_string.hpp"
// #include "./boost_cont_wstring.hpp"
#include "./boost_cont_vector.hpp"
#include "./boost_cont_static_vector.hpp"
#include "./boost_cont_stable_vector.hpp"
#include "./boost_cont_list.hpp"
#include "./boost_cont_slist.hpp"
#include "./boost_cont_map.hpp"
#include "./boost_cont_multimap.hpp"
#include "./boost_cont_set.hpp"
#include "./boost_cont_multiset.hpp"
#include "./boost_cont_flat_map.hpp"
#include "./boost_cont_flat_multimap.hpp"
#include "./boost_cont_flat_set.hpp"
#include "./boost_cont_flat_multiset.hpp"
#include "./boost_cont_deque.hpp"
#include "./boost_tuple.hpp"
#endif // defined(YAS_SERIALIZE_BOOST_TYPES)

#include "./list.hpp"
#include "./map.hpp"
#include "./multimap.hpp"
#include "./multiset.hpp"
#include "./optional.hpp"
#include "./pair.hpp"
#include "./deque.hpp"
#include "./set.hpp"
#include "./string.hpp"
#include "./tuple.hpp"
#include "./unordered_map.hpp"
#include "./unordered_multimap.hpp"
#include "./unordered_multiset.hpp"
#include "./unordered_set.hpp"
#include "./vector.hpp"
#include "./version.hpp"
#include "./wstring.hpp"
#include "./one_function.hpp"
#include "./one_method.hpp"
#include "./split_functions.hpp"
#include "./split_methods.hpp"
#include "./serialization_methods.hpp"
#include "./yas_object.hpp"

#endif // __yas__tests__base__include__base_hpp
