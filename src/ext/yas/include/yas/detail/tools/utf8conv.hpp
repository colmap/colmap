// Based on the code from Avanesov Tigran, 2009-2011:
// http://forum.vingrad.ru/index.php?showtopic=249965&view=findpost&p=1803380
// http://forum.vingrad.ru/index.php?showtopic=249965&view=findpost&p=1803439

#ifndef __yas__detail__tools__utf8conv_hpp
#define __yas__detail__tools__utf8conv_hpp

#include <yas/detail/tools/cast.hpp>

#include <string>

#if defined(YAS_SERIALIZE_BOOST_TYPES)
#	include <boost/container/string.hpp>
#endif

namespace yas {
namespace detail {

/***************************************************************************/

template<typename D, typename S>
void to_utf8(D &dst, const S &src) {
	for ( auto it = src.begin(); it != src.end(); ++it ) {
		std::wstring::value_type nchar = *it;
		if (nchar <= 0x7F) {
			dst += YAS_SCAST(std::string::value_type, nchar);
		} else if (nchar <= 0x07FF) {
			dst += YAS_SCAST(char, 0xC0 | (nchar >> 6));
			dst += YAS_SCAST(char, 0x80 | (nchar & 0x3F));
		}
#if WCHAR_MAX > 0xFFFF
		else if (nchar <= 0xFFFF) {
			dst += YAS_SCAST(char, 0xE0 | (nchar >> 12));
			dst += YAS_SCAST(char, 0x80 | ((nchar >> 6) & 0x3F));
			dst += YAS_SCAST(char, 0x80 | (nchar & 0x3F));
		} else if (nchar  <= 0x1FFFFF) {
			dst += YAS_SCAST(char, 0xF0 | (nchar >> 18));
			dst += YAS_SCAST(char, 0x80 | ((nchar >> 12) & 0x3F));
			dst += YAS_SCAST(char, 0x80 | ((nchar >> 6) & 0x3F));
			dst += YAS_SCAST(char, 0x80 | (nchar & 0x3F));
		}
#endif
	}
}

template<typename D, typename S>
void from_utf8(D &dst, const S &src) {
	std::wstring::value_type tmp = 0;
	for ( auto it = src.begin(); it != src.end(); ++it ) {
		unsigned char nchar = YAS_SCAST(unsigned char, *it);
		if (nchar <= 0x7F) {
			tmp = nchar;
		} else if ((nchar & 0xE0) == 0xC0) {
			tmp = (nchar & 0x1F) << 6;
			nchar = YAS_SCAST(unsigned char, *(++it));
			tmp |= nchar & 0x3F;
		} else if ((nchar & 0xF0) == 0xE0) {
			tmp = (nchar & 0x0F) << 12;
			nchar = YAS_SCAST(unsigned char, *(++it));
			tmp |= (nchar & 0x3F) << 6;
			nchar = YAS_SCAST(unsigned char, *(++it));
			tmp |= (nchar & 0x3F);
		} else if ((nchar & 0xF8) == 0xF0) {
			tmp = (nchar & 0x0F) << 18;
			nchar = YAS_SCAST(unsigned char, *(++it));
			tmp |= (nchar & 0x3F) << 12;
			nchar = YAS_SCAST(unsigned char, *(++it));
			tmp |= (nchar & 0x3F) << 6;
			nchar = YAS_SCAST(unsigned char, *(++it));
			tmp |= (nchar & 0x3F);
		}
		dst += tmp;
	}
}

/***************************************************************************/

template<typename To, typename From>
struct TypeConverter;

template<>
struct TypeConverter<std::string, std::wstring> {
	static void Convert(std::string &dst, const std::wstring &src) {
		to_utf8(dst, src);
	}
};

template<>
struct TypeConverter<std::wstring, std::string> {
	static void Convert(std::wstring &dst, const std::string &src) {
		from_utf8(dst, src);
	}
};

/***************************************************************************/

#if defined(YAS_SERIALIZE_BOOST_TYPES)

template<>
struct TypeConverter<boost::container::basic_string<char>, boost::container::basic_string<wchar_t>> {
	static void Convert(boost::container::basic_string<char> &dst, const boost::container::basic_string<wchar_t> &src) {
		to_utf8(dst, src);
	}
};

template<>
struct TypeConverter<boost::container::basic_string<wchar_t>, boost::container::basic_string<char>> {
	static void Convert(boost::container::basic_string<wchar_t> &dst, const boost::container::basic_string<char> &src) {
		from_utf8(dst, src);
	}
};

#endif // YAS_SERIALIZE_BOOST_TYPES

/***************************************************************************/

} // namespace detail
} // namespace yas

#endif // __yas__detail__tools__utf8conv_hpp
