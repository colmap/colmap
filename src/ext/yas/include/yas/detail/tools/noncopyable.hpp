
#ifndef __yas__detail__tools__noncopyable_hpp
#define __yas__detail__tools__noncopyable_hpp

#define YAS_NONCOPYABLE(type) \
	type(const type &) = delete; \
	type& operator=(const type &) = delete;

#endif // __yas__detail__tools__noncopyable_hpp
