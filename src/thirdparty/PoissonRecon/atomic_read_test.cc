// SPDX-License-Identifier: BSD-3-Clause

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <type_traits>

#include "MultiThreading.h"
#include "MyExceptions.h"
#include "MyAtomic.h"
#include <gtest/gtest.h>

namespace {

template <typename Value>
void ExpectAtomicRepresentation(Value value) {
  // Windows legacy reads use RMW operations, so the input must stay mutable.
  unsigned char expected[sizeof(Value)];
  std::memcpy(expected, reinterpret_cast<const void*>(&value), sizeof(Value));
  Value reads[] = {PoissonRecon::ReadAtomic(value),
                   PoissonRecon::Atomic<Value>::Read(value)};
  for (const Value& read : reads) {
    unsigned char actual[sizeof(Value)];
    std::memcpy(actual, reinterpret_cast<const void*>(&read), sizeof(Value));
    EXPECT_EQ(std::memcmp(actual, expected, sizeof(Value)), 0);
  }
}

template <typename Scalar>
class PrivateDefaultPod {
 public:
  PrivateDefaultPod(const PrivateDefaultPod&) = default;
  static PrivateDefaultPod Make(Scalar value) {
    PrivateDefaultPod result;
    result.value_ = value;
    return result;
  }

 private:
  PrivateDefaultPod() = default;
  Scalar value_;
};

template <typename Scalar>
void ExpectPrivateDefaultPod(Scalar value) {
  using Value = PrivateDefaultPod<Scalar>;
  static_assert(sizeof(Value) == sizeof(Scalar));
  static_assert(std::is_pod_v<Value> && !std::is_aggregate_v<Value>);
  static_assert(!std::is_default_constructible_v<Value>);
  static_assert(std::is_trivially_copy_constructible_v<Value>);
  ExpectAtomicRepresentation(Value::Make(value));
}

struct ConstMemberPod {
  const uint32_t value;
};
static_assert(std::is_trivially_copyable_v<ConstMemberPod>);
static_assert(!std::is_default_constructible_v<ConstMemberPod>);
// Compile contract only: never perform a Windows RMW on the const subobject.
[[maybe_unused]] ConstMemberPod CompileConstMemberRead(
    const volatile ConstMemberPod& value) {
  return PoissonRecon::ReadAtomic(value);
}

struct DefaultCountedValue {
  static int default_constructions;
  uint32_t value;
  DefaultCountedValue() : value(0) { ++default_constructions; }
  explicit DefaultCountedValue(uint32_t input) : value(input) {}
};
int DefaultCountedValue::default_constructions = 0;
static_assert(sizeof(DefaultCountedValue) == sizeof(uint32_t));
static_assert(std::is_trivially_copyable_v<DefaultCountedValue>);
static_assert(!std::is_trivially_default_constructible_v<DefaultCountedValue>);

TEST(PoissonTemplateReturnTest, AtomicReadPreservesScalarWidthsAndPointers) {
  ExpectAtomicRepresentation(uint8_t{0xa5});
  ExpectAtomicRepresentation(uint32_t{0x89abcdef});
  ExpectAtomicRepresentation(uint64_t{0xfedcba9876543210});
  int target = 7;
  int* pointer = &target;
  ExpectAtomicRepresentation(pointer);
  pointer = nullptr;
  ExpectAtomicRepresentation(pointer);
}

TEST(PoissonTemplateReturnTest, AtomicReadPreservesFloatingRepresentations) {
  static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4);
  static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8);
  for (uint32_t bits : {uint32_t{0x80000000}, uint32_t{0x7fc12345}}) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    ExpectAtomicRepresentation(value);
  }
  for (uint64_t bits :
       {uint64_t{0x8000000000000000}, uint64_t{0x7ff8000000012345}}) {
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    ExpectAtomicRepresentation(value);
  }
}

TEST(PoissonTemplateReturnTest, AtomicReadAcceptsPrivateDefaultPods) {
  ExpectPrivateDefaultPod(uint8_t{0xa5});
  ExpectPrivateDefaultPod(uint32_t{0x89abcdef});
  ExpectPrivateDefaultPod(uint64_t{0xfedcba9876543210});
}

TEST(PoissonTemplateReturnTest, FreeAtomicReadDoesNotDefaultConstruct) {
  DefaultCountedValue value(0x89abcdef);
  const int before = DefaultCountedValue::default_constructions;
  const auto read = PoissonRecon::ReadAtomic(value);
  EXPECT_EQ(read.value, value.value);
  EXPECT_EQ(DefaultCountedValue::default_constructions, before);
}

}  // namespace
