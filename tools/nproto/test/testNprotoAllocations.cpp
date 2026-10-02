/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

// The conversions of every test message allocate nothing once the objects they convert into have the value's shape,
// and the message side keeps that capacity when a repeated field shrinks. A binary of its own, because the allocation
// counter replaces malloc for the whole process.

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

#include "robot_runtime/robot_realtime/test/AllocationCounter.h"
#include "tools/nproto/test/ProtoAllocations.h"
#include "tools/nproto/test/editions.nproto.pb.h"
#include "tools/nproto/test/imports.nproto.pb.h"
#include "tools/nproto/test/maps.nproto.pb.h"
#include "tools/nproto/test/oneofs.nproto.pb.h"
#include "tools/nproto/test/optionals.nproto.pb.h"
#include "tools/nproto/test/outer.nproto.pb.h"
#include "tools/nproto/test/repeated_fields.nproto.pb.h"
#include "tools/nproto/test/scalars.nproto.pb.h"

namespace nproto::test {
namespace {

using robot::realtime::heapAllocationCount;

template <typename StructType, typename ProtoType>
struct Conversion {
  using Struct = StructType;
  using Proto = ProtoType;
};

template <typename T>
class AllocationTest : public ::testing::Test {};

using Messages = ::testing::Types<Conversion<Scalars, nproto_test::Scalars>,
                                  Conversion<Outer, nproto_test::Outer>,
                                  Conversion<RepeatedFields, nproto_test::RepeatedFields>,
                                  Conversion<Maps, nproto_test::Maps>,
                                  Conversion<Oneofs, nproto_test::Oneofs>,
                                  Conversion<Optionals, nproto_test::Optionals>,
                                  Conversion<Editions, nproto_test::Editions>,
                                  Conversion<Imports, nproto_test::Imports>>;
TYPED_TEST_SUITE(AllocationTest, Messages);

TYPED_TEST(AllocationTest, ConvertingIntoObjectsOfTheValuesShapeDoesNotAllocate) {
  test_support::ExpectConversionsIntoSizedObjectsDoNotAllocate<typename TypeParam::Struct, typename TypeParam::Proto>();
}

TEST(AllocationCounterTest, SeesTheFirstConversion) {
  RepeatedFields value;
  value.strings.assign(3, std::string(40, 'x'));
  value.doubles.setZero(16);
  nproto_test::RepeatedFields proto;
  const std::size_t before = heapAllocationCount();
  ToProto(value, &proto);
  EXPECT_GT(heapAllocationCount() - before, 0u) << "the first conversion sizes the message";
}

TEST(RepeatedCapacityTest, AMessageKeepsTheElementsAShorterValueRemoves) {
  RepeatedFields longer;
  longer.strings.assign(4, std::string(40, 'a'));
  longer.inners.resize(4);
  for (Outer::Inner& inner : longer.inners) {
    inner.leaf.label = std::string(40, 'b');
    inner.weights.setOnes(8);
  }
  RepeatedFields shorter = longer;
  shorter.strings.resize(1);
  shorter.inners.resize(1);

  nproto_test::RepeatedFields proto;
  ToProto(longer, &proto);
  ToProto(shorter, &proto);
  ASSERT_EQ(proto.inners_size(), 1);
  const std::size_t before = heapAllocationCount();
  ToProto(longer, &proto);
  EXPECT_EQ(heapAllocationCount() - before, 0u);
  EXPECT_EQ(proto.strings_size(), 4);
  EXPECT_EQ(proto.inners_size(), 4);
}

}  // namespace
}  // namespace nproto::test
