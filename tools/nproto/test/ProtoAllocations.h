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

// The realtime property of nproto's conversions, for the allocation tests: converting into a struct or message that
// already has the value's shape makes no heap allocation. Link //robot_runtime/robot_realtime:allocation_counter into
// the test binary, which counts every allocation of the process.

#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include "absl/status/status.h"

#include "robot_runtime/robot_realtime/test/AllocationCounter.h"
#include "tools/nproto/test/ProtoTestValues.h"

namespace nproto::test_support {

/** `message` serialized and parsed back: maps filled through reflection are moved into their map representation. */
template <typename Proto>
Proto Reparsed(const Proto& message) {
  Proto parsed;
  EXPECT_TRUE(parsed.ParseFromString(message.SerializeAsString()));
  return parsed;
}

/**
 * FromProto() of a message into a struct that holds a value of the same shape (same sizes, same oneof alternatives,
 * same map keys, strings of the same lengths), and ToProto() of a struct into a message of that shape, make no heap
 * allocation. The first conversions, into empty objects, are free to. `repeatedSize` is the number of elements of every
 * repeated field and map (TestValueOptions).
 */
template <typename Struct, typename Proto>
void ExpectConversionsIntoSizedObjectsDoNotAllocate(int repeatedSize = 3) {
  const std::string name(Proto::descriptor()->full_name());
  const Proto first = Reparsed([repeatedSize] {
    Proto message;
    FillWithTestValues(TestValueOptions{.seed = 1, .repeatedSize = repeatedSize}, &message);
    return message;
  }());
  const Proto second = Reparsed([repeatedSize] {
    Proto message;
    FillWithTestValues(TestValueOptions{.seed = 2, .repeatedSize = repeatedSize}, &message);
    return message;
  }());

  Struct value;
  ASSERT_TRUE(FromProto(first, &value).ok());
  std::size_t before = robot::realtime::heapAllocationCount();
  const absl::Status status = FromProto(second, &value);
  const std::size_t fromProtoAllocations = robot::realtime::heapAllocationCount() - before;
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(fromProtoAllocations, 0u) << "FromProto() of " << name << " into a struct of its shape";

  Struct firstValue;
  ASSERT_TRUE(FromProto(first, &firstValue).ok());
  Proto message;
  ToProto(firstValue, &message);
  before = robot::realtime::heapAllocationCount();
  ToProto(value, &message);
  const std::size_t toProtoAllocations = robot::realtime::heapAllocationCount() - before;
  EXPECT_EQ(toProtoAllocations, 0u) << "ToProto() of " << name << " into a message of its shape";
  EXPECT_TRUE(ProtoEquals(second, message));
}

}  // namespace nproto::test_support
