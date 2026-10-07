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

// Test values for any protobuf message, and the round-trip checks every nproto struct passes. Shared by nproto's own
// tests and by the tests of the packages that use nproto (humanoid_nmpc/humanoid_mpc_msgs).

#pragma once

#include <string>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "google/protobuf/message.h"
#include "gtest/gtest.h"

namespace nproto::test_support {

struct TestValueOptions {
  // Varies every value except map keys, sizes and the oneof alternative, so that two seeds give messages of one shape.
  int seed = 1;
  // The number of elements of every repeated field and map (maps with bool keys hold at most two).
  int repeatedSize = 2;
  // Which alternative of each oneof is set: this one, modulo the number of alternatives.
  int oneofChoice = 0;
  // When positive, the fields whose number plus the seed is a multiple of it are left unset.
  int skipEvery = 0;
};

/**
 * Sets the fields of `message`, recursively, to values that differ from their defaults: numbers from the seed and the
 * field number, strings of a fixed length longer than any small-string buffer, the enum value after the default,
 * `repeatedSize` elements per repeated field and map, map keys that depend on the index only. Messages deeper than
 * eight levels are left empty.
 */
void FillWithTestValues(const TestValueOptions& options, google::protobuf::Message* absl_nonnull message);

/** `message` serialized with maps in key order, so that equal messages give equal bytes. */
std::string DeterministicBytes(const google::protobuf::Message& message);

/** Success when the two messages are equal; otherwise a failure that prints both. */
::testing::AssertionResult ProtoEquals(const google::protobuf::Message& expected, const google::protobuf::Message& actual);

/**
 * The properties every nproto struct and its message have, for a message type with at least one field:
 * - a message full of test values converts to a struct that differs from the default one;
 * - proto -> struct -> proto and struct -> proto -> struct reproduce what they started from;
 * - converting into a struct or message that holds other values (another seed, other sizes, unset fields, another
 *   oneof alternative) overwrites all of it;
 * - the empty message is the default struct.
 */
template <typename Struct, typename Proto>
void ExpectRoundTrips() {
  const bool hasFields = Proto::descriptor()->field_count() > 0;
  Proto filled;
  FillWithTestValues(TestValueOptions{.seed = 1, .repeatedSize = 2}, &filled);
  Struct value;
  const absl::Status fromFilled = FromProto(filled, &value);
  ASSERT_TRUE(fromFilled.ok()) << fromFilled;
  if (hasFields) {
    EXPECT_TRUE(value != Struct{}) << "the test values of " << Proto::descriptor()->full_name() << " reach the struct";
  }

  Proto back;
  ToProto(value, &back);
  EXPECT_TRUE(ProtoEquals(filled, back));
  Struct again;
  const absl::Status fromBack = FromProto(back, &again);
  ASSERT_TRUE(fromBack.ok()) << fromBack;
  EXPECT_TRUE(again == value);
  EXPECT_FALSE(again != value);

  // Into objects that held something else: another seed, more elements, unset fields, another oneof alternative.
  Proto other;
  FillWithTestValues(TestValueOptions{.seed = 7, .repeatedSize = 3, .oneofChoice = 1, .skipEvery = 3}, &other);
  Struct dirty;
  const absl::Status fromOther = FromProto(other, &dirty);
  ASSERT_TRUE(fromOther.ok()) << fromOther;
  const absl::Status overwrite = FromProto(filled, &dirty);
  ASSERT_TRUE(overwrite.ok()) << overwrite;
  EXPECT_TRUE(dirty == value);
  ToProto(value, &other);
  EXPECT_TRUE(ProtoEquals(filled, other));

  // And the other way around: a sparse message into a struct that held a full one.
  Proto sparse;
  FillWithTestValues(TestValueOptions{.seed = 3, .repeatedSize = 1, .oneofChoice = 2, .skipEvery = 2}, &sparse);
  Struct fresh;
  const absl::Status fromSparse = FromProto(sparse, &fresh);
  ASSERT_TRUE(fromSparse.ok()) << fromSparse;
  Struct reused = value;
  const absl::Status intoReused = FromProto(sparse, &reused);
  ASSERT_TRUE(intoReused.ok()) << intoReused;
  EXPECT_TRUE(reused == fresh);

  Struct fromEmpty = value;
  const absl::Status empty = FromProto(Proto(), &fromEmpty);
  ASSERT_TRUE(empty.ok()) << empty;
  EXPECT_TRUE(fromEmpty == Struct{});
}

}  // namespace nproto::test_support
