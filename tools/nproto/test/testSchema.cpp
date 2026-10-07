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

// nproto/Schema.h: the unknown fields of a received message, by their paths, and the schema fingerprint, which the Python
// twin (nproto_schema.py, test_nproto_schema.py) must compute alike.

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "nproto/Schema.h"
#include "tools/nproto/test/outer.pb.h"
#include "tools/nproto/test/scalars.pb.h"

namespace nproto {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

// The bytes of a varint field 103 of value 1 and of a varint field 40 of value 1: fields no test message has.
constexpr char kUnknownField103[] = "\xb8\x06\x01";
constexpr char kUnknownField40[] = "\xc0\x02\x01";

TEST(UnknownFieldPaths, AMessageOfThisSchemaHasNone) {
  nproto_test::Outer outer;
  outer.mutable_inner()->set_kind(nproto_test::Outer::KIND_SWING);
  outer.add_inners()->add_weights(1.0);
  nproto_test::Outer parsed;
  ASSERT_TRUE(parsed.ParseFromString(outer.SerializeAsString()));
  EXPECT_THAT(UnknownFieldPaths(parsed), IsEmpty());
}

TEST(UnknownFieldPaths, AFieldOfAnotherSchemaIsNamedByItsPathAndNumber) {
  // Outer { inner { <40> } inners { } inners { <40> } <103> }, as a sender with two more fields would encode it.
  const std::string inner = std::string(kUnknownField40, sizeof(kUnknownField40) - 1);
  std::string bytes;
  bytes += '\x0a';  // inner, length-delimited
  bytes += static_cast<char>(inner.size());
  bytes += inner;
  bytes += std::string("\x12\x00", 2);  // inners[0], empty
  bytes += '\x12';                      // inners[1]
  bytes += static_cast<char>(inner.size());
  bytes += inner;
  bytes += std::string(kUnknownField103, sizeof(kUnknownField103) - 1);
  nproto_test::Outer parsed;
  ASSERT_TRUE(parsed.ParseFromString(bytes)) << "unknown fields parse: the receiver keeps them";
  EXPECT_THAT(UnknownFieldPaths(parsed), ElementsAre(": field 103", "inner: field 40", "inners[1]: field 40"));
}

// The fingerprint of nproto_test.Outer, which test_nproto_schema.py pins alike: Python and C++ walk a schema the same way.
// LINT.IfChange(outer_fingerprint)
constexpr char kOuterFingerprintText[] =
    "message nproto_test.Outer\n"
    "field 1 inner 11 singular presence nproto_test.Outer.Inner\n"
    "field 2 inners 11 repeated no_presence nproto_test.Outer.Inner\n"
    "field 3 kind 14 singular no_presence nproto_test.Outer.Kind\n"
    "field 4 leaf 11 singular presence nproto_test.Outer.Inner.Leaf\n"
    "field 5 inner_by_name 11 repeated no_presence nproto_test.Outer.InnerByNameEntry\n"
    "message nproto_test.Outer.Inner\n"
    "field 1 kind 14 singular no_presence nproto_test.Outer.Kind\n"
    "field 2 leaf 11 singular presence nproto_test.Outer.Inner.Leaf\n"
    "field 3 later 11 singular presence nproto_test.Outer.Later\n"
    "field 4 weights 1 repeated no_presence -\n"
    "message nproto_test.Outer.Inner.Leaf\n"
    "field 1 label 9 singular no_presence -\n"
    "field 2 kind 14 singular no_presence nproto_test.Outer.Kind\n"
    "message nproto_test.Outer.InnerByNameEntry\n"
    "field 1 key 9 singular no_presence -\n"
    "field 2 value 11 singular presence nproto_test.Outer.Inner\n"
    "message nproto_test.Outer.Later\n"
    "field 1 value 1 singular no_presence -\n"
    "enum nproto_test.Outer.Kind\n"
    "value 0 KIND_UNSPECIFIED\n"
    "value 1 KIND_STANCE\n"
    "value 2 KIND_SWING\n";
constexpr char kOuterFingerprint[] = "f97c11d0be6d2bfc";
// LINT.ThenChange(//tools/nproto/test/test_nproto_schema.py:outer_fingerprint)

TEST(SchemaFingerprint, IsTheHashOfEveryReachableMessageAndEnum) {
  EXPECT_EQ(SchemaFingerprintText(*nproto_test::Outer::descriptor()), kOuterFingerprintText);
  EXPECT_EQ(SchemaFingerprint(*nproto_test::Outer::descriptor()), kOuterFingerprint);
}

TEST(SchemaFingerprint, DiffersBetweenSchemas) {
  EXPECT_NE(SchemaFingerprint(*nproto_test::Outer::descriptor()), SchemaFingerprint(*nproto_test::Scalars::descriptor()));
  EXPECT_NE(SchemaFingerprint(*nproto_test::Outer::descriptor()), SchemaFingerprint(*nproto_test::Outer::Inner::descriptor()));
}

}  // namespace
}  // namespace nproto
