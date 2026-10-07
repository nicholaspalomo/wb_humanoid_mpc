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

// The structs nproto generates for the test protos of this directory and their conversions: the type of every kind
// of field, defaults, equality, round trips, presence, oneofs, maps, and the errors of FromProto().

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "gtest/gtest.h"

#include "nproto_test/other/point.nproto.pb.h"
#include "nproto_test/other/unit.nproto.pb.h"
#include "tools/nproto/test/ProtoTestValues.h"
#include "tools/nproto/test/defaults.nproto.pb.h"
#include "tools/nproto/test/editions.nproto.pb.h"
#include "tools/nproto/test/empty.nproto.pb.h"
#include "tools/nproto/test/imports.nproto.pb.h"
#include "tools/nproto/test/maps.nproto.pb.h"
#include "tools/nproto/test/oneofs.nproto.pb.h"
#include "tools/nproto/test/optionals.nproto.pb.h"
#include "tools/nproto/test/outer.nproto.pb.h"
#include "tools/nproto/test/repeated_fields.nproto.pb.h"
#include "tools/nproto/test/retired.nproto.pb.h"
#include "tools/nproto/test/scalars.nproto.pb.h"
#include "tools/nproto/test/severity.nproto.pb.h"
#include "tools/nproto/test/shadowing.nproto.pb.h"

namespace nproto::test {
namespace {

using test_support::ExpectRoundTrips;
using test_support::ProtoEquals;

// ---------------------------------------------------------------------------------------------------------------------
// The C++ type of every kind of field
// ---------------------------------------------------------------------------------------------------------------------

template <typename Member, typename Expected>
constexpr bool kIs = std::is_same_v<Member, Expected>;

// The option names the struct; the protobuf class of the same message is another type.
static_assert(std::is_class_v<Scalars> && !std::is_same_v<Scalars, nproto_test::Scalars>);
static_assert(std::is_class_v<other::Point> && !std::is_same_v<other::Point, nproto_test::other::Point>);
// Plain data: copyable and movable values.
static_assert(std::is_copy_constructible_v<RepeatedFields> && std::is_nothrow_move_constructible_v<Scalars>);

static_assert(kIs<decltype(Scalars::double_value), double>);
static_assert(kIs<decltype(Scalars::float_value), float>);
static_assert(kIs<decltype(Scalars::int32_value), int32_t>);
static_assert(kIs<decltype(Scalars::int64_value), int64_t>);
static_assert(kIs<decltype(Scalars::uint32_value), uint32_t>);
static_assert(kIs<decltype(Scalars::uint64_value), uint64_t>);
static_assert(kIs<decltype(Scalars::sint32_value), int32_t>);
static_assert(kIs<decltype(Scalars::sint64_value), int64_t>);
static_assert(kIs<decltype(Scalars::fixed32_value), uint32_t>);
static_assert(kIs<decltype(Scalars::fixed64_value), uint64_t>);
static_assert(kIs<decltype(Scalars::sfixed32_value), int32_t>);
static_assert(kIs<decltype(Scalars::sfixed64_value), int64_t>);
static_assert(kIs<decltype(Scalars::bool_value), bool>);
static_assert(kIs<decltype(Scalars::string_value), std::string>);
static_assert(kIs<decltype(Scalars::bytes_value), std::string>);

static_assert(kIs<decltype(RepeatedFields::doubles), Eigen::VectorXd>);
static_assert(kIs<decltype(RepeatedFields::floats), Eigen::VectorXf>);
static_assert(kIs<decltype(RepeatedFields::int32s), std::vector<int32_t>>);
static_assert(kIs<decltype(RepeatedFields::uint64s), std::vector<uint64_t>>);
static_assert(kIs<decltype(RepeatedFields::sfixed64s), std::vector<int64_t>>);
static_assert(kIs<decltype(RepeatedFields::bools), std::vector<bool>>);
static_assert(kIs<decltype(RepeatedFields::strings), std::vector<std::string>>);
static_assert(kIs<decltype(RepeatedFields::blobs), std::vector<std::string>>);
static_assert(kIs<decltype(RepeatedFields::severities), std::vector<Severity>>);
static_assert(kIs<decltype(RepeatedFields::kinds), std::vector<Outer::Kind>>);
static_assert(kIs<decltype(RepeatedFields::scalars), std::vector<Scalars>>);
static_assert(kIs<decltype(RepeatedFields::inners), std::vector<Outer::Inner>>);

static_assert(kIs<decltype(Maps::double_by_name), std::map<std::string, double>>);
static_assert(kIs<decltype(Maps::name_by_id), std::map<int32_t, std::string>>);
static_assert(kIs<decltype(Maps::scalars_by_id), std::map<int64_t, Scalars>>);
static_assert(kIs<decltype(Maps::severity_by_id), std::map<uint32_t, Severity>>);
static_assert(kIs<decltype(Maps::inner_by_id), std::map<uint64_t, Outer::Inner>>);
static_assert(kIs<decltype(Maps::count_by_flag), std::map<bool, int32_t>>);
static_assert(kIs<decltype(Maps::kind_by_code), std::map<int64_t, Outer::Kind>>);

// Nested messages by value; nested enums as nested enum classes.
static_assert(kIs<decltype(Outer::inner), Outer::Inner>);
static_assert(kIs<decltype(Outer::leaf), Outer::Inner::Leaf>);
static_assert(kIs<decltype(Outer::Inner::later), Outer::Later>);
static_assert(kIs<decltype(Outer::kind), Outer::Kind>);
static_assert(kIs<decltype(Imports::point), other::Point>);
static_assert(kIs<decltype(Imports::frame), other::Point::Frame>);
static_assert(kIs<decltype(Imports::units), std::vector<other::Unit>>);
static_assert(kIs<decltype(Imports::point_by_name), std::map<std::string, other::Point>>);

// A oneof is a std::variant whose index 0 means "not set"; alternatives of one type are told apart by index.
static_assert(
    kIs<decltype(Oneofs::shape), std::variant<std::monostate, double, double, std::string, Scalars, Severity, Outer::Inner, std::string>>);
static_assert(kIs<decltype(Oneofs::flag), std::variant<std::monostate, bool, int64_t>>);
static_assert(Oneofs::kRadiusIndex == 1 && Oneofs::kSideIndex == 2 && Oneofs::kBlobIndex == 7);
static_assert(Oneofs::kEnabledIndex == 1 && Oneofs::kCountIndex == 2);

// Explicit presence: proto3 `optional`, proto2 `optional` without a default, edition 2023 without IMPLICIT.
static_assert(kIs<decltype(Optionals::maybe_double), std::optional<double>>);
static_assert(kIs<decltype(Optionals::maybe_string), std::optional<std::string>>);
static_assert(kIs<decltype(Optionals::maybe_severity), std::optional<Severity>>);
static_assert(kIs<decltype(Optionals::maybe_scalars), std::optional<Scalars>>);
static_assert(kIs<decltype(Optionals::maybe_kind), std::optional<Outer::Kind>>);
static_assert(kIs<decltype(Optionals::plain_double), double>);
static_assert(kIs<decltype(Optionals::plain_scalars), Scalars>);
static_assert(kIs<decltype(Imports::maybe_point), std::optional<other::Point>>);
static_assert(kIs<decltype(Defaults::count), int32_t>);
static_assert(kIs<decltype(Defaults::no_default), std::optional<int32_t>>);
static_assert(kIs<decltype(Defaults::no_default_level), std::optional<Defaults::Level>>);
static_assert(kIs<decltype(Defaults::required_count), int32_t>);
static_assert(kIs<decltype(Editions::explicit_count), std::optional<int32_t>>);
static_assert(kIs<decltype(Editions::implicit_count), int32_t>);
static_assert(kIs<decltype(Editions::defaulted), int32_t>);
static_assert(kIs<decltype(Editions::name), std::optional<std::string>>);
static_assert(kIs<decltype(Editions::scalars), Scalars>);
// A message field with (nproto.optional_message) keeps its presence.
static_assert(kIs<decltype(Editions::maybe_scalars), std::optional<Scalars>>);
// Retired fields are options only: the struct holds the live fields.
static_assert(kIs<decltype(Retired::step_width), double> && kIs<decltype(Retired::block), Retired::Block>);

// Members named like types: the generated code names the types fully qualified.
static_assert(kIs<decltype(Shadowing::Scalars), Scalars>);
static_assert(kIs<decltype(Shadowing::Outer), Outer::Kind>);

// Enums are scoped, 32-bit, with Google-style enumerators.
static_assert(std::is_enum_v<Severity> && !std::is_convertible_v<Severity, int>);
static_assert(kIs<std::underlying_type_t<Severity>, int32_t>);
static_assert(static_cast<int>(Severity::kUnspecified) == 0 && static_cast<int>(Severity::kLevel10) == 10);
static_assert(Severity::kSevere == Severity::kHigh);
static_assert(static_cast<int>(Outer::Kind::kSwing) == 2 && static_cast<int>(other::Unit::kFoot) == 1);

// ---------------------------------------------------------------------------------------------------------------------
// Round trips of every message
// ---------------------------------------------------------------------------------------------------------------------

template <typename StructType, typename ProtoType>
struct Conversion {
  using Struct = StructType;
  using Proto = ProtoType;
};

template <typename T>
class RoundTripTest : public ::testing::Test {};

using AllMessages = ::testing::Types<Conversion<Scalars, nproto_test::Scalars>,
                                     Conversion<Outer, nproto_test::Outer>,
                                     Conversion<Outer::Inner, nproto_test::Outer_Inner>,
                                     Conversion<Outer::Inner::Leaf, nproto_test::Outer_Inner_Leaf>,
                                     Conversion<Outer::Later, nproto_test::Outer_Later>,
                                     Conversion<RepeatedFields, nproto_test::RepeatedFields>,
                                     Conversion<Maps, nproto_test::Maps>,
                                     Conversion<Oneofs, nproto_test::Oneofs>,
                                     Conversion<Optionals, nproto_test::Optionals>,
                                     Conversion<Defaults, nproto_test::Defaults>,
                                     Conversion<Editions, nproto_test::Editions>,
                                     Conversion<Imports, nproto_test::Imports>,
                                     Conversion<Empty, nproto_test::Empty>,
                                     Conversion<Shadowing, nproto_test::Shadowing>,
                                     Conversion<Retired, nproto_test::Retired>,
                                     Conversion<Retired::Block, nproto_test::Retired_Block>,
                                     Conversion<other::Point, nproto_test::other::Point>>;
TYPED_TEST_SUITE(RoundTripTest, AllMessages);

TYPED_TEST(RoundTripTest, ConvertsBothWaysWithoutLosingAnything) {
  ExpectRoundTrips<typename TypeParam::Struct, typename TypeParam::Proto>();
}

// Every alternative of every oneof, not only the one the shared check picks.
TEST(RoundTripTest, EveryOneofAlternativeRoundTrips) {
  for (int choice = 0; choice < 7; ++choice) {
    nproto_test::Oneofs proto;
    test_support::FillWithTestValues(test_support::TestValueOptions{.seed = 2, .oneofChoice = choice}, &proto);
    Oneofs value;
    ASSERT_TRUE(FromProto(proto, &value).ok());
    EXPECT_EQ(value.shape.index(), static_cast<size_t>(choice) + 1);
    EXPECT_EQ(value.flag.index(), static_cast<size_t>(choice % 2) + 1);
    nproto_test::Oneofs back;
    ToProto(value, &back);
    EXPECT_TRUE(ProtoEquals(proto, back)) << "alternative " << choice;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------------------------------------------------

TEST(DefaultsTest, Proto3MembersStartAtZero) {
  const Scalars scalars;
  EXPECT_EQ(scalars.double_value, 0.0);
  EXPECT_EQ(scalars.int64_value, 0);
  EXPECT_EQ(scalars.uint32_value, 0u);
  EXPECT_FALSE(scalars.bool_value);
  EXPECT_TRUE(scalars.string_value.empty());
  const RepeatedFields repeated;
  EXPECT_EQ(repeated.doubles.size(), 0);
  EXPECT_TRUE(repeated.strings.empty());
  EXPECT_EQ(Outer().kind, Outer::Kind::kUnspecified);
  EXPECT_EQ(Oneofs().shape.index(), 0u);
  EXPECT_FALSE(Optionals().maybe_double.has_value());
}

TEST(DefaultsTest, Proto2AndEditionsDefaultsAreTheMessagesDefaults) {
  const nproto_test::Defaults proto;
  const Defaults value;
  EXPECT_EQ(value.count, proto.count());
  EXPECT_EQ(value.count, 5);
  EXPECT_EQ(value.gain, proto.gain());
  EXPECT_EQ(value.ratio, proto.ratio());
  EXPECT_EQ(value.name, proto.name());
  EXPECT_EQ(value.name, "hello \"world\"\n");
  EXPECT_EQ(value.blob, proto.blob());
  EXPECT_EQ(value.blob, std::string("\001\002\377"));
  EXPECT_EQ(value.enabled, proto.enabled());
  EXPECT_EQ(value.level, Defaults::Level::kHigh);
  EXPECT_EQ(value.smallest, std::numeric_limits<int64_t>::min());
  EXPECT_EQ(value.smallest, proto.smallest());
  EXPECT_EQ(value.largest, std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(value.largest, proto.largest());
  EXPECT_EQ(value.limit, std::numeric_limits<double>::infinity());
  EXPECT_EQ(value.floor, -std::numeric_limits<double>::infinity());
  EXPECT_EQ(value.smallest32, proto.smallest32());
  EXPECT_EQ(value.tiny, proto.tiny());
  EXPECT_EQ(value.epsilon, proto.epsilon());
  EXPECT_EQ(value.epsilon, 1.0e-8);
  EXPECT_FALSE(value.no_default.has_value());
  EXPECT_FALSE(value.no_default_level.has_value());
  EXPECT_EQ(value.required_count, 0);

  const nproto_test::Editions editions;
  EXPECT_EQ(Editions().defaulted, editions.defaulted());
  EXPECT_EQ(Editions().defaulted, 7);
  EXPECT_FALSE(Editions().explicit_count.has_value());
}

// ---------------------------------------------------------------------------------------------------------------------
// Equality
// ---------------------------------------------------------------------------------------------------------------------

TEST(EqualityTest, EveryMemberTakesPart) {
  Scalars changed;
  EXPECT_TRUE(changed == Scalars());
  changed.sfixed64_value = 1;
  EXPECT_TRUE(changed != Scalars());
  changed = Scalars();
  changed.bytes_value = "x";
  EXPECT_FALSE(changed == Scalars());

  Outer outer;
  outer.inner.leaf.label = "deep";
  EXPECT_TRUE(outer != Outer());
}

TEST(EqualityTest, EigenVectorsOfDifferentSizesAreUnequal) {
  RepeatedFields shorter;
  shorter.doubles = Eigen::VectorXd::Constant(2, 1.0);
  RepeatedFields longer;
  longer.doubles = Eigen::VectorXd::Constant(3, 1.0);
  EXPECT_TRUE(shorter != longer);
  EXPECT_TRUE(longer != shorter);
  RepeatedFields same = shorter;
  EXPECT_TRUE(same == shorter);
  same.doubles[1] = 2.0;
  EXPECT_TRUE(same != shorter);
}

TEST(EqualityTest, ComparesExactlyAsDoubleDoes) {
  Scalars nan;
  nan.double_value = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(nan != nan);
  Scalars negativeZero;
  negativeZero.double_value = -0.0;
  EXPECT_TRUE(negativeZero == Scalars());
}

TEST(EqualityTest, OneofAlternativesOfOneTypeAreDifferent) {
  Oneofs radius;
  radius.shape.emplace<Oneofs::kRadiusIndex>(1.0);
  Oneofs side;
  side.shape.emplace<Oneofs::kSideIndex>(1.0);
  EXPECT_TRUE(radius != side);
  EXPECT_TRUE(radius != Oneofs());
}

TEST(EqualityTest, PresenceTakesPart) {
  Optionals present;
  present.maybe_double = 0.0;
  EXPECT_TRUE(present != Optionals());
}

// ---------------------------------------------------------------------------------------------------------------------
// Presence, oneofs, maps and repeated fields
// ---------------------------------------------------------------------------------------------------------------------

TEST(PresenceTest, OptionalFieldsKeepPresenceBothWays) {
  nproto_test::Optionals proto;
  proto.set_maybe_double(0.0);
  proto.set_maybe_string("");
  proto.mutable_maybe_scalars();
  Optionals value;
  ASSERT_TRUE(FromProto(proto, &value).ok());
  EXPECT_EQ(value.maybe_double, std::optional<double>(0.0));
  ASSERT_TRUE(value.maybe_string.has_value());
  EXPECT_TRUE(value.maybe_scalars.has_value());
  EXPECT_FALSE(value.maybe_int32.has_value());
  EXPECT_FALSE(value.maybe_severity.has_value());

  nproto_test::Optionals back;
  back.set_maybe_int32(4);
  ToProto(value, &back);
  EXPECT_TRUE(back.has_maybe_double());
  EXPECT_TRUE(back.has_maybe_string());
  EXPECT_TRUE(back.has_maybe_scalars());
  EXPECT_FALSE(back.has_maybe_int32()) << "an absent member clears the field";
  EXPECT_FALSE(back.has_maybe_kind());

  // proto3 fields without `optional` and edition 2023 IMPLICIT fields have no presence of their own.
  nproto_test::Editions editions;
  editions.set_explicit_count(0);
  Editions fromEditions;
  ASSERT_TRUE(FromProto(editions, &fromEditions).ok());
  ASSERT_TRUE(fromEditions.explicit_count.has_value());
  EXPECT_FALSE(fromEditions.name.has_value());
  nproto_test::Editions editionsBack;
  ToProto(fromEditions, &editionsBack);
  EXPECT_TRUE(editionsBack.has_explicit_count());
  EXPECT_FALSE(editionsBack.has_name());
}

TEST(PresenceTest, AnOptionalMessageFieldKeepsPresenceBothWays) {
  nproto_test::Editions proto;
  Editions value;
  ASSERT_TRUE(FromProto(proto, &value).ok());
  EXPECT_FALSE(value.maybe_scalars.has_value()) << "absent is std::nullopt, not a default struct";

  // Present with every field at its default: a default struct, which is not std::nullopt.
  proto.mutable_maybe_scalars();
  ASSERT_TRUE(FromProto(proto, &value).ok());
  EXPECT_TRUE(value.maybe_scalars == std::optional<Scalars>(Scalars{}));

  Scalars changed;
  changed.int32_value = 3;
  value.maybe_scalars = changed;
  nproto_test::Editions back;
  ToProto(value, &back);
  ASSERT_TRUE(back.has_maybe_scalars());
  EXPECT_EQ(back.maybe_scalars().int32_value(), 3);
  value.maybe_scalars.reset();
  ToProto(value, &back);
  EXPECT_FALSE(back.has_maybe_scalars()) << "std::nullopt clears the field";
  // The plain message field beside it keeps no presence: ToProto() always sets it.
  EXPECT_TRUE(back.has_scalars());
}

TEST(OneofTest, TheVariantFollowsTheSetAlternative) {
  nproto_test::Oneofs proto;
  proto.set_side(2.5);
  proto.set_count(9);
  Oneofs value;
  ASSERT_TRUE(FromProto(proto, &value).ok());
  ASSERT_EQ(value.shape.index(), Oneofs::kSideIndex);
  EXPECT_EQ(std::get<Oneofs::kSideIndex>(value.shape), 2.5);
  ASSERT_EQ(value.flag.index(), Oneofs::kCountIndex);
  EXPECT_EQ(std::get<Oneofs::kCountIndex>(value.flag), 9);

  // Another alternative replaces it, in the struct and in the message.
  proto.mutable_inner()->set_kind(nproto_test::Outer_Kind_KIND_SWING);
  ASSERT_TRUE(FromProto(proto, &value).ok());
  ASSERT_EQ(value.shape.index(), Oneofs::kInnerIndex);
  EXPECT_EQ(std::get<Oneofs::kInnerIndex>(value.shape).kind, Outer::Kind::kSwing);
  value.shape.emplace<Oneofs::kLabelIndex>("label");
  ToProto(value, &proto);
  EXPECT_EQ(proto.shape_case(), nproto_test::Oneofs::kLabel);
  EXPECT_EQ(proto.label(), "label");

  // std::monostate is "none set".
  value.shape = std::monostate();
  ToProto(value, &proto);
  EXPECT_EQ(proto.shape_case(), nproto_test::Oneofs::SHAPE_NOT_SET);
  proto.clear_flag();
  ASSERT_TRUE(FromProto(proto, &value).ok());
  EXPECT_EQ(value.flag.index(), 0u);
}

TEST(MapTest, MapsAreInKeyOrderAndFollowTheMessage) {
  nproto_test::Maps proto;
  (*proto.mutable_double_by_name())["b"] = 2.0;
  (*proto.mutable_double_by_name())["a"] = 1.0;
  (*proto.mutable_double_by_name())["c"] = 3.0;
  Maps value;
  ASSERT_TRUE(FromProto(proto, &value).ok());
  std::vector<std::string> keys;
  for (const std::pair<const std::string, double>& entry : value.double_by_name) {
    keys.push_back(entry.first);
  }
  EXPECT_EQ(keys, (std::vector<std::string>{"a", "b", "c"}));

  // The same keys update the map in place; other keys replace it, stale keys included.
  (*proto.mutable_double_by_name())["a"] = 10.0;
  ASSERT_TRUE(FromProto(proto, &value).ok());
  EXPECT_EQ(value.double_by_name.at("a"), 10.0);
  proto.mutable_double_by_name()->erase("b");
  (*proto.mutable_double_by_name())["d"] = 4.0;
  ASSERT_TRUE(FromProto(proto, &value).ok());
  EXPECT_EQ(value.double_by_name, (std::map<std::string, double>{{"a", 10.0}, {"c", 3.0}, {"d", 4.0}}));

  value.double_by_name.erase("d");
  value.double_by_name["e"] = 5.0;
  ToProto(value, &proto);
  EXPECT_EQ(proto.double_by_name().size(), 3u);
  EXPECT_EQ(proto.double_by_name().count("d"), 0u);
  EXPECT_EQ(proto.double_by_name().at("e"), 5.0);
}

TEST(RepeatedTest, ShorterAndLongerValuesResizeTheMessage) {
  RepeatedFields value;
  value.strings = {"a", "b", "c"};
  value.inners.resize(3);
  value.doubles = Eigen::VectorXd::LinSpaced(4, 0.0, 3.0);
  nproto_test::RepeatedFields proto;
  ToProto(value, &proto);
  EXPECT_EQ(proto.strings_size(), 3);
  EXPECT_EQ(proto.inners_size(), 3);
  ASSERT_EQ(proto.doubles_size(), 4);
  EXPECT_EQ(proto.doubles(3), 3.0);

  value.strings = {"z"};
  value.inners.resize(1);
  value.doubles.resize(0);
  ToProto(value, &proto);
  ASSERT_EQ(proto.strings_size(), 1);
  EXPECT_EQ(proto.strings(0), "z");
  EXPECT_EQ(proto.inners_size(), 1);
  EXPECT_EQ(proto.doubles_size(), 0);

  RepeatedFields back;
  back.strings = {"x", "y", "w", "v"};
  ASSERT_TRUE(FromProto(proto, &back).ok());
  EXPECT_TRUE(back == value);
}

// ---------------------------------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------------------------------

TEST(EnumTest, EveryValueConvertsBothWays) {
  for (const Severity severity : {Severity::kUnspecified, Severity::kLow, Severity::kHigh, Severity::kLevel10}) {
    nproto_test::Severity proto = nproto_test::SEVERITY_UNSPECIFIED;
    ToProto(severity, &proto);
    EXPECT_EQ(static_cast<int>(proto), static_cast<int>(severity));
    Severity back = Severity::kLow;
    ASSERT_TRUE(FromProto(proto, &back).ok());
    EXPECT_EQ(back, severity);
  }
  // The alias is the number of the value it aliases.
  nproto_test::Severity severe = nproto_test::SEVERITY_UNSPECIFIED;
  ToProto(Severity::kSevere, &severe);
  EXPECT_EQ(severe, nproto_test::SEVERITY_HIGH);
}

TEST(EnumTest, AnUndefinedNumberIsAnError) {
  Severity value = Severity::kLow;
  const absl::Status status = FromProto(static_cast<nproto_test::Severity>(99), &value);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(), "99 is not a value of nproto_test.Severity");
}

// The error of FromProto() names the field path down to the enum.
void expectError(const absl::Status& status, const std::string& message) {
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_EQ(status.message(), message);
}

TEST(ErrorTest, ErrorsNameTheFieldPath) {
  {
    nproto_test::Oneofs proto;
    proto.set_severity(static_cast<nproto_test::Severity>(99));
    Oneofs value;
    expectError(FromProto(proto, &value), "severity: 99 is not a value of nproto_test.Severity");
  }
  {
    nproto_test::RepeatedFields proto;
    proto.add_severities(nproto_test::SEVERITY_LOW);
    proto.add_severities(static_cast<nproto_test::Severity>(42));
    RepeatedFields value;
    expectError(FromProto(proto, &value), "severities[1]: 42 is not a value of nproto_test.Severity");
  }
  {
    nproto_test::RepeatedFields proto;
    proto.add_inners();
    proto.add_inners()->mutable_leaf()->set_kind(static_cast<nproto_test::Outer_Kind>(5));
    RepeatedFields value;
    expectError(FromProto(proto, &value), "inners[1].leaf.kind: 5 is not a value of nproto_test.Outer.Kind");
  }
  {
    nproto_test::Maps proto;
    (*proto.mutable_severity_by_id())[7] = static_cast<nproto_test::Severity>(13);
    Maps value;
    expectError(FromProto(proto, &value), "severity_by_id[7]: 13 is not a value of nproto_test.Severity");
  }
  {
    nproto_test::Outer proto;
    (*proto.mutable_inner_by_name())["a b"].set_kind(static_cast<nproto_test::Outer_Kind>(9));
    Outer value;
    expectError(FromProto(proto, &value), "inner_by_name[\"a b\"].kind: 9 is not a value of nproto_test.Outer.Kind");
  }
  {
    nproto_test::Optionals proto;
    proto.set_maybe_kind(static_cast<nproto_test::Outer_Kind>(8));
    Optionals value;
    expectError(FromProto(proto, &value), "maybe_kind: 8 is not a value of nproto_test.Outer.Kind");
  }
  {
    nproto_test::Imports proto;
    proto.add_points()->set_frame(static_cast<nproto_test::other::Point_Frame>(3));
    Imports value;
    expectError(FromProto(proto, &value), "points[0].frame: 3 is not a value of nproto_test.other.Point.Frame");
  }
}

}  // namespace
}  // namespace nproto::test
