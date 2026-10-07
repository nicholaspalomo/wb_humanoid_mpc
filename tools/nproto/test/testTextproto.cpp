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

// nproto's textproto helpers: a configuration file that matches its schema parses, and one that does not is an error
// naming the file, the line and the column; retired fields and a file of another message are answered from the schema.
// tools/nproto/test/test_nproto_textproto.py feeds the Python parser the same inputs and expects the same answers.

#include <cstdlib>
#include <fstream>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "nproto/Textproto.h"
#include "tools/nproto/test/ProtoTestValues.h"
#include "tools/nproto/test/defaults.pb.h"
#include "tools/nproto/test/maps.pb.h"
#include "tools/nproto/test/oneofs.nproto.pb.h"
#include "tools/nproto/test/repeated_fields.pb.h"
#include "tools/nproto/test/retired.pb.h"
#include "tools/nproto/test/scalars.nproto.pb.h"

namespace nproto::test {
namespace {

// The testdata file, a data dependency of the test, relative to the runfiles tree the test runs in.
constexpr char kScalarsFile[] = "tools/nproto/test/testdata/scalars.textproto";

std::string temporaryFile(const std::string& name, const std::string& contents) {
  const char* absl_nullable directory = std::getenv("TEST_TMPDIR");
  const std::string path = absl::StrCat(directory != nullptr ? directory : "/tmp", "/", name);
  std::ofstream file(path);
  file << contents;
  return path;
}

void expectInvalid(const absl::Status& status, const std::string& expected) {
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_TRUE(absl::StrContains(status.message(), expected)) << "'" << status.message() << "' lacks '" << expected << "'";
}

TEST(TextprotoTest, AFileThatMatchesTheSchemaParses) {
  const absl::StatusOr<nproto_test::Scalars> proto = ParseTextprotoFile<nproto_test::Scalars>(kScalarsFile);
  ASSERT_TRUE(proto.ok()) << proto.status();
  EXPECT_EQ(proto->double_value(), 1.5);
  EXPECT_EQ(proto->uint64_value(), 18000000000u);
  EXPECT_EQ(proto->string_value(), "a configuration value");
  EXPECT_EQ(proto->bytes_value(), std::string("\001\002"));

  const absl::StatusOr<Scalars> value = LoadTextprotoFile<Scalars, nproto_test::Scalars>(kScalarsFile);
  ASSERT_TRUE(value.ok()) << value.status();
  EXPECT_EQ(value->double_value, 1.5);
  EXPECT_EQ(value->float_value, -0.25f);
  EXPECT_EQ(value->int64_value, 9000000000);
  EXPECT_EQ(value->sint32_value, -3);
  EXPECT_TRUE(value->bool_value);
  EXPECT_EQ(value->string_value, "a configuration value");
}

TEST(TextprotoTest, AnUnknownFieldIsAnErrorWithItsLineAndColumn) {
  const absl::StatusOr<nproto_test::Scalars> proto =
      ParseTextproto<nproto_test::Scalars>("double_value: 1.5\n  doubel_value: 2\n", "config.textproto");
  expectInvalid(proto.status(), "config.textproto:2:3: ");
  expectInvalid(proto.status(), "doubel_value");
  // The position is the field name's also inside a message, after comments, and at the end of the text.
  expectInvalid(ParseTextproto<nproto_test::Oneofs>("# A comment.\nscalars { int32_value: 1 bogus: 2 }\n", "nested.textproto").status(),
                R"(nested.textproto:2:26: Message type "nproto_test.Scalars" has no field named "bogus".)");
  expectInvalid(ParseTextproto<nproto_test::Scalars>("int32_value: 1\nbogus", "end.textproto").status(), "end.textproto:2:1: ");
  expectInvalid(ParseTextproto<nproto_test::Oneofs>("scalars {\n  bogus {}\n}\n", "message.textproto").status(), "message.textproto:2:3: ");
}

TEST(TextprotoTest, AValueOfTheWrongTypeIsAnErrorWithItsLineAndColumn) {
  expectInvalid(ParseTextproto<nproto_test::Scalars>("int32_value: \"seven\"\n", "config.textproto").status(), "config.textproto:1:14: ");
  expectInvalid(ParseTextproto<nproto_test::Scalars>("\nint32_value: 3000000000\n", "config.textproto").status(),
                "config.textproto:2:14: ");
  expectInvalid(ParseTextproto<nproto_test::Scalars>("bool_value: maybe\n", "config.textproto").status(), "config.textproto:1:13: ");
  expectInvalid(ParseTextproto<nproto_test::Oneofs>("severity: SEVERITY_NOPE\n", "config.textproto").status(), "config.textproto:1:11: ");
}

TEST(TextprotoTest, OtherMistakesAreErrorsToo) {
  // A non-repeated field given twice.
  expectInvalid(ParseTextproto<nproto_test::Scalars>("int32_value: 1\nint32_value: 2\n", "twice.textproto").status(),
                "twice.textproto:2:1: ");
  // Two alternatives of one oneof.
  expectInvalid(ParseTextproto<nproto_test::Oneofs>("radius: 1\n  side: 2\n", "oneof.textproto").status(), "oneof.textproto:2:3: ");
  // A syntax error.
  expectInvalid(ParseTextproto<nproto_test::Scalars>("int32_value 1\n", "syntax.textproto").status(), "syntax.textproto:1:");
  // A missing proto2 required field has no position.
  expectInvalid(ParseTextproto<nproto_test::Defaults>("count: 1\n", "required.textproto").status(), "required.textproto: ");
  expectInvalid(ParseTextproto<nproto_test::Defaults>("count: 1\n", "required.textproto").status(), "required_count");
  // A field number instead of a name.
  expectInvalid(ParseTextproto<nproto_test::Scalars>("3: 1\n", "number.textproto").status(), "number.textproto:1:");
}

TEST(RetiredFieldTest, ARetiredNameIsAnsweredWithItsReplacement) {
  const absl::StatusOr<nproto_test::Retired> retired =
      ParseTextproto<nproto_test::Retired>("switches: \"a\"\nuse_old_switch: true\n", "retired.textproto");
  EXPECT_EQ(retired.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(retired.status().message(), "retired.textproto:2:1: 'use_old_switch' is retired: list old_switch under switches");
}

TEST(RetiredFieldTest, EverySpellingOfTheNameHitsAndThePositionIsKept) {
  // camelCase of a snake_case entry, and snake_case of a camelCase entry.
  expectInvalid(ParseTextproto<nproto_test::Retired>("  useOldSwitch: 1\n", "camel.textproto").status(),
                "camel.textproto:1:3: 'useOldSwitch' is retired: list old_switch under switches");
  expectInvalid(ParseTextproto<nproto_test::Retired>("step_width: 0.1 legacy_gain: 2\n", "snake.textproto").status(),
                "snake.textproto:1:17: 'legacy_gain' is retired: set block.gain instead");
}

TEST(RetiredFieldTest, AnUnknownNameKeepsThePlainError) {
  const absl::StatusOr<nproto_test::Retired> retired = ParseTextproto<nproto_test::Retired>("bogus: 1\n", "plain.textproto");
  EXPECT_EQ(retired.status().message(), R"(plain.textproto:1:1: Message type "nproto_test.Retired" has no field named "bogus".)");
}

TEST(RetiredFieldTest, ASnakeCasedFieldIsSuggested) {
  const absl::StatusOr<nproto_test::Retired> retired = ParseTextproto<nproto_test::Retired>("stepWidth: 0.2\n", "suggest.textproto");
  EXPECT_EQ(retired.status().message(),
            R"(suggest.textproto:1:1: Message type "nproto_test.Retired" has no field named "stepWidth". Did you mean "step_width"?)");
}

TEST(RetiredFieldTest, TheLayoutHintEndsEveryUnknownFieldOfItsMessage) {
  const absl::StatusOr<nproto_test::Retired> retired =
      ParseTextproto<nproto_test::Retired>("block {\n  flat_gain: 1\n}\n", "layout.textproto");
  EXPECT_EQ(retired.status().message(), R"(layout.textproto:2:3: Message type "nproto_test.Retired.Block" has no field named "flat_gain". )"
                                        "(the flat keys moved into block, tools/nproto/README.md)");
}

TEST(HeaderTest, AFileOfAnotherMessageIsRefused) {
  const std::string text = "# proto-file: tools/nproto/test/scalars.proto\n# proto-message: nproto_test.Scalars\n\nint32_value: 1\n";
  const absl::StatusOr<nproto_test::Oneofs> wrong = ParseTextproto<nproto_test::Oneofs>(text, "scalars.textproto");
  EXPECT_EQ(wrong.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(wrong.status().message(),
            "scalars.textproto:2:1: scalars.textproto is a nproto_test.Scalars (its '# proto-message:' header), not a nproto_test.Oneofs");
  // The message the header names parses, and so does a file without a header.
  EXPECT_TRUE(ParseTextproto<nproto_test::Scalars>(text, "scalars.textproto").ok());
  EXPECT_TRUE(ParseTextproto<nproto_test::Oneofs>("radius: 1\n", "no_header.textproto").ok());
}

TEST(HeaderTest, OnlyTheLeadingCommentBlockIsTheHeader) {
  // After the first field, a comment that looks like a header is a comment.
  const absl::StatusOr<nproto_test::Oneofs> later =
      ParseTextproto<nproto_test::Oneofs>("# A comment.\n\nradius: 1\n# proto-message: nproto_test.Scalars\n", "later.textproto");
  EXPECT_TRUE(later.ok()) << later.status();
}

TEST(TextprotoTest, AMissingFileIsNotFound) {
  const std::string path = "tools/nproto/test/testdata/no_such_file.textproto";
  const absl::StatusOr<nproto_test::Scalars> proto = ParseTextprotoFile<nproto_test::Scalars>(path);
  EXPECT_EQ(proto.status().code(), absl::StatusCode::kNotFound) << proto.status();
  EXPECT_TRUE(absl::StrContains(proto.status().message(), path)) << proto.status();
  const absl::StatusOr<Scalars> value = LoadTextprotoFile<Scalars, nproto_test::Scalars>(path);
  EXPECT_EQ(value.status().code(), absl::StatusCode::kNotFound) << value.status();
}

TEST(TextprotoTest, AConversionErrorNamesTheFile) {
  // The text format accepts the number of an open enum that the enum does not define; the struct cannot hold it.
  const std::string path = temporaryFile("undefined_enum.textproto", "severity: 99\n");
  const absl::StatusOr<Oneofs> value = LoadTextprotoFile<Oneofs, nproto_test::Oneofs>(path);
  EXPECT_EQ(value.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(value.status().message(), absl::StrCat(path, ": severity: 99 is not a value of nproto_test.Severity"));
}

TEST(TextprotoTest, ParsingReplacesTheMessage) {
  nproto_test::Scalars message;
  message.set_int64_value(5);
  ASSERT_TRUE(ParseTextprotoInto("int32_value: 1", "inline", &message).ok());
  EXPECT_EQ(message.int32_value(), 1);
  EXPECT_EQ(message.int64_value(), 0);
}

TEST(TextprotoTest, WrittenTextprotosParseBackToTheSameMessage) {
  nproto_test::Maps maps;
  test_support::FillWithTestValues(test_support::TestValueOptions{.seed = 4, .repeatedSize = 3}, &maps);
  const absl::StatusOr<nproto_test::Maps> mapsBack = ParseTextproto<nproto_test::Maps>(WriteTextproto(maps), "maps");
  ASSERT_TRUE(mapsBack.ok()) << mapsBack.status();
  EXPECT_TRUE(test_support::ProtoEquals(maps, *mapsBack));

  nproto_test::RepeatedFields repeated;
  test_support::FillWithTestValues(test_support::TestValueOptions{.seed = 5, .repeatedSize = 3}, &repeated);
  const std::string text = WriteTextproto(repeated);
  // Repeated numbers on one line, as configuration files write vectors.
  EXPECT_TRUE(absl::StrContains(text, "doubles: [")) << text;
  const absl::StatusOr<nproto_test::RepeatedFields> repeatedBack = ParseTextproto<nproto_test::RepeatedFields>(text, "repeated");
  ASSERT_TRUE(repeatedBack.ok()) << repeatedBack.status();
  EXPECT_TRUE(test_support::ProtoEquals(repeated, *repeatedBack));
}

}  // namespace
}  // namespace nproto::test
