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

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"

#include "humanoid_mpc_validation/io/GoldenIo.h"
#include "humanoid_mpc_validation/io/JsonValue.h"
#include "humanoid_mpc_validation/io/Sha256.h"

/*
 * The I/O of the measurement infrastructure: SHA-256 against the FIPS 180-4 test vectors, JSON documents that read back
 * to the same values, and golden files whose doubles read back bit for bit with their provenance.
 */

namespace ocs2::humanoid::validation {
namespace {

std::string temporaryPath(const std::string& name) {
  const char* tmp = std::getenv("TEST_TMPDIR");
  return (std::filesystem::path(tmp != nullptr ? tmp : "/tmp") / name).string();
}

bool sameBits(double a, double b) {
  uint64_t bitsA = 0;
  uint64_t bitsB = 0;
  std::memcpy(&bitsA, &a, sizeof(a));
  std::memcpy(&bitsB, &b, sizeof(b));
  return bitsA == bitsB;
}

TEST(Sha256, MatchesTheFipsTestVectors) {
  EXPECT_EQ(sha256Hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(sha256Hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  EXPECT_EQ(sha256Hex(std::string(1000000, 'a')), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  // Lengths around the padding boundary (55, 56 and 64 bytes) take one, two and two blocks.
  EXPECT_NE(sha256Hex(std::string(55, 'x')), sha256Hex(std::string(56, 'x')));
  EXPECT_EQ(sha256Hex(std::string(64, 'x')).size(), 64u);
}

TEST(Sha256, OfAFileIsOfItsBytes) {
  const std::string path = temporaryPath("sha256_file.bin");
  const char raw[] = "line one\nline two\0with a zero byte";
  const std::string contents(raw, sizeof(raw) - 1);
  {
    std::ofstream file(path, std::ios::binary);
    file << contents;
  }
  const absl::StatusOr<std::string> hash = sha256HexOfFile(path);
  ASSERT_TRUE(hash.ok()) << hash.status();
  EXPECT_EQ(*hash, sha256Hex(contents));
  EXPECT_EQ(sha256HexOfFile(temporaryPath("does_not_exist")).status().code(), absl::StatusCode::kNotFound);
}

TEST(JsonValue, ADocumentReadsBackToTheSameValues) {
  JsonValue document = JsonValue::object();
  document.set("name", JsonValue::string("quote \" backslash \\ newline \n tab \t control \x01 unicode \xc3\xa9"));
  document.set("flag", JsonValue::boolean(true));
  document.set("nothing", JsonValue());
  JsonValue& nested = document.set("nested", JsonValue::object());
  nested.set("pi", JsonValue::number(M_PI));
  nested.set("empty", JsonValue::object());
  JsonValue& list = document.set("list", JsonValue::array());
  list.append(JsonValue::number(-0.0));
  list.append(JsonValue::number(1e-300));
  list.append(JsonValue::number(123456789012345.0));
  JsonValue objects = JsonValue::array();
  objects.append(JsonValue::object()).set("a", JsonValue::number(1.0));
  objects.append(JsonValue::array());
  document.set("objects", objects);

  const absl::StatusOr<JsonValue> parsed = JsonValue::parse(document.serialize());
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  EXPECT_EQ(*parsed, document);
  EXPECT_EQ(parsed->serialize(), document.serialize()) << "writing is deterministic";
  EXPECT_EQ(parsed->findPath("nested.pi")->asNumber(), M_PI);
  EXPECT_EQ(parsed->findPath("nested.missing"), nullptr);
  EXPECT_EQ(parsed->keyAt(0), "name") << "members keep their insertion order";
}

TEST(JsonValue, NumbersReadBackBitForBitWithTheFewestDigits) {
  std::mt19937_64 generator(42);
  std::uniform_int_distribution<uint64_t> bits;
  for (int i = 0; i < 20000; ++i) {
    uint64_t raw = bits(generator);
    double value = 0.0;
    std::memcpy(&value, &raw, sizeof(value));
    if (!std::isfinite(value)) continue;
    const std::string text = JsonValue::number(value).serialize();
    const absl::StatusOr<JsonValue> parsed = JsonValue::parse(text);
    ASSERT_TRUE(parsed.ok()) << text;
    ASSERT_TRUE(sameBits(parsed->asNumber(), value)) << text;
  }
  EXPECT_EQ(JsonValue::number(0.1).serialize(), "0.1");
  EXPECT_EQ(JsonValue::number(3.0).serialize(), "3");
}

TEST(JsonValue, ANumberThatIsNotFiniteIsWrittenAsNull) {
  EXPECT_EQ(JsonValue::number(std::numeric_limits<double>::quiet_NaN()).serialize(), "null");
  EXPECT_EQ(JsonValue::number(std::numeric_limits<double>::infinity()).serialize(), "null");
  EXPECT_EQ(JsonValue::optionalNumber(std::nullopt).serialize(), "null");
}

TEST(JsonValue, SetReplacesAMemberInPlace) {
  JsonValue object = JsonValue::object();
  object.set("a", JsonValue::number(1.0));
  object.set("b", JsonValue::number(2.0));
  object.set("a", JsonValue::number(3.0));
  ASSERT_EQ(object.size(), 2u);
  EXPECT_EQ(object.keyAt(0), "a");
  EXPECT_EQ(object.valueAt(0).asNumber(), 3.0);
}

TEST(JsonValue, MalformedDocumentsAreRejectedWithTheirOffset) {
  for (const std::string& text :
       {std::string("{\"a\": 1,}"), std::string("[1, 2"), std::string("{\"a\": 1, \"a\": 2}"), std::string("01"),
        std::string("\"unterminated"), std::string("{} extra"), std::string("tru"), std::string("1."), std::string("\"\\q\"")}) {
    const absl::StatusOr<JsonValue> parsed = JsonValue::parse(text);
    ASSERT_FALSE(parsed.ok()) << text;
    EXPECT_EQ(parsed.status().code(), absl::StatusCode::kInvalidArgument) << text;
    EXPECT_TRUE(absl::StrContains(parsed.status().message(), "at byte")) << parsed.status();
  }
  const absl::StatusOr<JsonValue> surrogate = JsonValue::parse("\"\\ud83d\\ude00\"");
  ASSERT_TRUE(surrogate.ok()) << surrogate.status();
  EXPECT_EQ(surrogate->asString(), "\xf0\x9f\x98\x80");
}

TEST(JsonValue, AFileReadsBackToTheValueItWasWrittenFrom) {
  JsonValue document = JsonValue::object();
  document.set("x", JsonValue::number(2.5));
  const std::string path = temporaryPath("json/nested/value.json");
  ASSERT_TRUE(writeJsonFile(path, document).ok());
  const absl::StatusOr<JsonValue> read = readJsonFile(path);
  ASSERT_TRUE(read.ok()) << read.status();
  EXPECT_EQ(*read, document);
  std::ifstream file(path);
  const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(absl::EndsWith(text, "}\n")) << "a final newline, for the repository's lint";
}

GoldenFile makeGolden() {
  GoldenFile golden;
  golden.provenance.gitCommit = "c331ddd";
  golden.provenance.worktreeState = "dirty: diff sha256 0123";
  golden.provenance.configurationHashes = {{"robot_models/a/task.yaml", sha256Hex("task")}, {"path with spaces.yaml", sha256Hex("x")}};
  golden.provenance.notes = {{"robot", "unitree_g1"}, {"machine", "a CPU: 8 cores"}};
  golden_matrix_t special(2, 4);
  special << std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
      -0.0, std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::max(), std::nextafter(1.0, 2.0), -1e-310;
  golden.entries.push_back({"special_values", special});
  golden.entries.push_back({"random", golden_matrix_t::Random(7, 5)});
  golden.entries.push_back({"no_rows", golden_matrix_t(0, 3)});
  golden.entries.push_back({"no_columns", golden_matrix_t(4, 0)});
  return golden;
}

TEST(GoldenIo, AGoldenFileReadsBackBitForBitWithItsProvenance) {
  const GoldenFile golden = makeGolden();
  const absl::StatusOr<std::string> text = formatGoldenFile(golden);
  ASSERT_TRUE(text.ok()) << text.status();
  const absl::StatusOr<GoldenFile> parsed = parseGoldenFile(*text);
  ASSERT_TRUE(parsed.ok()) << parsed.status();

  EXPECT_EQ(parsed->provenance.gitCommit, golden.provenance.gitCommit);
  EXPECT_EQ(parsed->provenance.worktreeState, golden.provenance.worktreeState);
  EXPECT_EQ(parsed->provenance.configurationHashes, golden.provenance.configurationHashes);
  EXPECT_EQ(parsed->provenance.notes, golden.provenance.notes);
  ASSERT_EQ(parsed->entries.size(), golden.entries.size());
  for (size_t i = 0; i < golden.entries.size(); ++i) {
    const GoldenEntry& expected = golden.entries[i];
    const GoldenEntry& actual = parsed->entries[i];
    EXPECT_EQ(actual.label, expected.label);
    ASSERT_EQ(actual.value.rows(), expected.value.rows()) << expected.label;
    ASSERT_EQ(actual.value.cols(), expected.value.cols()) << expected.label;
    for (Eigen::Index row = 0; row < expected.value.rows(); ++row) {
      for (Eigen::Index col = 0; col < expected.value.cols(); ++col) {
        const double a = actual.value(row, col);
        const double b = expected.value(row, col);
        EXPECT_TRUE(std::isnan(b) ? std::isnan(a) : sameBits(a, b)) << expected.label << "(" << row << ", " << col << ")";
      }
    }
  }
  EXPECT_NE(parsed->find("random"), nullptr);
  EXPECT_EQ(parsed->find("absent"), nullptr);
  // Formatting what was parsed gives the same text: the format is canonical.
  const absl::StatusOr<std::string> again = formatGoldenFile(*parsed);
  ASSERT_TRUE(again.ok());
  EXPECT_EQ(*again, *text);
}

TEST(GoldenIo, AFileRoundTripsAndItsProvenanceHashesTheConfiguration) {
  const std::string configuration = temporaryPath("golden_config.yaml");
  {
    std::ofstream file(configuration);
    file << "key: 1\n";
  }
  const absl::StatusOr<GoldenProvenance> provenance = makeGoldenProvenance("abc123", "clean", {configuration});
  ASSERT_TRUE(provenance.ok()) << provenance.status();
  ASSERT_EQ(provenance->configurationHashes.size(), 1u);
  EXPECT_EQ(provenance->configurationHashes[0].second, sha256Hex("key: 1\n"));
  EXPECT_EQ(makeGoldenProvenance("abc123", "clean", {temporaryPath("missing.yaml")}).status().code(), absl::StatusCode::kNotFound);

  GoldenFile golden = makeGolden();
  golden.provenance = *provenance;
  const std::string path = temporaryPath("golden/dir/file.txt");
  ASSERT_TRUE(writeGoldenFile(path, golden).ok());
  const absl::StatusOr<GoldenFile> read = readGoldenFile(path);
  ASSERT_TRUE(read.ok()) << read.status();
  EXPECT_EQ(read->provenance.configurationHashes, golden.provenance.configurationHashes);
  EXPECT_EQ(read->entries.size(), golden.entries.size());
  EXPECT_EQ(readGoldenFile(temporaryPath("absent.txt")).status().code(), absl::StatusCode::kNotFound);
}

TEST(GoldenIo, InvalidLabelsAndNotesAreRejected) {
  GoldenFile spaced;
  spaced.entries.push_back({"two words", golden_matrix_t::Zero(1, 1)});
  EXPECT_EQ(formatGoldenFile(spaced).status().code(), absl::StatusCode::kInvalidArgument);
  GoldenFile repeated;
  repeated.entries.push_back({"same", golden_matrix_t::Zero(1, 1)});
  repeated.entries.push_back({"same", golden_matrix_t::Zero(1, 1)});
  EXPECT_EQ(formatGoldenFile(repeated).status().code(), absl::StatusCode::kInvalidArgument);
  GoldenFile badNote;
  badNote.provenance.notes = {{"key:colon", "value"}};
  EXPECT_EQ(formatGoldenFile(badNote).status().code(), absl::StatusCode::kInvalidArgument);
  GoldenFile multiline;
  multiline.provenance.gitCommit = "a\nb";
  EXPECT_EQ(formatGoldenFile(multiline).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(GoldenIo, MalformedTextIsRejectedNamingTheLine) {
  const std::string header = "# wb_humanoid_mpc golden v1\n";
  for (const std::string& text :
       {std::string("not a golden\n"), header + "matrix m 2 2\n1 2\n", header + "matrix m 1 2\n1 x\n", header + "matrix m 1 2\n1 2 3\n",
        header + "matrix m two 2\n", header + "garbage\n", header + "matrix m 1 1\n1\nmatrix m 1 1\n2\n"}) {
    const absl::StatusOr<GoldenFile> parsed = parseGoldenFile(text);
    ASSERT_FALSE(parsed.ok()) << text;
    EXPECT_EQ(parsed.status().code(), absl::StatusCode::kInvalidArgument) << text;
    EXPECT_TRUE(absl::StrContains(parsed.status().message(), "line ")) << parsed.status();
  }
}

}  // namespace
}  // namespace ocs2::humanoid::validation
