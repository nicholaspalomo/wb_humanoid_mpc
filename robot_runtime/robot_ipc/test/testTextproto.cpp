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

// The strict textproto parser (src/Textproto.h): a valid file fills the message, and every kind of malformed file is
// an InvalidArgument error that names the file, the line and the column, a deprecated field included.

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "robot_ipc_test/test_settings.pb.h"
#include "robot_runtime/robot_ipc/src/Textproto.h"

namespace robot::ipc {
namespace {

using robot_ipc_test::TestSettings;

constexpr char kSource[] = "settings.textproto";

// A directory the test may write to: Bazel's TEST_TMPDIR, or the working directory outside Bazel.
std::string temporaryPath(const std::string& name) {
  const char* directory = std::getenv("TEST_TMPDIR");
  return directory == nullptr ? name : absl::StrCat(directory, "/", name);
}

TEST(TextprotoTest, AValidFileFillsTheMessageAfterClearingIt) {
  TestSettings settings;
  settings.set_name("stale");
  settings.add_events()->set_name("stale");
  const absl::Status status = parseTextproto(R"(
# A comment.
events { name: "first" }
events { name: "second" }
)",
                                             kSource, settings);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_TRUE(settings.name().empty());
  ASSERT_EQ(settings.events_size(), 2);
  EXPECT_EQ(settings.events(0).name(), "first");
  EXPECT_EQ(settings.events(1).name(), "second");
}

struct Malformed {
  std::string description;
  std::string text;
  // "<source>:<line>:", with the 1-based line (the column a parser reports for one problem differs between C++ and
  // Python, so the tests pin only that there is one).
  std::string location;
  std::string fragment;
};

class MalformedTextprotoTest : public ::testing::TestWithParam<Malformed> {};

TEST_P(MalformedTextprotoTest, IsAnErrorNamingTheFileLineAndColumn) {
  const Malformed& file = GetParam();
  TestSettings settings;
  const absl::Status status = parseTextproto(file.text, kSource, settings);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << file.description << ": " << status;
  ASSERT_TRUE(absl::StartsWith(status.message(), file.location)) << file.description << ": " << status;
  // Then the column and the problem.
  const absl::string_view rest = status.message().substr(file.location.size());
  const size_t colon = rest.find(':');
  int column = 0;
  ASSERT_NE(colon, absl::string_view::npos) << status;
  EXPECT_TRUE(absl::SimpleAtoi(rest.substr(0, colon), &column) && column >= 1) << file.description << ": " << status;
  EXPECT_TRUE(absl::StrContains(rest.substr(colon), file.fragment)) << file.description << ": " << status;
}

INSTANTIATE_TEST_SUITE_P(Rejections,
                         MalformedTextprotoTest,
                         ::testing::Values(Malformed{"an unknown field", "name: \"a\"\nnmae: \"b\"", "settings.textproto:2:", "nmae"},
                                           Malformed{"an unknown nested field", "events {\n  nmae: \"b\"\n}",
                                                     "settings.textproto:2:", "nmae"},
                                           Malformed{"a field given twice", "name: \"a\"\nname: \"b\"", "settings.textproto:2:", "name"},
                                           Malformed{"a deprecated field", "name: \"a\"\nretired: 1", "settings.textproto:2:", "retired"},
                                           Malformed{"a string for a message", "events: \"b\"", "settings.textproto:1:", ""},
                                           Malformed{"an unquoted string", "name: a b", "settings.textproto:1:", ""},
                                           Malformed{"an unclosed message", "events {", "settings.textproto:1:", ""}));

TEST(TextprotoTest, LoadingNamesThePathAndAMissingFileIsNotFound) {
  TestSettings settings;
  const absl::Status missing = loadTextproto(temporaryPath("absent.textproto"), settings);
  EXPECT_EQ(missing.code(), absl::StatusCode::kNotFound) << missing;
  EXPECT_TRUE(absl::StrContains(missing.message(), "absent.textproto")) << missing;

  const std::string path = temporaryPath("settings.textproto");
  {
    std::ofstream file(path);
    file << "name: \"loaded\"\nevents { nmae: \"x\" }\n";
  }
  const absl::Status malformed = loadTextproto(path, settings);
  EXPECT_EQ(malformed.code(), absl::StatusCode::kInvalidArgument) << malformed;
  EXPECT_TRUE(absl::StartsWith(malformed.message(), absl::StrCat(path, ":2:"))) << malformed;

  {
    std::ofstream file(path);
    file << "name: \"loaded\"\n";
  }
  const absl::Status loaded = loadTextproto(path, settings);
  ASSERT_TRUE(loaded.ok()) << loaded;
  EXPECT_EQ(settings.name(), "loaded");
}

}  // namespace
}  // namespace robot::ipc
