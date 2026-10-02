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

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "absl/status/status.h"

#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {
namespace {

constexpr const char* kGaitFile = "humanoid_nmpc/humanoid_common_mpc/config/command/gait.yaml";

/** The shipped gait file in this test's runfiles, or empty when it is not there. */
std::string gaitFilePath() {
  std::vector<std::filesystem::path> roots;
  if (const char* srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
    roots.emplace_back(std::filesystem::path(srcDir) / "wb_humanoid_mpc");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / kGaitFile;
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

TEST(ModeSequenceTemplateTest, EveryShippedGaitIsValid) {
  // getGaitMap() loads every gait through loadModeSequenceTemplate(), which refuses one validate rejects. The `skip`
  // gait once had a switching time of 0.08 between 0.75 and 1.0 - a phase of -0.67 s - and loaded without complaint.
  const std::string gaitFile = gaitFilePath();
  ASSERT_FALSE(gaitFile.empty()) << kGaitFile << " is not in the runfiles";
  std::map<std::string, ModeSequenceTemplate> gaits;
  ASSERT_NO_THROW(gaits = getGaitMap(gaitFile, /*verbose=*/false));
  ASSERT_GT(gaits.size(), 10u) << "the gait list was not read";
  for (const std::pair<const std::string, ModeSequenceTemplate>& gait : gaits) {
    const absl::Status status = validateModeSequenceTemplate(gait.second, gait.first);
    EXPECT_TRUE(status.ok()) << status.message();
  }
}

TEST(ModeSequenceTemplateTest, TheDocumentedYamlLayoutLoads) {
  // The layouts that the comments of loadModeSequenceTemplate() and loadModeSchedule() show, read back by them: the
  // documentation describes a file the loaders accept.
  const std::string file = testing::TempDir() + "/documented_mode_sequence_layout.yaml";
  {
    std::ofstream out(file);
    out << "topicName:\n"
           "  modeSequence:\n"
           "    - LF\n"
           "    - RF\n"
           "  switchingTimes:\n"
           "    - 0.0\n"
           "    - 0.4\n"
           "    - 0.8\n"
           "scheduleName:\n"
           "  modeSequence:\n"
           "    - STANCE\n"
           "    - LF\n"
           "    - STANCE\n"
           "  eventTimes:\n"
           "    - 0.5\n"
           "    - 1.0\n";
  }
  const ModeSequenceTemplate modeSequenceTemplate = loadModeSequenceTemplate(file, "topicName", /*verbose=*/false);
  EXPECT_EQ(modeSequenceTemplate.modeSequence, (std::vector<size_t>{ModeNumber::LF, ModeNumber::RF}));
  EXPECT_EQ(modeSequenceTemplate.switchingTimes, (std::vector<scalar_t>{0.0, 0.4, 0.8}));

  const ModeSchedule modeSchedule = loadModeSchedule(file, "scheduleName", /*verbose=*/false);
  EXPECT_EQ(modeSchedule.modeSequence, (std::vector<size_t>{ModeNumber::STANCE, ModeNumber::LF, ModeNumber::STANCE}));
  EXPECT_EQ(modeSchedule.eventTimes, (std::vector<scalar_t>{0.5, 1.0}));
  std::remove(file.c_str());
}

TEST(ModeSequenceTemplateTest, SwitchingTimesOutOfOrderAreRejectedNamingTheKey) {
  const ModeSequenceTemplate skip({0.0, 0.25, 0.3, 0.5, 0.75, 0.08, 1.0},
                                  {ModeNumber::LF, ModeNumber::FLY, ModeNumber::LF, ModeNumber::RF, ModeNumber::FLY, ModeNumber::RF});
  const absl::Status status = validateModeSequenceTemplate(skip, "skip");
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(status.message().find("skip.switchingTimes"), absl::string_view::npos) << status.message();
  EXPECT_NE(status.message().find("entry 5"), absl::string_view::npos) << status.message();

  // Positive control: the same gait with 0.8 in its place is valid.
  const ModeSequenceTemplate fixed({0.0, 0.25, 0.3, 0.5, 0.75, 0.8, 1.0},
                                   {ModeNumber::LF, ModeNumber::FLY, ModeNumber::LF, ModeNumber::RF, ModeNumber::FLY, ModeNumber::RF});
  EXPECT_TRUE(validateModeSequenceTemplate(fixed, "skip").ok());
}

TEST(ModeSequenceTemplateTest, ARepeatedSwitchingTimeIsRejected) {
  // A zero-length mode is as meaningless as a negative one: the strict inequality is the property.
  const ModeSequenceTemplate repeated({0.0, 0.5, 0.5, 1.0}, {ModeNumber::LF, ModeNumber::STANCE, ModeNumber::RF});
  EXPECT_FALSE(validateModeSequenceTemplate(repeated, "repeated").ok());
}

TEST(ModeSequenceTemplateTest, AMismatchedLengthIsRejected) {
  const ModeSequenceTemplate tooFew({0.0, 1.0}, {ModeNumber::LF, ModeNumber::RF});
  const absl::Status status = validateModeSequenceTemplate(tooFew, "short");
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(status.message().find("short.modeSequence"), absl::string_view::npos) << status.message();
}

}  // namespace
}  // namespace ocs2::humanoid
