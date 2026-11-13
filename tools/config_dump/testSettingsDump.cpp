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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <cstdlib>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "tools/config_dump/SettingsDump.h"
#include "tools/config_dump/ValueDump.h"

namespace ocs2::humanoid::config_dump {
namespace {

using ::testing::HasSubstr;

// The shipped Tron1 configuration, whose two wheels are rolling contacts.
constexpr char kTron1TaskFile[] =
    "robot_models/limx_tron1/tron1_centroidal_mpc/config/mpc/task.textproto";
constexpr char kTron1UrdfFile[] =
    "robot_models/limx_tron1/tron1_description/urdf/tron1.urdf";

/** `relativePath` in the test's runfiles. */
std::string runfilesPath(absl::string_view relativePath) {
  const char* absl_nullable directory = std::getenv("TEST_SRCDIR");
  return directory == nullptr
             ? std::string(relativePath)
             : absl::StrCat(directory, "/_main/", relativePath);
}

absl::StatusOr<ModelSettings> tron1Settings() {
  return ModelSettings::Create(runfilesPath(kTron1TaskFile),
                               runfilesPath(kTron1UrdfFile), "test_",
                               /*verbose=*/false);
}

std::string dumpOf(const ModelSettings& settings) {
  ValueDump dump;
  dumpModelSettings(dump, "model_settings", settings);
  return dump.text();
}

TEST(SettingsDumpTest, TheRollingContactsOfTron1AreDumped) {
  const absl::StatusOr<ModelSettings> settings = tron1Settings();
  ASSERT_TRUE(settings.ok()) << settings.status();
  const std::string text = dumpOf(*settings);
  EXPECT_THAT(
      text,
      HasSubstr(R"(model_settings.pointContacts[0].name = "wheel_L_contact")"));
  EXPECT_THAT(
      text,
      HasSubstr(
          R"(model_settings.pointContacts[1].parentJointName = "wheel_R_Joint")"));
  EXPECT_THAT(
      text,
      HasSubstr(
          "model_settings.pointContacts[0].posContactRtLinkEwrtLink.y = "));
  EXPECT_THAT(text,
              HasSubstr(absl::StrCat(
                  "model_settings.pointContacts[1].kind = ",
                  static_cast<int>(ModelSettings::ContactKind::kRolling))));
  EXPECT_THAT(text,
              HasSubstr("model_settings.pointContacts[0].wheelRadius = "));
  EXPECT_THAT(text, HasSubstr(R"("wheel_L_Joint")"));
  EXPECT_THAT(text, HasSubstr("model_settings.rollingContactJointNames"));
}

TEST(SettingsDumpTest, EveryValueOfARollingContactChangesTheDump) {
  const absl::StatusOr<ModelSettings> shipped = tron1Settings();
  ASSERT_TRUE(shipped.ok()) << shipped.status();
  const std::string base = dumpOf(*shipped);
  {
    absl::StatusOr<ModelSettings> changed = tron1Settings();
    ASSERT_TRUE(changed.ok()) << changed.status();
    changed->pointContacts[0].wheelRadius += 0.01;
    EXPECT_NE(dumpOf(*changed), base) << "wheelRadius";
  }
  {
    absl::StatusOr<ModelSettings> changed = tron1Settings();
    ASSERT_TRUE(changed.ok()) << changed.status();
    changed->pointContacts[1].posContactRtLinkEwrtLink.y() += 0.01;
    EXPECT_NE(dumpOf(*changed), base) << "posContactRtLinkEwrtLink";
  }
  {
    absl::StatusOr<ModelSettings> changed = tron1Settings();
    ASSERT_TRUE(changed.ok()) << changed.status();
    changed->pointContacts[0].kind = ModelSettings::ContactKind::kPoint;
    EXPECT_NE(dumpOf(*changed), base) << "kind";
  }
}

}  // namespace
}  // namespace ocs2::humanoid::config_dump
