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

// The keyboard teleoperation's command limits from a typed reference file: the four limits and the base height are
// required, finite, and the limits positive, since the command is normalized by them.

#include <cstddef>
#include <iterator>
#include <limits>
#include <optional>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc_app/teleop/KeyboardVelocityCommand.h"
#include "humanoid_common_mpc_app/teleop/config/reference/KeyboardCommandLimitsFromConfig.h"
#include "humanoid_mpc_config/reference_file.nproto.h"

namespace ocs2::humanoid::teleop {
namespace {

using Field = std::optional<double> mpc_config::ReferenceFile::*absl_nonnull;

constexpr std::pair<absl::string_view, Field> kFields[] = {
    {"max_displacement_velocity_x", &mpc_config::ReferenceFile::max_displacement_velocity_x},
    {"max_displacement_velocity_y", &mpc_config::ReferenceFile::max_displacement_velocity_y},
    {"max_delta_pelvis_height", &mpc_config::ReferenceFile::max_delta_pelvis_height},
    {"max_rotation_velocity", &mpc_config::ReferenceFile::max_rotation_velocity},
    {"default_base_height", &mpc_config::ReferenceFile::default_base_height},
};

mpc_config::ReferenceFile limitsFile() {
  mpc_config::ReferenceFile file;
  file.max_displacement_velocity_x = 1.25;
  file.max_displacement_velocity_y = 0.25;
  file.max_delta_pelvis_height = 0.5;
  file.max_rotation_velocity = 0.75;
  file.default_base_height = 0.875;
  return file;
}

TEST(KeyboardCommandLimitsFromConfigTest, EachFieldReachesItsLimit) {
  const absl::StatusOr<KeyboardCommandLimits> limits = keyboardCommandLimitsFromConfig(limitsFile());
  ASSERT_TRUE(limits.ok()) << limits.status();
  EXPECT_EQ(limits->limits(0), 1.25);
  EXPECT_EQ(limits->limits(1), 0.25);
  EXPECT_EQ(limits->limits(2), 0.5);
  EXPECT_EQ(limits->limits(3), 0.75);
  EXPECT_EQ(limits->defaultBaseHeight, 0.875);
}

TEST(KeyboardCommandLimitsFromConfigTest, EachFieldIsRequiredAndFinite) {
  for (const std::pair<absl::string_view, Field>& field : kFields) {
    for (const std::optional<double> bad : {std::optional<double>(), std::optional<double>(std::numeric_limits<double>::quiet_NaN()),
                                            std::optional<double>(std::numeric_limits<double>::infinity())}) {
      mpc_config::ReferenceFile file = limitsFile();
      file.*field.second = bad;
      const absl::StatusOr<KeyboardCommandLimits> limits = keyboardCommandLimitsFromConfig(file);
      EXPECT_EQ(limits.status().code(), absl::StatusCode::kInvalidArgument) << field.first;
      EXPECT_TRUE(absl::StrContains(limits.status().message(), field.first)) << limits.status();
    }
  }
}

TEST(KeyboardCommandLimitsFromConfigTest, ALimitThatIsNotPositiveIsRefused) {
  for (size_t i = 0; i + 1 < std::size(kFields); ++i) {
    for (const double bad : {0.0, -0.5}) {
      mpc_config::ReferenceFile file = limitsFile();
      file.*kFields[i].second = bad;
      const absl::StatusOr<KeyboardCommandLimits> limits = keyboardCommandLimitsFromConfig(file);
      EXPECT_TRUE(absl::StrContains(limits.status().message(), "must be positive")) << limits.status();
    }
  }
  // The base height is a height, not a limit.
  mpc_config::ReferenceFile file = limitsFile();
  file.default_base_height = 0.0;
  EXPECT_TRUE(keyboardCommandLimitsFromConfig(file).ok());
}

}  // namespace
}  // namespace ocs2::humanoid::teleop
