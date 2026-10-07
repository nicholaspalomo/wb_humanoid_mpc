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

#include <filesystem>
#include <limits>
#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_replace.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "support/AtlasReferenceStack.h"
#include "support/TypedConfigFiles.h"

/*
 * ProceduralMpcMotionManager::Create() and its command-limit reloads, on the DRC Atlas references: a gait file without a
 * gait the velocity command selects and a reference file whose values do not parse are refused by Status, naming the
 * file, and a reloaded limit that is not a number is refused and keeps the running one. The constructor used to throw
 * both, and the reloads threw what the parameter updater then caught.
 */

namespace ocs2::humanoid {
namespace {

/** `text` written to the file `name` in the test's temporary directory; its path. */
std::string writeTempFile(const std::string& name, const std::string& text) {
  const std::string file = (std::filesystem::path(::testing::TempDir()) / name).string();
  EXPECT_TRUE(writeTextFile(file, text).ok()) << file;
  return file;
}

absl::StatusOr<std::unique_ptr<ProceduralMpcMotionManager>> create(AtlasReferenceStack& stack,
                                                                   const std::string& gaitFile,
                                                                   const std::string& referenceFile) {
  CentroidalMpcTargetTrajectoriesCalculator* absl_nonnull calculator = &stack.targetCalculator();
  return ProceduralMpcMotionManager::Create(
      gaitFile, referenceFile, stack.referenceManagerPtr(), stack.model(),
      [calculator](const vector4_t& velocityTarget, scalar_t initTime, scalar_t /*finalTime*/, const vector_t& initState) {
        return calculator->commandedVelocityToTargetTrajectories(velocityTarget, initTime, initState);
      });
}

TEST(ProceduralMpcMotionManagerCreate, BuildsTheManagerOfTheShippedFiles) {
  AtlasReferenceStack stack;
  const absl::StatusOr<std::unique_ptr<ProceduralMpcMotionManager>> manager = create(stack, stack.gaitFile(), stack.referenceFile());
  ASSERT_TRUE(manager.ok()) << manager.status();
  EXPECT_EQ((*manager)->getCurrentGaitCommand(), "stance");
}

TEST(ProceduralMpcMotionManagerCreate, AGaitFileWithoutAGaitTheCommandSelectsIsRefusedNamingIt) {
  AtlasReferenceStack stack;
  const std::string stanceOnly = writeTempFile(
      "stance_only_gait.textproto",
      "gait_list: \"stance\"\ngaits { name: \"stance\" mode_sequence: \"STANCE\" switching_times: 0.0 switching_times: 0.5 }\n");
  const absl::StatusOr<std::unique_ptr<ProceduralMpcMotionManager>> manager = create(stack, stanceOnly, stack.referenceFile());
  EXPECT_EQ(manager.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(manager.status().message(), "has no gait")) << manager.status();
  EXPECT_TRUE(absl::StrContains(manager.status().message(), stanceOnly)) << manager.status();
}

TEST(ProceduralMpcMotionManagerCreate, AReferenceFileWhoseLimitDoesNotParseIsRefusedNamingTheFile) {
  AtlasReferenceStack stack;
  mpc_config::ReferenceFile reference = stack.config().reference;
  reference.max_displacement_velocity_x = 0.123;
  const std::string text = referenceFileText(reference);
  const std::string unparsable = absl::StrReplaceAll(text, {{"max_displacement_velocity_x: 0.123", "max_displacement_velocity_x: fast"}});
  ASSERT_NE(unparsable, text) << "the reference file's text no longer writes max_displacement_velocity_x as expected";
  const std::string broken = writeTempFile("broken_limit_reference.textproto", unparsable);
  const absl::StatusOr<std::unique_ptr<ProceduralMpcMotionManager>> manager = create(stack, stack.gaitFile(), broken);
  EXPECT_EQ(manager.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(manager.status().message(), broken)) << manager.status();
}

TEST(ProceduralMpcMotionManagerReload, AnAccelerationLimitThatIsNotANumberIsRefusedAndKeepsTheRamp) {
  AtlasReferenceStack stack;
  const scalar_t before = stack.motionManager().getMaxLinearAcceleration();
  // A reload converts the file first (MpcParameterUpdaterModule), which refuses the value by its field, so that nothing
  // of the file is applied ...
  mpc_config::ReferenceFile reference = stack.config().reference;
  reference.max_linear_acceleration = std::numeric_limits<scalar_t>::quiet_NaN();
  const absl::StatusOr<ReferenceSettings> converted = referenceSettingsFromConfig(reference);
  EXPECT_EQ(converted.status().code(), absl::StatusCode::kInvalidArgument) << converted.status();
  EXPECT_TRUE(absl::StrContains(converted.status().message(), "max_linear_acceleration")) << converted.status();
  // ... and the manager refuses the limit itself when it is handed one, keeping the ramp it runs.
  absl::StatusOr<ReferenceSettings> settings = referenceSettingsFromConfig(stack.config().reference);
  ASSERT_TRUE(settings.ok()) << settings.status();
  settings->maxLinearAcceleration = std::numeric_limits<scalar_t>::quiet_NaN();
  const absl::Status applied = stack.motionManager().applyCommandLimits(*settings);
  EXPECT_EQ(applied.code(), absl::StatusCode::kInvalidArgument) << applied;
  EXPECT_EQ(stack.motionManager().getMaxLinearAcceleration(), before) << "a refused reload switched the ramp off";
}

}  // namespace
}  // namespace ocs2::humanoid
