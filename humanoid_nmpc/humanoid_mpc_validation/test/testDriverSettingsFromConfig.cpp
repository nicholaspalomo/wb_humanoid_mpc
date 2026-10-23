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

#include <array>
#include <optional>
#include <string>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/robot/ControllerSideSettingsFromConfig.h"
#include "humanoid_common_mpc/contact/ContactWrenchGate.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_validation/closed_loop/DriverSettingsFromConfig.h"
#include "humanoid_mpc_validation/closed_loop/RobotConfiguration.h"

/*
 * What the closed-loop driver reads of the configuration files itself (DriverSettingsFromConfig.h): the GUI's command
 * scaling of the reference file, and the contact wrench gate every configuration's task file sets
 * (wholeFileContactWrenchGate(), tested with the conversion in humanoid_common_mpc).
 */

namespace ocs2::humanoid::validation {
namespace {

/** A reference file that gives every value referenceSettingsFromConfig() requires, each a value of its own. */
mpc_config::ReferenceFile completeReferenceFile() {
  mpc_config::ReferenceFile file;
  file.target_displacement_velocity = 0.45;
  file.target_rotation_velocity = 0.55;
  file.max_displacement_velocity_x = 1.1;
  file.max_displacement_velocity_y = 0.35;
  file.max_delta_pelvis_height = 0.2;
  file.max_rotation_velocity = 0.9;
  file.target_joint_state_interpolation_time_constant = 0.3;
  file.default_base_height = 0.85;
  return file;
}

/** A field of the reference file that the GUI's command scaling reads. */
struct ScalingField {
  std::string name;
  std::optional<double> mpc_config::ReferenceFile::*absl_nonnull member;
};

/** The command limits, in the order of GuiCommandScaling::commandLimits. */
std::array<ScalingField, 3> commandLimitFields() {
  return {{
      {.name = "max_displacement_velocity_x", .member = &mpc_config::ReferenceFile::max_displacement_velocity_x},
      {.name = "max_displacement_velocity_y", .member = &mpc_config::ReferenceFile::max_displacement_velocity_y},
      {.name = "max_rotation_velocity", .member = &mpc_config::ReferenceFile::max_rotation_velocity},
  }};
}

/** The scaling of `file`, checked against the fields it comes from. */
void expectScalingOfFile(const GuiCommandScaling& scaling, const mpc_config::ReferenceFile& file, absl::string_view what) {
  const std::array<ScalingField, 3> fields = commandLimitFields();
  for (size_t i = 0; i < fields.size(); ++i) {
    EXPECT_EQ(std::optional<double>(scaling.commandLimits(static_cast<Eigen::Index>(i))), file.*(fields[i].member))
        << what << ": " << fields[i].name;
  }
  EXPECT_EQ(std::optional<double>(scaling.defaultPelvisHeight), file.default_base_height) << what;
}

TEST(GuiCommandScalingTest, EachValueComesFromItsOwnField) {
  const mpc_config::ReferenceFile file = completeReferenceFile();
  const absl::StatusOr<GuiCommandScaling> scaling = guiCommandScalingFromConfig(file);
  ASSERT_TRUE(scaling.ok()) << scaling.status();
  expectScalingOfFile(*scaling, file, /*what=*/"the complete file");
}

TEST(GuiCommandScalingTest, ACommandLimitThatIsNotPositiveIsRefusedByName) {
  for (const ScalingField& field : commandLimitFields()) {
    for (const double value : {0.0, -0.5}) {
      mpc_config::ReferenceFile file = completeReferenceFile();
      file.*(field.member) = value;
      const absl::StatusOr<GuiCommandScaling> scaling = guiCommandScalingFromConfig(file);
      ASSERT_FALSE(scaling.ok()) << field.name << " = " << value;
      EXPECT_EQ(scaling.status().code(), absl::StatusCode::kInvalidArgument) << field.name;
      EXPECT_TRUE(absl::StrContains(scaling.status().message(), field.name)) << scaling.status();
    }
  }
}

TEST(GuiCommandScalingTest, AFileThatLeavesOutAValueItReadsIsRefusedByName) {
  const std::array<ScalingField, 3> limits = commandLimitFields();
  const std::array<ScalingField, 4> fields = {
      {limits[0], limits[1], limits[2], {.name = "default_base_height", .member = &mpc_config::ReferenceFile::default_base_height}}};
  for (const ScalingField& field : fields) {
    mpc_config::ReferenceFile file = completeReferenceFile();
    (file.*(field.member)).reset();
    const absl::StatusOr<GuiCommandScaling> scaling = guiCommandScalingFromConfig(file);
    ASSERT_FALSE(scaling.ok()) << field.name;
    EXPECT_EQ(scaling.status().code(), absl::StatusCode::kInvalidArgument) << field.name;
    EXPECT_TRUE(absl::StrContains(scaling.status().message(), field.name)) << scaling.status();
  }
}

TEST(GuiCommandScalingTest, EveryConfigurationScalesTheCommandByItsReferenceFile) {
  for (const RobotConfiguration& configuration : robotConfigurations()) {
    // The path the driver reads: the configuration's.
    const absl::StatusOr<mpc_config::ReferenceFile> file = loadReferenceFile(configuration.referenceFile);
    ASSERT_TRUE(file.ok()) << configuration.name << ": " << file.status();
    const absl::StatusOr<GuiCommandScaling> scaling = guiCommandScalingFromConfig(*file);
    ASSERT_TRUE(scaling.ok()) << configuration.name << ": " << scaling.status();
    expectScalingOfFile(*scaling, *file, configuration.name);
    EXPECT_TRUE((scaling->commandLimits.array() > 0.0).all()) << configuration.name;
    EXPECT_GT(scaling->defaultPelvisHeight, 0.0) << configuration.name;
  }
}

TEST(WholeFileContactWrenchGateTest, EveryConfigurationsTaskFileSetsItsGate) {
  for (const RobotConfiguration& configuration : robotConfigurations()) {
    const absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(configuration.taskFile);
    ASSERT_TRUE(task.ok()) << configuration.name << ": " << task.status();
    const ControllerSideConfig settings = controllerSideSettingsFromConfig(*task);
    EXPECT_TRUE(settings.problems.empty()) << configuration.name;
    const std::optional<ContactWrenchGate::Config> gate = wholeFileContactWrenchGate(settings);
    if (!gate.has_value()) GTEST_FAIL() << configuration.name << ": no gate, the gate in use would be kept";
    const mpc_config::ContactWrenchGateConfig block = task->contact_wrench_gate.value_or(mpc_config::ContactWrenchGateConfig{});
    EXPECT_EQ(gate->debounceTime, block.debounce_time) << configuration.name;
    EXPECT_EQ(gate->rampTime, block.ramp_time) << configuration.name;
  }
}

}  // namespace
}  // namespace ocs2::humanoid::validation
