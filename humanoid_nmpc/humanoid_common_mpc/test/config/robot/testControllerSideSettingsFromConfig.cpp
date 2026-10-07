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

// The controller-side settings of a typed task file: the estimator always, the gate when the file has a valid block, an
// invalid gate reported and left out while the estimator still applies, and the schema's gate defaults equal to the
// gate's own.

#include <limits>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/robot/ControllerSideSettingsFromConfig.h"
#include "humanoid_common_mpc/contact/ContactWrenchGate.h"
#include "humanoid_mpc_config/contact_wrench_gate_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::HasSubstr;
using ::testing::Optional;

// The times of a gate the tests set and read back [s].
constexpr double kDebounceTime = 0.01;
constexpr double kRampTime = 0.04;

TEST(ControllerSideSettingsFromConfigTest, AFileThatSaysNothingNamesTheDefaultEstimatorAndNoGate) {
  const ControllerSideConfig settings = controllerSideSettingsFromConfig(mpc_config::TaskFile{});
  EXPECT_EQ(settings.contactEstimator, mpc_config::TaskFile{}.contact_estimator);
  EXPECT_FALSE(settings.contactWrenchGate.has_value()) << "no block: the instantaneous gate";
  EXPECT_TRUE(settings.problems.empty());
}

TEST(ControllerSideSettingsFromConfigTest, TheSchemaDefaultsOfTheGateAreTheGatesOwn) {
  const absl::StatusOr<ContactWrenchGate::Config> gate = contactWrenchGateFromConfig(mpc_config::ContactWrenchGateConfig{});
  ASSERT_TRUE(gate.ok()) << gate.status();
  EXPECT_EQ(gate->debounceTime, ContactWrenchGate::Config{}.debounceTime);
  EXPECT_EQ(gate->rampTime, ContactWrenchGate::Config{}.rampTime);
}

TEST(ControllerSideSettingsFromConfigTest, TheFilesEstimatorAndGateAreRead) {
  mpc_config::TaskFile task;
  task.contact_estimator = "robot_state";
  mpc_config::ContactWrenchGateConfig gate;
  gate.debounce_time = kDebounceTime;
  gate.ramp_time = kRampTime;
  task.contact_wrench_gate = gate;
  const ControllerSideConfig settings = controllerSideSettingsFromConfig(task);
  EXPECT_EQ(settings.contactEstimator, "robot_state");
  EXPECT_THAT(settings.contactWrenchGate, Optional(AllOf(Field(&ContactWrenchGate::Config::debounceTime, kDebounceTime),
                                                         Field(&ContactWrenchGate::Config::rampTime, kRampTime))));
  EXPECT_TRUE(settings.problems.empty());
}

TEST(ControllerSideSettingsFromConfigTest, AnInvalidGateIsReportedAndLeftOutWhileTheEstimatorApplies) {
  for (const double bad : {-0.01, std::numeric_limits<double>::quiet_NaN()}) {
    mpc_config::TaskFile task;
    task.contact_estimator = "always_in_contact";
    mpc_config::ContactWrenchGateConfig gate;
    gate.ramp_time = bad;
    task.contact_wrench_gate = gate;
    const ControllerSideConfig settings = controllerSideSettingsFromConfig(task);
    EXPECT_EQ(settings.contactEstimator, "always_in_contact");
    EXPECT_FALSE(settings.contactWrenchGate.has_value());
    EXPECT_THAT(settings.problems, ElementsAre(HasSubstr("contact_wrench_gate.ramp_time is")));

    const absl::StatusOr<ContactWrenchGate::Config> refused = contactWrenchGateFromConfig(gate);
    EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  }
}

TEST(ControllerSideSettingsFromConfigTest, ARefusedGateNamesTheFieldTheOperatorEdits) {
  mpc_config::ContactWrenchGateConfig gate;
  gate.debounce_time = -1.0;
  EXPECT_EQ(contactWrenchGateFromConfig(gate).status().message(),
            "contact_wrench_gate.debounce_time is -1 [s]; it must be a non-negative number");
  gate.debounce_time = 0.0;
  gate.ramp_time = -0.5;
  EXPECT_EQ(contactWrenchGateFromConfig(gate).status().message(),
            "contact_wrench_gate.ramp_time is -0.5 [s]; it must be a non-negative number");
}

/** The gate the controller-side settings of `task` set when the robot process applies it whole. */
std::optional<ContactWrenchGate::Config> gateOfTaskFile(const mpc_config::TaskFile& task) {
  return wholeFileContactWrenchGate(controllerSideSettingsFromConfig(task));
}

TEST(WholeFileContactWrenchGateTest, AFileWithoutAGateBlockSetsTheInstantaneousGate) {
  const mpc_config::TaskFile task;
  ASSERT_FALSE(task.contact_wrench_gate.has_value());
  const std::optional<ContactWrenchGate::Config> gate = gateOfTaskFile(task);
  if (!gate.has_value()) GTEST_FAIL() << "no gate: the gate in use would be kept";
  const ContactWrenchGate::Config instantaneous;
  EXPECT_EQ(gate->debounceTime, instantaneous.debounceTime);
  EXPECT_EQ(gate->rampTime, instantaneous.rampTime);
}

TEST(WholeFileContactWrenchGateTest, AGateBlockSetsItsOwnTimes) {
  const mpc_config::ContactWrenchGateConfig block = {.debounce_time = kDebounceTime, .ramp_time = kRampTime};
  mpc_config::TaskFile task;
  task.contact_wrench_gate = block;
  const std::optional<ContactWrenchGate::Config> gate = gateOfTaskFile(task);
  if (!gate.has_value()) GTEST_FAIL() << "no gate: the gate in use would be kept";
  EXPECT_EQ(gate->debounceTime, block.debounce_time);
  EXPECT_EQ(gate->rampTime, block.ramp_time);
}

TEST(WholeFileContactWrenchGateTest, ARefusedGateBlockKeepsTheGateInUse) {
  for (const mpc_config::ContactWrenchGateConfig& block : {mpc_config::ContactWrenchGateConfig{.debounce_time = -0.01, .ramp_time = 0.0},
                                                           mpc_config::ContactWrenchGateConfig{.debounce_time = 0.0, .ramp_time = -0.01}}) {
    mpc_config::TaskFile task;
    task.contact_wrench_gate = block;
    const ControllerSideConfig settings = controllerSideSettingsFromConfig(task);
    EXPECT_FALSE(settings.problems.empty()) << "a negative time is refused";
    EXPECT_FALSE(wholeFileContactWrenchGate(settings).has_value());
  }
}

}  // namespace
}  // namespace ocs2::humanoid
