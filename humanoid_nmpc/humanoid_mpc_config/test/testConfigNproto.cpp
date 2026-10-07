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

// The nproto structs of the configuration schemas (ocs2::humanoid::mpc_config, tools/nproto/README.md): every message
// round-trips through its protobuf message, a file of only its header parses into the default struct of each file
// message, the members keep the presence the schemas ask for, and the retired fields of the task file are answered.

#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "humanoid_mpc_config/config_registries.nproto.pb.h"
#include "humanoid_mpc_config/contact_planning_file.nproto.pb.h"
#include "humanoid_mpc_config/gait_file.nproto.pb.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.pb.h"
#include "humanoid_mpc_config/joint_value.nproto.pb.h"
#include "humanoid_mpc_config/mpc_parameter_update.nproto.pb.h"
#include "humanoid_mpc_config/reference_file.nproto.pb.h"
#include "humanoid_mpc_config/task_file.nproto.pb.h"
#include "humanoid_mpc_config/tuning_group_options.nproto.pb.h"
#include "humanoid_mpc_config/tuning_options.nproto.pb.h"
#include "humanoid_mpc_config/xyz.nproto.pb.h"
#include "humanoid_mpc_config/yaw_pitch_roll.nproto.pb.h"
#include "nproto/Textproto.h"
#include "tools/nproto/test/ProtoTestValues.h"

namespace ocs2::humanoid::mpc_config {
namespace {

using nproto::test_support::ExpectRoundTrips;

template <typename Member, typename Expected>
constexpr bool kIs = std::is_same_v<Member, Expected>;

// Absent means the default: plain members. Absent has its own meaning: std::optional.
static_assert(kIs<decltype(TaskFile::telemetry_sinks), std::vector<std::string>>);
static_assert(kIs<decltype(TaskFile::telemetry_frequency), std::optional<double>>);
static_assert(kIs<decltype(TaskFile::gantry_hold), std::string>);
static_assert(kIs<decltype(TaskFile::enable_online_tuning), bool>);
static_assert(kIs<decltype(TaskFile::centroidal_model), std::optional<std::string>>);
static_assert(kIs<decltype(DcmTerminalCostConfig::com_height), std::optional<double>>);
static_assert(kIs<decltype(ContactPlanningFile::SharedConfig::com_height), std::optional<double>>);
static_assert(kIs<decltype(TaskFile::terminal_cost_scaling), std::optional<double>>);
static_assert(kIs<decltype(TaskFile::contact_wrench_gate), std::optional<ContactWrenchGateConfig>>);
static_assert(kIs<decltype(TaskFile::state_weights), StateWeights>);
static_assert(kIs<decltype(TaskFile::task_space_costs), std::vector<TaskSpaceCostConfig>>);
static_assert(kIs<decltype(MpcParameterUpdate::contact_planning), std::optional<ContactPlanningFile>>);
static_assert(kIs<decltype(ReferenceFile::default_joint_state), std::vector<JointValue>>);
static_assert(kIs<decltype(ReferenceFile::max_displacement_velocity_x), std::optional<double>>);
static_assert(kIs<decltype(GaitFile::gaits), std::vector<ModeSequenceTemplateConfig>>);
static_assert(kIs<decltype(JointPdGainsFile::joint_gains), std::vector<JointPdGainsFile::JointGains>>);
static_assert(kIs<decltype(Xyz::x), double> && kIs<decltype(YawPitchRoll::yaw), double> && kIs<decltype(JointValue::joint), std::string>);
static_assert(kIs<decltype(TuningOptions::slider_min), std::optional<double>>);
static_assert(kIs<decltype(TuningOptions::reload), TuningOptions::Reload>);
static_assert(kIs<decltype(TuningOptions::formulations), std::vector<std::string>>);

template <typename StructType, typename ProtoType>
struct Conversion {
  using Struct = StructType;
  using Proto = ProtoType;
};

// The file messages, which hold every block message (FillWithTestValues fills submessages recursively), the registries'
// names, the primitives, and the option messages, which no file holds.
using AllMessages = ::testing::Types<Conversion<TaskFile, humanoid_mpc_config::TaskFile>,
                                     Conversion<ReferenceFile, humanoid_mpc_config::ReferenceFile>,
                                     Conversion<JointPdGainsFile, humanoid_mpc_config::JointPdGainsFile>,
                                     Conversion<ContactPlanningFile, humanoid_mpc_config::ContactPlanningFile>,
                                     Conversion<GaitFile, humanoid_mpc_config::GaitFile>,
                                     Conversion<MpcParameterUpdate, humanoid_mpc_config::MpcParameterUpdate>,
                                     Conversion<ConfigRegistries, humanoid_mpc_config::ConfigRegistries>,
                                     Conversion<Xyz, humanoid_mpc_config::Xyz>,
                                     Conversion<YawPitchRoll, humanoid_mpc_config::YawPitchRoll>,
                                     Conversion<JointValue, humanoid_mpc_config::JointValue>,
                                     Conversion<TuningOptions, humanoid_mpc_config::TuningOptions>,
                                     Conversion<TuningOptions::Condition, humanoid_mpc_config::TuningOptions_Condition>,
                                     Conversion<TuningGroupOptions, humanoid_mpc_config::TuningGroupOptions>>;

template <typename T>
class ConfigRoundTripTest : public ::testing::Test {};
TYPED_TEST_SUITE(ConfigRoundTripTest, AllMessages);

TYPED_TEST(ConfigRoundTripTest, ConvertsBothWaysWithoutLosingAnything) {
  ExpectRoundTrips<typename TypeParam::Struct, typename TypeParam::Proto>();
}

// The file messages, which a configuration file of nothing but its header must parse into.
using FileMessages = ::testing::Types<Conversion<TaskFile, humanoid_mpc_config::TaskFile>,
                                      Conversion<ReferenceFile, humanoid_mpc_config::ReferenceFile>,
                                      Conversion<JointPdGainsFile, humanoid_mpc_config::JointPdGainsFile>,
                                      Conversion<ContactPlanningFile, humanoid_mpc_config::ContactPlanningFile>,
                                      Conversion<GaitFile, humanoid_mpc_config::GaitFile>>;

template <typename T>
class HeaderOnlyFileTest : public ::testing::Test {};
TYPED_TEST_SUITE(HeaderOnlyFileTest, FileMessages);

TYPED_TEST(HeaderOnlyFileTest, ParsesIntoTheDefaultStruct) {
  using Proto = typename TypeParam::Proto;
  const std::string text = absl::StrCat(
      "# proto-file: humanoid_nmpc/humanoid_mpc_config/<file>.proto\n# proto-message: ", Proto::descriptor()->full_name(), "\n");
  const absl::StatusOr<Proto> proto = nproto::ParseTextproto<Proto>(text, "header_only.textproto");
  ASSERT_TRUE(proto.ok()) << proto.status();
  typename TypeParam::Struct value;
  const absl::Status converted = FromProto(*proto, &value);
  ASSERT_TRUE(converted.ok()) << converted;
  EXPECT_TRUE(value == typename TypeParam::Struct{});
}

TEST(PresenceTest, TheContactWrenchGateAndTheContactPlanningFileKeepTheirPresence) {
  humanoid_mpc_config::MpcParameterUpdate proto;
  MpcParameterUpdate value;
  ASSERT_TRUE(FromProto(proto, &value).ok());
  EXPECT_FALSE(value.contact_planning.has_value()) << "a robot without a contact planner";
  EXPECT_FALSE(value.task.contact_wrench_gate.has_value()) << "the instantaneous gate";

  proto.mutable_contact_planning();
  proto.mutable_task()->mutable_contact_wrench_gate();
  ASSERT_TRUE(FromProto(proto, &value).ok());
  EXPECT_TRUE(value.contact_planning.has_value());
  EXPECT_TRUE(value.task.contact_wrench_gate.has_value());
}

TEST(RetiredFieldTest, TheRetiredKeysOfTheTaskFileNameTheirReplacement) {
  const absl::StatusOr<humanoid_mpc_config::TaskFile> camel =
      nproto::ParseTextproto<humanoid_mpc_config::TaskFile>("useDcmTerminalCost: true\n", "task.textproto");
  EXPECT_EQ(camel.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StartsWith(camel.status().message(), "task.textproto:1:1: 'useDcmTerminalCost' is retired: ")) << camel.status();
  EXPECT_TRUE(absl::StrContains(camel.status().message(), "list dcm_terminal_cost under costs")) << camel.status();

  // A retired key of a block, and a block that is retired as a whole.
  const absl::StatusOr<humanoid_mpc_config::TaskFile> nested =
      nproto::ParseTextproto<humanoid_mpc_config::TaskFile>("contact_implicit {\n  terrainHeight: 0\n}\n", "task.textproto");
  EXPECT_TRUE(absl::StrContains(nested.status().message(), "task.textproto:2:3: 'terrainHeight' is retired: ")) << nested.status();
  const absl::StatusOr<humanoid_mpc_config::TaskFile> ddp =
      nproto::ParseTextproto<humanoid_mpc_config::TaskFile>("ddp {\n}\n", "task.textproto");
  EXPECT_TRUE(absl::StrContains(ddp.status().message(), "'ddp' is retired: the MPC runs the SQP solver")) << ddp.status();
  const absl::StatusOr<humanoid_mpc_config::JointPdGainsFile> gains =
      nproto::ParseTextproto<humanoid_mpc_config::JointPdGainsFile>("default {\n}\n", "joint_pd_gains.textproto");
  EXPECT_TRUE(absl::StrContains(gains.status().message(), "'default' is retired: ")) << gains.status();
}

/** A text, its file, where its retired key is and what the replacement names. */
struct RetiredKeyCase {
  const char* absl_nonnull text;
  const char* absl_nonnull position;
  const char* absl_nonnull replacement;
};

TEST(RetiredFieldTest, TheFormulationSwitchesThatBecameNamesAreAnsweredWithTheirNames) {
  // The int codes and the booleans that selected a formulation, each refused where it was with the name that replaced
  // it, in its snake_case and in its old camelCase spelling.
  for (const RetiredKeyCase& retired : std::vector<RetiredKeyCase>{
           {.text = "centroidal_model_type: 0\n",
            .position = "task.textproto:1:1: 'centroidal_model_type' is retired: ",
            .replacement = "centroidal_model: \"full_centroidal_dynamics\" where it was 0"},
           {.text = "centroidalModelType: 1\n",
            .position = "task.textproto:1:1: 'centroidalModelType' is retired: ",
            .replacement = "\"single_rigid_body_dynamics\" where it was 1"},
           {.text = "rollout {\n  root_finding_algorithm: 0\n}\n",
            .position = "task.textproto:2:3: 'root_finding_algorithm' is retired: ",
            .replacement = "root_finder_type: \"ANDERSON_BJORCK\" where it was 0"},
           {.text = "model_settings {\n  foot_constraint {\n    constrain_orientation: true\n  }\n}\n",
            .position = "task.textproto:3:5: 'constrain_orientation' is retired: ",
            .replacement = R"(stance_constraint: "position" where it was false, "position_and_tilt" where it was true)"},
           {.text = "model_settings {\n  foot_constraint {\n    constrainYawRateAboutContactNormal: true\n  }\n}\n",
            .position = "task.textproto:3:5: 'constrainYawRateAboutContactNormal' is retired: ",
            .replacement = "stance_constraint: \"position_and_orientation\" where it was true"},
           {.text = "task_space_foot_cost {\n  active_in_stance: true\n}\n",
            .position = "task.textproto:2:3: 'active_in_stance' is retired: ",
            .replacement = "active_phases: \"swing_and_stance\" where it was true"},
       }) {
    const absl::StatusOr<humanoid_mpc_config::TaskFile> task =
        nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(retired.text, "task.textproto");
    EXPECT_EQ(task.status().code(), absl::StatusCode::kInvalidArgument) << retired.text;
    EXPECT_TRUE(absl::StartsWith(task.status().message(), retired.position)) << task.status();
    EXPECT_TRUE(absl::StrContains(task.status().message(), retired.replacement)) << task.status();
  }
  for (const RetiredKeyCase& retired : std::vector<RetiredKeyCase>{
           {.text = "planner {\n  run_in_background_thread: false\n}\n",
            .position = "contact_planning.textproto:2:3: 'run_in_background_thread' is retired: ",
            .replacement = "\"pre_solve_hook\" where it was false"},
           {.text = "terminal_dcm {\n  trackCommandedVelocity: true\n}\n",
            .position = "contact_planning.textproto:2:3: 'trackCommandedVelocity' is retired: ",
            .replacement = "target: \"commanded_velocity\" where it was true"},
       }) {
    const absl::StatusOr<humanoid_mpc_config::ContactPlanningFile> planning =
        nproto::ParseTextproto<humanoid_mpc_config::ContactPlanningFile>(retired.text, "contact_planning.textproto");
    EXPECT_EQ(planning.status().code(), absl::StatusCode::kInvalidArgument) << retired.text;
    EXPECT_TRUE(absl::StartsWith(planning.status().message(), retired.position)) << planning.status();
    EXPECT_TRUE(absl::StrContains(planning.status().message(), retired.replacement)) << planning.status();
  }
}

TEST(RetiredFieldTest, TheSqpSettingsNothingReadsAreRefusedWithWhatToDo) {
  // multiple_shooting.inequality_constraint_mu and _delta: nothing in the SQP solver read them, so a file that sets them
  // is told to delete the line instead of tuning a value that changes nothing.
  for (const RetiredKeyCase& retired : std::vector<RetiredKeyCase>{
           {.text = "multiple_shooting {\n  inequality_constraint_mu: 0.1\n}\n",
            .position = "task.textproto:2:3: 'inequality_constraint_mu' is retired: ",
            .replacement = "nothing in the SQP solver reads it; delete the line"},
           {.text = "multiple_shooting {\n  inequalityConstraintDelta: 5.0\n}\n",
            .position = "task.textproto:2:3: 'inequalityConstraintDelta' is retired: ",
            .replacement = "nothing in the SQP solver reads it; delete the line"},
       }) {
    const absl::StatusOr<humanoid_mpc_config::TaskFile> task =
        nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(retired.text, "task.textproto");
    EXPECT_EQ(task.status().code(), absl::StatusCode::kInvalidArgument) << retired.text;
    EXPECT_TRUE(absl::StartsWith(task.status().message(), retired.position)) << task.status();
    EXPECT_TRUE(absl::StrContains(task.status().message(), retired.replacement)) << task.status();
  }
}

TEST(RetiredFieldTest, AnOldCamelCaseKeyOfALiveFieldIsSuggested) {
  const absl::StatusOr<humanoid_mpc_config::TaskFile> task =
      nproto::ParseTextproto<humanoid_mpc_config::TaskFile>("gantryHold: \"weld_constraint\"\n", "task.textproto");
  EXPECT_TRUE(absl::StrContains(task.status().message(), "Did you mean \"gantry_hold\"?")) << task.status();
}

}  // namespace
}  // namespace ocs2::humanoid::mpc_config
