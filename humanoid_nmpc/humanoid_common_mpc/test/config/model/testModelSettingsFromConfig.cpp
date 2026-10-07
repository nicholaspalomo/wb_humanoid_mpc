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

#include <bit>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "gmock/gmock.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/model/ModelSettingsFromConfig.h"
#include "humanoid_mpc_config/model_settings_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "nproto/Textproto.h"

/**
 * ModelSettings::Create() of the typed task file and the conversions of its blocks, on a two-joint robot written by the
 * test: what it builds, what it refuses, that every field of the schema reaches the settings, and that the schema's
 * defaults are the settings' defaults.
 */
namespace ocs2::humanoid {
namespace {

using ::google::protobuf::Descriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::Message;
using ::google::protobuf::Reflection;
using ::testing::HasSubstr;

constexpr char kInertial[] =
    R"(<inertial><mass value="1.0"/><origin xyz="0 0 0"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>)";

/** A floating base with two revolute joints, `shoulder` and `hip`; $0 is each link's inertial block. */
constexpr char kTwoJointUrdf[] = R"(<robot name="two_joints">
  <link name="base">$0</link>
  <link name="arm">$0</link>
  <link name="leg">$0</link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="arm"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <joint name="hip" type="revolute">
    <parent link="base"/><child link="leg"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>
)";

/** The robot of kTwoJointUrdf, written to the test's scratch directory. */
std::string twoJointUrdf() {
  const std::string path = absl::StrCat(::testing::TempDir(), "/two_joints.urdf");
  std::ofstream(path) << absl::Substitute(kTwoJointUrdf, kInertial);
  return path;
}

/** A task file whose model_settings name the two-joint robot. */
mpc_config::TaskFile twoJointTask() {
  mpc_config::TaskFile task;
  task.model_settings.robot_name = "two_joints";
  task.model_settings.phase_transition_stance_time = 0.25;
  return task;
}

/** The typed task file of the textproto `text`. */
absl::StatusOr<mpc_config::TaskFile> parseTask(absl::string_view text) {
  const absl::StatusOr<humanoid_mpc_config::TaskFile> message =
      nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(text, "task.textproto");
  if (!message.ok()) return message.status();
  mpc_config::TaskFile task;
  RETURN_IF_ERROR(mpc_config::FromProto(*message, &task));
  return task;
}

/** `value` exactly: its bits, so that two values that print alike in decimal still differ. */
std::string text(double value) {
  return absl::StrCat(std::bit_cast<uint64_t>(value));
}

std::string describe(const ModelSettings::FootConstraintConfig& config) {
  return absl::StrJoin(
      {text(config.positionErrorGain_z), text(config.orientationErrorGain), text(config.linearVelocityErrorGain_z),
       text(config.linearVelocityErrorGain_xy), text(config.angularVelocityErrorGain), text(config.linearAccelerationErrorGain_z),
       text(config.linearAccelerationErrorGain_xy), text(config.angularAccelerationErrorGain), text(config.softConstraintWeight),
       text(config.normalVelocitySoftConstraintWeight), std::string(config.constrainOrientation ? "true" : "false"),
       std::string(config.constrainYawRateAboutContactNormal ? "true" : "false")},
      " ");
}

std::string describe(const ModelSettings::ContactImplicitConfig& config) {
  return absl::StrJoin(
      {text(config.complementarityWeight), text(config.slipWeight), text(config.heightReference), text(config.velocityReference),
       text(config.angularVelocityReference), text(config.penetrationWeight), text(config.gapSmoothing)},
      " ");
}

/** Every value of `settings`, or the error that `settings` is, as one string per field. */
std::string describe(const absl::StatusOr<ModelSettings>& settings) {
  if (!settings.ok()) {
    return settings.status().ToString();
  }
  std::vector<std::string> indices;
  for (const size_t index : settings->mpcModelToFullJointsIndices) {
    indices.push_back(absl::StrCat(index));
  }
  return absl::StrJoin({settings->robotName,
                        std::string(settings->verboseCppAd ? "true" : "false"),
                        std::string(settings->recompileLibrariesCppAd ? "true" : "false"),
                        settings->modelFolderCppAd,
                        text(settings->phaseTransitionStanceTime),
                        absl::StrJoin(settings->fullJointNames, ","),
                        absl::StrJoin(settings->fixedJointNames, ","),
                        absl::StrJoin(settings->contactNames6DoF, ","),
                        absl::StrJoin(settings->contactNames3DoF, ","),
                        absl::StrJoin(settings->contactParentJointNames, ","),
                        absl::StrJoin(settings->mpcModelJointNames, ","),
                        absl::StrJoin(indices, ","),
                        absl::StrJoin(settings->contactNames, ","),
                        absl::StrCat(settings->mpc_joint_dim, "/", settings->full_joint_dim),
                        std::string(settings->hasArmSwingJoints ? "true" : "false"),
                        absl::StrCat(settings->j_l_shoulder_y_index, ",", settings->j_r_shoulder_y_index, ",", settings->j_l_elbow_y_index,
                                     ",", settings->j_r_elbow_y_index),
                        describe(settings->footConstraintConfig),
                        describe(settings->contactImplicitConfig),
                        text(settings->nominalFootholdConfig.stepWidth),
                        text(settings->terrainHeight)},
                       "\n");
}

TEST(ModelSettingsFromConfig, LoadsTheActiveJointsOfTheUrdf) {
  const absl::StatusOr<ModelSettings> settings = ModelSettings::Create(twoJointTask(), twoJointUrdf(), "test_", /*verbose=*/false);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_EQ(settings->robotName, "two_joints");
  EXPECT_EQ(settings->mpc_joint_dim, 2U);
  EXPECT_EQ(settings->full_joint_dim, 2U);
  EXPECT_THAT(settings->mpcModelJointNames, ::testing::UnorderedElementsAre("shoulder", "hip"));
  EXPECT_EQ(settings->jointIndexMap.size(), 2U);
  EXPECT_FALSE(settings->hasArmSwingJoints) << "a file without arm_joint_names disables the arm swing";
  EXPECT_EQ(settings->modelFolderCppAd, "cppad_code_gen/cppad_test_two_joints");
  EXPECT_EQ(settings->phaseTransitionStanceTime, 0.25);
}

TEST(ModelSettingsFromConfig, AFixedJointLeavesTheMpcModel) {
  mpc_config::TaskFile task = twoJointTask();
  task.model_settings.fixed_joint_names = {"shoulder"};
  task.model_settings.contact_names_6dof = {"foot"};
  const absl::StatusOr<ModelSettings> settings = ModelSettings::Create(task, twoJointUrdf(), "test_", /*verbose=*/true);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_THAT(settings->mpcModelJointNames, ::testing::ElementsAre("hip"));
  EXPECT_EQ(settings->mpcModelToFullJointsIndices.size(), 1U);
  EXPECT_EQ(settings->fullJointNames.at(settings->mpcModelToFullJointsIndices[0]), "hip");
  EXPECT_THAT(settings->contactNames, ::testing::ElementsAre("foot"));
}

TEST(ModelSettingsFromConfig, FixingEveryJointIsRefused) {
  mpc_config::TaskFile task = twoJointTask();
  task.model_settings.fixed_joint_names = {"shoulder", "hip"};
  const absl::StatusOr<ModelSettings> settings = ModelSettings::Create(task, twoJointUrdf(), "test_", /*verbose=*/false);
  EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(settings.status().message(), HasSubstr("model_settings.fixed_joint_names"));
  EXPECT_THAT(settings.status().message(), HasSubstr("at least one active joint"));
}

TEST(ModelSettingsFromConfig, AnArmJointThatIsNotAnActiveJointIsRefusedNamingIt) {
  mpc_config::TaskFile misspelled = twoJointTask();
  misspelled.model_settings.arm_joint_names = {
      .left_shoulder_y = "shoulder", .right_shoulder_y = "shoulder", .left_elbow_y = "shoulder", .right_elbow_y = "elbow"};
  const absl::StatusOr<ModelSettings> refused = ModelSettings::Create(misspelled, twoJointUrdf(), "test_", /*verbose=*/false);
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(refused.status().message(), HasSubstr("model_settings.arm_joint_names.right_elbow_y is 'elbow'"));

  mpc_config::TaskFile fixedArm = twoJointTask();
  fixedArm.model_settings.fixed_joint_names = {"shoulder"};
  fixedArm.model_settings.arm_joint_names = {
      .left_shoulder_y = "shoulder", .right_shoulder_y = "hip", .left_elbow_y = "hip", .right_elbow_y = "hip"};
  const absl::StatusOr<ModelSettings> fixed = ModelSettings::Create(fixedArm, twoJointUrdf(), "test_", /*verbose=*/false);
  EXPECT_EQ(fixed.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(fixed.status().message(), HasSubstr("left_shoulder_y"));

  mpc_config::TaskFile partial = twoJointTask();
  partial.model_settings.arm_joint_names.left_shoulder_y = "shoulder";
  EXPECT_EQ(ModelSettings::Create(partial, twoJointUrdf(), "test_", /*verbose=*/false).status().code(), absl::StatusCode::kInvalidArgument)
      << "naming one arm joint of the four is refused";

  mpc_config::TaskFile valid = twoJointTask();
  valid.model_settings.arm_joint_names = {
      .left_shoulder_y = "shoulder", .right_shoulder_y = "hip", .left_elbow_y = "shoulder", .right_elbow_y = "hip"};
  const absl::StatusOr<ModelSettings> resolved = ModelSettings::Create(valid, twoJointUrdf(), "test_", /*verbose=*/false);
  ASSERT_TRUE(resolved.ok()) << resolved.status();
  EXPECT_TRUE(resolved->hasArmSwingJoints);
  EXPECT_EQ(resolved->j_l_shoulder_y_index, resolved->jointIndexMap.at("shoulder"));
  EXPECT_EQ(resolved->j_r_shoulder_y_index, resolved->jointIndexMap.at("hip"));
  EXPECT_EQ(resolved->j_l_elbow_y_index, resolved->jointIndexMap.at("shoulder"));
  EXPECT_EQ(resolved->j_r_elbow_y_index, resolved->jointIndexMap.at("hip"));
}

TEST(ModelSettingsFromConfig, AUrdfThatCannotBeReadIsRefusedNamingIt) {
  const std::string missing = absl::StrCat(::testing::TempDir(), "/no_such_robot.urdf");
  const absl::StatusOr<ModelSettings> settings = ModelSettings::Create(twoJointTask(), missing, "test_", /*verbose=*/false);
  EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(settings.status().message(), HasSubstr("no_such_robot.urdf"));
}

TEST(ModelSettingsFromConfig, TheSchemaDefaultsAreTheSettingsDefaults) {
  const absl::StatusOr<ModelSettings::FootConstraintConfig> footConstraint =
      footConstraintFromConfig(mpc_config::ModelSettingsConfig::FootConstraintConfig{});
  ASSERT_TRUE(footConstraint.ok()) << footConstraint.status();
  EXPECT_EQ(describe(*footConstraint), describe(ModelSettings::FootConstraintConfig{}));
  EXPECT_EQ(describe(contactImplicitFromConfig(mpc_config::ContactImplicitConfig{})), describe(ModelSettings::ContactImplicitConfig{}));
  EXPECT_EQ(text(nominalFootholdFromConfig(mpc_config::NominalFootholdConfig{}).stepWidth),
            text(ModelSettings::NominalFootholdConfig{}.stepWidth));
}

/** A value of `field` other than its default: a number scaled and shifted, a bool flipped, a joint name of the robot. */
void setNonDefault(const FieldDescriptor* absl_nonnull field, Message* absl_nonnull message) {
  const Reflection* absl_nonnull reflection = message->GetReflection();
  switch (field->cpp_type()) {
    case FieldDescriptor::CPPTYPE_DOUBLE:
      reflection->SetDouble(message, field, field->default_value_double() * 1.5 + 0.25);
      break;
    case FieldDescriptor::CPPTYPE_BOOL:
      reflection->SetBool(message, field, !field->default_value_bool());
      break;
    case FieldDescriptor::CPPTYPE_STRING:
      field->is_repeated() ? reflection->AddString(message, field, "shoulder") : reflection->SetString(message, field, "shoulder");
      break;
    default:
      ADD_FAILURE() << field->full_name() << " is of a type this test does not mutate";
      break;
  }
}

/** The dotted paths, `prefix` first, of every scalar or repeated scalar field below `descriptor`. */
void collectLeaves(const Descriptor* absl_nonnull descriptor, const std::string& prefix, std::vector<std::string>& leaves) {
  for (int index = 0; index < descriptor->field_count(); ++index) {
    const FieldDescriptor* absl_nonnull field = descriptor->field(index);
    const std::string path = absl::StrCat(prefix, ".", field->name());
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      ASSERT_FALSE(field->is_repeated()) << path << ": a repeated block, which this test does not mutate";
      collectLeaves(field->message_type(), path, leaves);
    } else {
      leaves.push_back(path);
    }
  }
}

TEST(ModelSettingsFromConfig, EveryFieldOfTheSchemaReachesTheSettings) {
  // The blocks ModelSettings is built from: a field that no conversion reads leaves the settings as they were.
  const Descriptor* absl_nonnull task = humanoid_mpc_config::TaskFile::descriptor();
  std::vector<std::string> leaves = {"terrain_height"};
  for (const char* absl_nonnull block : {"model_settings", "contact_implicit", "nominal_foothold"}) {
    const FieldDescriptor* absl_nullable field = task->FindFieldByName(block);
    ASSERT_NE(field, nullptr) << block;
    collectLeaves(field->message_type(), block, leaves);
  }

  const std::string urdf = twoJointUrdf();
  const std::string baseline = describe(ModelSettings::Create(mpc_config::TaskFile{}, urdf, "test_", /*verbose=*/false));
  for (const std::string& leaf : leaves) {
    humanoid_mpc_config::TaskFile message;
    Message* absl_nonnull current = &message;
    const std::vector<std::string> names = absl::StrSplit(leaf, '.');
    for (size_t depth = 0; depth < names.size(); ++depth) {
      const FieldDescriptor* absl_nullable field = current->GetDescriptor()->FindFieldByName(names[depth]);
      ASSERT_NE(field, nullptr) << leaf;
      if (depth + 1 < names.size()) {
        current = current->GetReflection()->MutableMessage(current, field);
      } else {
        setNonDefault(field, current);
      }
    }
    mpc_config::TaskFile changed;
    ASSERT_TRUE(mpc_config::FromProto(message, &changed).ok());
    EXPECT_NE(describe(ModelSettings::Create(changed, urdf, "test_", /*verbose=*/false)), baseline)
        << leaf << " does not reach ModelSettings";
  }
  EXPECT_GT(leaves.size(), 4U) << "the walk found the fields of the blocks";
}

TEST(ModelSettingsFromConfig, EachStanceConstraintNameSetsTheRowsTheBooleansItReplacedSet) {
  // The parity of the two retired booleans: (constrain_orientation, constrain_yaw_rate_about_contact_normal) = (false,
  // false) is "position", (true, false) "position_and_tilt" and (true, true) "position_and_orientation"; a yaw-rate row
  // without the orientation rows, (false, true), did nothing and has no name.
  struct Rows {
    const char* absl_nonnull name;
    bool constrainOrientation;
    bool constrainYawRateAboutContactNormal;
  };
  const std::vector<Rows> rows = {
      {.name = "position", .constrainOrientation = false, .constrainYawRateAboutContactNormal = false},
      {.name = "position_and_tilt", .constrainOrientation = true, .constrainYawRateAboutContactNormal = false},
      {.name = "position_and_orientation", .constrainOrientation = true, .constrainYawRateAboutContactNormal = true}};
  std::vector<std::string> names;
  for (const Rows& expected : rows) {
    mpc_config::ModelSettingsConfig::FootConstraintConfig config;
    config.stance_constraint = expected.name;
    const absl::StatusOr<ModelSettings::FootConstraintConfig> converted = footConstraintFromConfig(config);
    ASSERT_TRUE(converted.ok()) << converted.status();
    EXPECT_EQ(converted->constrainOrientation, expected.constrainOrientation) << expected.name;
    EXPECT_EQ(converted->constrainYawRateAboutContactNormal, expected.constrainYawRateAboutContactNormal) << expected.name;
    EXPECT_EQ(stanceConstraintName(*converted), expected.name);
    names.emplace_back(expected.name);
  }
  EXPECT_EQ(stanceConstraintNames(), names);
  ModelSettings::FootConstraintConfig yawRateWithoutOrientation;
  yawRateWithoutOrientation.constrainOrientation = false;
  yawRateWithoutOrientation.constrainYawRateAboutContactNormal = true;
  EXPECT_EQ(stanceConstraintName(yawRateWithoutOrientation), "position") << "three rows have no yaw-rate row";
  EXPECT_EQ(mpc_config::ModelSettingsConfig::FootConstraintConfig{}.stance_constraint, "position_and_tilt")
      << "the default is the booleans' defaults, (true, false)";

  mpc_config::ModelSettingsConfig::FootConstraintConfig unknown;
  unknown.stance_constraint = "full_pose";
  const absl::StatusOr<ModelSettings::FootConstraintConfig> refused = footConstraintFromConfig(unknown);
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(refused.status().message(), HasSubstr("model_settings.foot_constraint.stance_constraint is 'full_pose'"));
  EXPECT_THAT(refused.status().message(), HasSubstr("position, position_and_tilt, position_and_orientation"));
}

TEST(ModelSettingsFromConfig, ASoftFootWeightThatIsNotAFinitePositiveNumberIsRefusedNamingIt) {
  using FootConstraintConfig = mpc_config::ModelSettingsConfig::FootConstraintConfig;
  struct Weight {
    const char* absl_nonnull name;
    double FootConstraintConfig::*absl_nonnull field;
  };
  const std::vector<Weight> weights = {
      {.name = "soft_constraint_weight", .field = &FootConstraintConfig::soft_constraint_weight},
      {.name = "normal_velocity_soft_constraint_weight", .field = &FootConstraintConfig::normal_velocity_soft_constraint_weight}};
  for (const Weight& weight : weights) {
    for (const double value : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
      FootConstraintConfig config;
      config.*weight.field = value;
      const absl::StatusOr<ModelSettings::FootConstraintConfig> refused = footConstraintFromConfig(config);
      EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument) << weight.name << " = " << value;
      EXPECT_THAT(refused.status().message(), HasSubstr(absl::StrCat("model_settings.foot_constraint.", weight.name, " is ")));
    }
    FootConstraintConfig positive;
    positive.*weight.field = 1.0e-3;
    EXPECT_TRUE(footConstraintFromConfig(positive).ok()) << weight.name << " = 1e-3";
  }
}

TEST(ModelSettingsFromConfig, TheRetiredKeysOfTheBlocksAreRefusedNamingTheirReplacement) {
  const absl::StatusOr<humanoid_mpc_config::TaskFile> folder =
      nproto::ParseTextproto<humanoid_mpc_config::TaskFile>("model_settings {\n  modelFolderCppAd: \"build\"\n}\n", "task.textproto");
  EXPECT_EQ(folder.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(folder.status().message(), HasSubstr("task.textproto:2:3: 'modelFolderCppAd' is retired: the CppAD libraries are built"));

  const absl::StatusOr<humanoid_mpc_config::TaskFile> ground =
      nproto::ParseTextproto<humanoid_mpc_config::TaskFile>("contact_implicit { terrain_height: 0.1 }\n", "task.textproto");
  EXPECT_EQ(ground.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(ground.status().message(), HasSubstr("'terrain_height' is retired: the ground is the top-level terrain_height"));
}

TEST(ModelSettingsFromConfig, ABlockParsesIntoItsSettings) {
  const absl::StatusOr<mpc_config::TaskFile> task = parseTask(
      "model_settings {\n"
      "  robot_name: \"two_joints\"\n"
      "  foot_constraint { soft_constraint_weight: 250 stance_constraint: \"position\" }\n"
      "  contact_names_6dof: \"foot_l\"\n"
      "  contact_names_6dof: \"foot_r\"\n"
      "}\n"
      "terrain_height: -0.5\n"
      "contact_implicit { slip_weight: 7.25 }\n"
      "nominal_foothold { step_width: 0.45 }\n");
  ASSERT_TRUE(task.ok()) << task.status();
  const absl::StatusOr<ModelSettings> settings = ModelSettings::Create(*task, twoJointUrdf(), "test_", /*verbose=*/false);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_EQ(settings->footConstraintConfig.softConstraintWeight, 250.0);
  EXPECT_FALSE(settings->footConstraintConfig.constrainOrientation);
  EXPECT_EQ(settings->footConstraintConfig.positionErrorGain_z, ModelSettings::FootConstraintConfig{}.positionErrorGain_z);
  EXPECT_THAT(settings->contactNames6DoF, ::testing::ElementsAre("foot_l", "foot_r"));
  EXPECT_EQ(settings->terrainHeight, -0.5);
  EXPECT_EQ(settings->contactImplicitConfig.slipWeight, 7.25);
  EXPECT_EQ(settings->contactImplicitConfig.complementarityWeight, ModelSettings::ContactImplicitConfig{}.complementarityWeight);
  EXPECT_EQ(settings->nominalFootholdConfig.stepWidth, 0.45);
}

}  // namespace
}  // namespace ocs2::humanoid
