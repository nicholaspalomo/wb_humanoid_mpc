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
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
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
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "gtest/gtest.h"

#include "legged_common_mpc/common/ContactArray.h"
#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/StateLayout.h"
#include "legged_common_mpc/common/StatusMacros.h"
#include "legged_common_mpc/config/model/ModelSettingsFromConfig.h"
#include "legged_mpc_config/contact_implicit_config.nproto.h"
#include "legged_mpc_config/model_settings_config.nproto.h"
#include "legged_mpc_config/nominal_foothold_config.nproto.h"
#include "legged_mpc_config/task_file.nproto.h"
#include "legged_mpc_config/task_file.nproto.pb.h"
#include "legged_mpc_config/task_file.pb.h"
#include "nproto/Fnv1a64Constants.h"
#include "nproto/Textproto.h"

/**
 * ModelSettings::Create() of the typed task file and the conversions of its
 * blocks, on a two-joint robot written by the test: what it builds, what it
 * refuses, that every field of the schema reaches the settings, and that the
 * schema's defaults are the settings' defaults.
 */
namespace ocs2::legged {
namespace {

using ::google::protobuf::Descriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::Message;
using ::google::protobuf::Reflection;
using ::testing::HasSubstr;

constexpr char kInertial[] =
    R"(<inertial><mass value="1.0"/><origin xyz="0 0 0"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>)";

/** A floating base with two revolute joints, `shoulder` and `hip`; $0 is each
 * link's inertial block. */
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
  const std::string path =
      absl::StrCat(::testing::TempDir(), "/two_joints.urdf");
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
  const absl::StatusOr<legged_mpc_config::TaskFile> message =
      nproto::ParseTextproto<legged_mpc_config::TaskFile>(text,
                                                          "task.textproto");
  if (!message.ok()) return message.status();
  mpc_config::TaskFile task;
  RETURN_IF_ERROR(mpc_config::FromProto(*message, &task));
  return task;
}

/** `value` exactly: its bits, so that two values that print alike in decimal
 * still differ. */
std::string text(double value) {
  return absl::StrCat(std::bit_cast<uint64_t>(value));
}

std::string describe(const ModelSettings::FootConstraintConfig& config) {
  return absl::StrJoin(
      {text(config.positionErrorGain_z), text(config.orientationErrorGain),
       text(config.linearVelocityErrorGain_z),
       text(config.linearVelocityErrorGain_xy),
       text(config.angularVelocityErrorGain),
       text(config.linearAccelerationErrorGain_z),
       text(config.linearAccelerationErrorGain_xy),
       text(config.angularAccelerationErrorGain),
       text(config.softConstraintWeight),
       text(config.normalVelocitySoftConstraintWeight),
       std::string(config.constrainAttitude ? "true" : "false"),
       std::string(config.constrainYawRateAboutContactNormal ? "true"
                                                             : "false")},
      " ");
}

std::string describe(const ModelSettings::ContactImplicitConfig& config) {
  return absl::StrJoin(
      {text(config.complementarityWeight), text(config.slipWeight),
       text(config.scalePosContactRtSurfaceEwrtWorldZ),
       text(config.scaleLinvelContactRtWorldEwrtWorldXy),
       text(config.scaleAngvelContactRtWorldEwrtWorldZ),
       text(config.penetrationWeight), text(config.gapSmoothing)},
      " ");
}

std::string describe(absl::Span<const ModelSettings::PointContact> contacts) {
  std::vector<std::string> described;
  for (const ModelSettings::PointContact& contact : contacts) {
    described.push_back(absl::StrCat(
        contact.name, "@", contact.parentJointName, "+",
        text(contact.posContactRtLinkEwrtLink.x()), ",",
        text(contact.posContactRtLinkEwrtLink.y()), ",",
        text(contact.posContactRtLinkEwrtLink.z()), ":",
        static_cast<int>(contact.kind), "/", text(contact.wheelRadius)));
  }
  return absl::StrJoin(described, ";");
}

/** Every value of `settings`, or the error that `settings` is, as one string
 * per field. */
std::string describe(const absl::StatusOr<ModelSettings>& settings) {
  if (!settings.ok()) {
    return settings.status().ToString();
  }
  std::vector<std::string> indices;
  for (const size_t index : settings->mpcModelToFullJointsIndices) {
    indices.push_back(absl::StrCat(index));
  }
  return absl::StrJoin(
      {settings->robotName,
       std::string(settings->verboseCppAd ? "true" : "false"),
       std::string(settings->recompileLibrariesCppAd ? "true" : "false"),
       settings->modelFolderCppAd,
       text(settings->phaseTransitionStanceTime),
       absl::StrJoin(settings->fullJointNames, ","),
       absl::StrJoin(settings->fixedJointNames, ","),
       absl::StrJoin(settings->contactNames6DoF, ","),
       absl::StrJoin(settings->contactNames3DoF, ","),
       describe(settings->pointContacts),
       absl::StrJoin(settings->contactParentJointNames, ","),
       absl::StrJoin(settings->mpcModelJointNames, ","),
       absl::StrJoin(indices, ","),
       absl::StrJoin(settings->contactNames, ","),
       absl::StrCat(settings->mpc_joint_dim, "/", settings->full_joint_dim),
       std::string(settings->hasArmSwingJoints ? "true" : "false"),
       absl::StrCat(
           settings->j_l_shoulder_y_index, ",", settings->j_r_shoulder_y_index,
           ",", settings->j_l_elbow_y_index, ",", settings->j_r_elbow_y_index),
       describe(settings->footConstraintConfig),
       describe(settings->contactImplicitConfig),
       text(settings->nominalFootholdConfig.nomPosLfootRtRfootEwrtHeadingY),
       text(settings->posSurfaceRtWorldEwrtWorldZ)},
      "\n");
}

TEST(ModelSettingsFromConfig, LoadsTheActiveJointsOfTheUrdf) {
  const absl::StatusOr<ModelSettings> settings = ModelSettings::Create(
      twoJointTask(), twoJointUrdf(), "test_", /*verbose=*/false);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_EQ(settings->robotName, "two_joints");
  EXPECT_EQ(settings->mpc_joint_dim, 2U);
  EXPECT_EQ(settings->full_joint_dim, 2U);
  EXPECT_THAT(settings->mpcModelJointNames,
              ::testing::UnorderedElementsAre("shoulder", "hip"));
  EXPECT_EQ(settings->jointIndexMap.size(), 2U);
  EXPECT_FALSE(settings->hasArmSwingJoints)
      << "a file without arm_joint_names disables the arm swing";
  EXPECT_EQ(
      settings->modelFolderCppAd,
      absl::StrCat("cppad_code_gen/cppad_test_two_joints/", kStateLayoutTag));
  EXPECT_EQ(settings->phaseTransitionStanceTime, 0.25);
}

TEST(ModelSettingsFromConfig, AFixedJointLeavesTheMpcModel) {
  mpc_config::TaskFile task = twoJointTask();
  task.model_settings.fixed_joint_names = {"shoulder"};
  task.model_settings.contact_names_6dof = {"foot"};
  const absl::StatusOr<ModelSettings> settings =
      ModelSettings::Create(task, twoJointUrdf(), "test_", /*verbose=*/true);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_THAT(settings->mpcModelJointNames, ::testing::ElementsAre("hip"));
  EXPECT_EQ(settings->mpcModelToFullJointsIndices.size(), 1U);
  EXPECT_EQ(
      settings->fullJointNames.at(settings->mpcModelToFullJointsIndices[0]),
      "hip");
  EXPECT_THAT(settings->contactNames, ::testing::ElementsAre("foot"));
}

TEST(ModelSettingsFromConfig, FixingEveryJointIsRefused) {
  mpc_config::TaskFile task = twoJointTask();
  task.model_settings.fixed_joint_names = {"shoulder", "hip"};
  const absl::StatusOr<ModelSettings> settings =
      ModelSettings::Create(task, twoJointUrdf(), "test_", /*verbose=*/false);
  EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(settings.status().message(),
              HasSubstr("model_settings.fixed_joint_names"));
  EXPECT_THAT(settings.status().message(),
              HasSubstr("at least one active joint"));
}

TEST(ModelSettingsFromConfig, AnUnknownOrRepeatedFixedJointIsRefusedNamingIt) {
  // A misspelled fixed joint would otherwise stay active without a word.
  mpc_config::TaskFile unknown = twoJointTask();
  unknown.model_settings.fixed_joint_names = {"shoulderr"};
  const absl::StatusOr<ModelSettings> misspelled = ModelSettings::Create(
      unknown, twoJointUrdf(), "test_", /*verbose=*/false);
  EXPECT_EQ(misspelled.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(misspelled.status().message(),
              HasSubstr("model_settings.fixed_joint_names names shoulderr, "
                        "which the URDF does not have"));
  mpc_config::TaskFile repeated = twoJointTask();
  repeated.model_settings.fixed_joint_names = {"shoulder", "shoulder"};
  const absl::StatusOr<ModelSettings> twice = ModelSettings::Create(
      repeated, twoJointUrdf(), "test_", /*verbose=*/false);
  EXPECT_EQ(twice.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(twice.status().message(),
              HasSubstr("names shoulder more than once"));
}

TEST(ModelSettingsFromConfig,
     ANegativeOrNanGainTimeOrStepWidthIsRefusedNamingIt) {
  using FootConstraintConfig =
      mpc_config::ModelSettingsConfig::FootConstraintConfig;
  struct Gain {
    const char* absl_nonnull name;
    double FootConstraintConfig::*absl_nonnull field;
  };
  const std::vector<Gain> gains = {
      {.name = "position_error_gain_z",
       .field = &FootConstraintConfig::position_error_gain_z},
      {.name = "orientation_error_gain",
       .field = &FootConstraintConfig::orientation_error_gain},
      {.name = "linear_velocity_error_gain_z",
       .field = &FootConstraintConfig::linear_velocity_error_gain_z},
      {.name = "linear_velocity_error_gain_xy",
       .field = &FootConstraintConfig::linear_velocity_error_gain_xy},
      {.name = "angular_velocity_error_gain",
       .field = &FootConstraintConfig::angular_velocity_error_gain},
      {.name = "linear_acceleration_error_gain_z",
       .field = &FootConstraintConfig::linear_acceleration_error_gain_z},
      {.name = "linear_acceleration_error_gain_xy",
       .field = &FootConstraintConfig::linear_acceleration_error_gain_xy},
      {.name = "angular_acceleration_error_gain",
       .field = &FootConstraintConfig::angular_acceleration_error_gain},
  };
  for (const Gain& gain : gains) {
    for (const double value :
         {-1.0, std::numeric_limits<double>::quiet_NaN()}) {
      FootConstraintConfig config;
      config.*gain.field = value;
      EXPECT_THAT(footConstraintFromConfig(config).status().message(),
                  HasSubstr(absl::StrCat("model_settings.foot_constraint.",
                                         gain.name, ": ")))
          << value;
    }
    FootConstraintConfig zero;
    zero.*gain.field = 0.0;
    EXPECT_TRUE(footConstraintFromConfig(zero).ok())
        << gain.name << " = 0 switches the error feedback off";
  }
  for (const double value : {-1.0, std::numeric_limits<double>::quiet_NaN()}) {
    mpc_config::TaskFile stance = twoJointTask();
    stance.model_settings.phase_transition_stance_time = value;
    EXPECT_THAT(ModelSettings::Create(stance, twoJointUrdf(), "test_",
                                      /*verbose=*/false)
                    .status()
                    .message(),
                HasSubstr("model_settings.phase_transition_stance_time: "));
    mpc_config::NominalFootholdConfig foothold;
    foothold.nom_pos_lfoot_rt_rfoot_ewrt_heading_y = value;
    EXPECT_THAT(
        nominalFootholdFromConfig(foothold).status().message(),
        HasSubstr("nominal_foothold.nom_pos_lfoot_rt_rfoot_ewrt_heading_y: "));
  }
  mpc_config::TaskFile terrain = twoJointTask();
  terrain.pos_surface_rt_world_ewrt_world_z =
      std::numeric_limits<double>::quiet_NaN();
  EXPECT_THAT(
      ModelSettings::Create(terrain, twoJointUrdf(), "test_", /*verbose=*/false)
          .status()
          .message(),
      HasSubstr("pos_surface_rt_world_ewrt_world_z: "));
}

TEST(ModelSettingsFromConfig,
     AnArmJointThatIsNotAnActiveJointIsRefusedNamingIt) {
  mpc_config::TaskFile misspelled = twoJointTask();
  misspelled.model_settings.arm_joint_names = {.left_shoulder_y = "shoulder",
                                               .right_shoulder_y = "shoulder",
                                               .left_elbow_y = "shoulder",
                                               .right_elbow_y = "elbow"};
  const absl::StatusOr<ModelSettings> refused = ModelSettings::Create(
      misspelled, twoJointUrdf(), "test_", /*verbose=*/false);
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      refused.status().message(),
      HasSubstr("model_settings.arm_joint_names.right_elbow_y is 'elbow'"));

  mpc_config::TaskFile fixedArm = twoJointTask();
  fixedArm.model_settings.fixed_joint_names = {"shoulder"};
  fixedArm.model_settings.arm_joint_names = {.left_shoulder_y = "shoulder",
                                             .right_shoulder_y = "hip",
                                             .left_elbow_y = "hip",
                                             .right_elbow_y = "hip"};
  const absl::StatusOr<ModelSettings> fixed = ModelSettings::Create(
      fixedArm, twoJointUrdf(), "test_", /*verbose=*/false);
  EXPECT_EQ(fixed.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(fixed.status().message(), HasSubstr("left_shoulder_y"));

  mpc_config::TaskFile partial = twoJointTask();
  partial.model_settings.arm_joint_names.left_shoulder_y = "shoulder";
  EXPECT_EQ(
      ModelSettings::Create(partial, twoJointUrdf(), "test_", /*verbose=*/false)
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument)
      << "naming one arm joint of the four is refused";

  mpc_config::TaskFile valid = twoJointTask();
  valid.model_settings.arm_joint_names = {.left_shoulder_y = "shoulder",
                                          .right_shoulder_y = "hip",
                                          .left_elbow_y = "shoulder",
                                          .right_elbow_y = "hip"};
  const absl::StatusOr<ModelSettings> resolved =
      ModelSettings::Create(valid, twoJointUrdf(), "test_", /*verbose=*/false);
  ASSERT_TRUE(resolved.ok()) << resolved.status();
  EXPECT_TRUE(resolved->hasArmSwingJoints);
  EXPECT_EQ(resolved->j_l_shoulder_y_index,
            resolved->jointIndexMap.at("shoulder"));
  EXPECT_EQ(resolved->j_r_shoulder_y_index, resolved->jointIndexMap.at("hip"));
  EXPECT_EQ(resolved->j_l_elbow_y_index,
            resolved->jointIndexMap.at("shoulder"));
  EXPECT_EQ(resolved->j_r_elbow_y_index, resolved->jointIndexMap.at("hip"));
}

TEST(ModelSettingsFromConfig, AUrdfThatCannotBeReadIsRefusedNamingIt) {
  const std::string missing =
      absl::StrCat(::testing::TempDir(), "/no_such_robot.urdf");
  const absl::StatusOr<ModelSettings> settings = ModelSettings::Create(
      twoJointTask(), missing, "test_", /*verbose=*/false);
  EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(settings.status().message(), HasSubstr("no_such_robot.urdf"));
}

TEST(ModelSettingsFromConfig, TheSchemaDefaultsAreTheSettingsDefaults) {
  const absl::StatusOr<ModelSettings::FootConstraintConfig> footConstraint =
      footConstraintFromConfig(
          mpc_config::ModelSettingsConfig::FootConstraintConfig{});
  ASSERT_TRUE(footConstraint.ok()) << footConstraint.status();
  EXPECT_EQ(describe(*footConstraint),
            describe(ModelSettings::FootConstraintConfig{}));
  EXPECT_EQ(
      describe(contactImplicitFromConfig(mpc_config::ContactImplicitConfig{})),
      describe(ModelSettings::ContactImplicitConfig{}));
  const absl::StatusOr<ModelSettings::NominalFootholdConfig> nominalFoothold =
      nominalFootholdFromConfig(mpc_config::NominalFootholdConfig{});
  ASSERT_TRUE(nominalFoothold.ok()) << nominalFoothold.status();
  EXPECT_EQ(text(nominalFoothold->nomPosLfootRtRfootEwrtHeadingY),
            text(ModelSettings::NominalFootholdConfig{}
                     .nomPosLfootRtRfootEwrtHeadingY));
}

/** A value of `field` other than its default: a number scaled and shifted, a
 * bool flipped, a joint name of the robot. */
void setNonDefault(const FieldDescriptor* absl_nonnull field,
                   Message* absl_nonnull message) {
  const Reflection* absl_nonnull reflection = message->GetReflection();
  switch (field->cpp_type()) {
    case FieldDescriptor::CPPTYPE_DOUBLE:
      reflection->SetDouble(message, field,
                            field->default_value_double() * 1.5 + 0.25);
      break;
    case FieldDescriptor::CPPTYPE_BOOL:
      reflection->SetBool(message, field, !field->default_value_bool());
      break;
    case FieldDescriptor::CPPTYPE_STRING:
      field->is_repeated() ? reflection->AddString(message, field, "shoulder")
                           : reflection->SetString(message, field, "shoulder");
      break;
    default:
      ADD_FAILURE() << field->full_name()
                    << " is of a type this test does not mutate";
      break;
  }
}

/**
 * A new element of the repeated block `field` of `message`, its string fields
 * set to names other than setNonDefault()'s, so that the element is valid and a
 * change of any one of its fields shows.
 */
Message* absl_nonnull elementWithNamesSet(
    const FieldDescriptor* absl_nonnull field, Message* absl_nonnull message) {
  Message* absl_nonnull element =
      message->GetReflection()->AddMessage(message, field);
  const Descriptor* absl_nonnull descriptor = element->GetDescriptor();
  for (int index = 0; index < descriptor->field_count(); ++index) {
    const FieldDescriptor* absl_nonnull member = descriptor->field(index);
    if (member->cpp_type() == FieldDescriptor::CPPTYPE_STRING &&
        !member->is_repeated()) {
      element->GetReflection()->SetString(element, member,
                                          absl::StrCat("element_", index));
    }
  }
  return element;
}

/** The dotted paths, `prefix` first, of every scalar or repeated scalar field
 * below `descriptor`. */
void collectLeaves(const Descriptor* absl_nonnull descriptor,
                   absl::string_view prefix,
                   std::vector<std::string>& leaves) {
  for (int index = 0; index < descriptor->field_count(); ++index) {
    const FieldDescriptor* absl_nonnull field = descriptor->field(index);
    const std::string path = absl::StrCat(prefix, ".", field->name());
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      // A repeated block is walked through one element of it
      // (elementWithNamesSet()).
      collectLeaves(field->message_type(), path, leaves);
    } else {
      leaves.push_back(path);
    }
  }
}

TEST(ModelSettingsFromConfig, EveryFieldOfTheSchemaReachesTheSettings) {
  // The blocks ModelSettings is built from: a field that no conversion reads
  // leaves the settings as they were.
  const Descriptor* absl_nonnull task =
      legged_mpc_config::TaskFile::descriptor();
  std::vector<std::string> leaves = {"pos_surface_rt_world_ewrt_world_z"};
  for (const char* absl_nonnull block :
       {"model_settings", "contact_implicit", "nominal_foothold"}) {
    const FieldDescriptor* absl_nullable field = task->FindFieldByName(block);
    ASSERT_NE(field, nullptr) << block;
    collectLeaves(field->message_type(), block, leaves);
  }

  const std::string urdf = twoJointUrdf();
  const std::string baseline = describe(ModelSettings::Create(
      mpc_config::TaskFile{}, urdf, "test_", /*verbose=*/false));
  for (const std::string& leaf : leaves) {
    legged_mpc_config::TaskFile message;
    Message* absl_nonnull current = &message;
    const std::vector<std::string> names = absl::StrSplit(leaf, '.');
    for (size_t depth = 0; depth < names.size(); ++depth) {
      const FieldDescriptor* absl_nullable field =
          current->GetDescriptor()->FindFieldByName(names[depth]);
      ASSERT_NE(field, nullptr) << leaf;
      if (depth + 1 < names.size()) {
        current =
            field->is_repeated()
                ? elementWithNamesSet(field, current)
                : current->GetReflection()->MutableMessage(current, field);
      } else {
        setNonDefault(field, current);
      }
    }
    mpc_config::TaskFile changed;
    ASSERT_TRUE(mpc_config::FromProto(message, &changed).ok());
    EXPECT_NE(describe(ModelSettings::Create(changed, urdf, "test_",
                                             /*verbose=*/false)),
              baseline)
        << leaf << " does not reach ModelSettings";
  }
  EXPECT_GT(leaves.size(), 4U) << "the walk found the fields of the blocks";
}

TEST(ModelSettingsFromConfig,
     EachStanceConstraintNameSetsTheRowsTheBooleansItReplacedSet) {
  // The parity of the two retired booleans: (constrain_orientation,
  // constrain_yaw_rate_about_contact_normal) = (false, false) is "position",
  // (true, false) "position_and_tilt" and (true, true)
  // "position_and_orientation"; a yaw-rate row without the orientation rows,
  // (false, true), did nothing and has no name.
  struct Rows {
    const char* absl_nonnull name;
    bool constrainAttitude;
    bool constrainYawRateAboutContactNormal;
  };
  const std::vector<Rows> rows = {{.name = "position",
                                   .constrainAttitude = false,
                                   .constrainYawRateAboutContactNormal = false},
                                  {.name = "position_and_tilt",
                                   .constrainAttitude = true,
                                   .constrainYawRateAboutContactNormal = false},
                                  {.name = "position_and_orientation",
                                   .constrainAttitude = true,
                                   .constrainYawRateAboutContactNormal = true}};
  std::vector<std::string> names;
  for (const Rows& expected : rows) {
    mpc_config::ModelSettingsConfig::FootConstraintConfig config;
    config.stance_constraint = expected.name;
    const absl::StatusOr<ModelSettings::FootConstraintConfig> converted =
        footConstraintFromConfig(config);
    ASSERT_TRUE(converted.ok()) << converted.status();
    EXPECT_EQ(converted->constrainAttitude, expected.constrainAttitude)
        << expected.name;
    EXPECT_EQ(converted->constrainYawRateAboutContactNormal,
              expected.constrainYawRateAboutContactNormal)
        << expected.name;
    EXPECT_EQ(stanceConstraintName(*converted), expected.name);
    names.emplace_back(expected.name);
  }
  EXPECT_EQ(stanceConstraintNames(), names);
  ModelSettings::FootConstraintConfig yawRateWithoutAttitude;
  yawRateWithoutAttitude.constrainAttitude = false;
  yawRateWithoutAttitude.constrainYawRateAboutContactNormal = true;
  EXPECT_EQ(stanceConstraintName(yawRateWithoutAttitude), "position")
      << "three rows have no yaw-rate row";
  EXPECT_EQ(
      mpc_config::ModelSettingsConfig::FootConstraintConfig{}.stance_constraint,
      "position_and_tilt")
      << "the default is the booleans' defaults, (true, false)";

  mpc_config::ModelSettingsConfig::FootConstraintConfig unknown;
  unknown.stance_constraint = "full_pose";
  const absl::StatusOr<ModelSettings::FootConstraintConfig> refused =
      footConstraintFromConfig(unknown);
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      refused.status().message(),
      HasSubstr(
          "model_settings.foot_constraint.stance_constraint is 'full_pose'"));
  EXPECT_THAT(
      refused.status().message(),
      HasSubstr("position, position_and_tilt, position_and_orientation"));
}

TEST(ModelSettingsFromConfig,
     ASoftFootWeightThatIsNotAFinitePositiveNumberIsRefusedNamingIt) {
  using FootConstraintConfig =
      mpc_config::ModelSettingsConfig::FootConstraintConfig;
  struct Weight {
    const char* absl_nonnull name;
    double FootConstraintConfig::*absl_nonnull field;
  };
  const std::vector<Weight> weights = {
      {.name = "soft_constraint_weight",
       .field = &FootConstraintConfig::soft_constraint_weight},
      {.name = "normal_velocity_soft_constraint_weight",
       .field = &FootConstraintConfig::normal_velocity_soft_constraint_weight}};
  for (const Weight& weight : weights) {
    for (const double value :
         {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::infinity()}) {
      FootConstraintConfig config;
      config.*weight.field = value;
      const absl::StatusOr<ModelSettings::FootConstraintConfig> refused =
          footConstraintFromConfig(config);
      EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument)
          << weight.name << " = " << value;
      EXPECT_THAT(refused.status().message(),
                  HasSubstr(absl::StrCat("model_settings.foot_constraint.",
                                         weight.name, " is ")));
    }
    FootConstraintConfig positive;
    positive.*weight.field = 1.0e-3;
    EXPECT_TRUE(footConstraintFromConfig(positive).ok())
        << weight.name << " = 1e-3";
  }
}

TEST(ModelSettingsFromConfig,
     TheRetiredKeysOfTheBlocksAreRefusedNamingTheirReplacement) {
  const absl::StatusOr<legged_mpc_config::TaskFile> folder =
      nproto::ParseTextproto<legged_mpc_config::TaskFile>(
          "model_settings {\n  modelFolderCppAd: \"build\"\n}\n",
          "task.textproto");
  EXPECT_EQ(folder.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(folder.status().message(),
              HasSubstr("task.textproto:2:3: 'modelFolderCppAd' is retired: "
                        "the CppAD libraries are built"));

  const absl::StatusOr<legged_mpc_config::TaskFile> ground =
      nproto::ParseTextproto<legged_mpc_config::TaskFile>(
          "contact_implicit { terrain_height: 0.1 }\n", "task.textproto");
  EXPECT_EQ(ground.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(ground.status().message(),
              HasSubstr("'terrain_height' is retired: the ground is the "
                        "top-level pos_surface_rt_world_ewrt_world_z"));
}

TEST(ModelSettingsFromConfig, ABlockParsesIntoItsSettings) {
  const absl::StatusOr<mpc_config::TaskFile> task = parseTask(
      "model_settings {\n"
      "  robot_name: \"two_joints\"\n"
      "  foot_constraint { soft_constraint_weight: 250 stance_constraint: "
      "\"position\" }\n"
      "  contact_names_6dof: \"foot_l\"\n"
      "  contact_names_6dof: \"foot_r\"\n"
      "}\n"
      "pos_surface_rt_world_ewrt_world_z: -0.5\n"
      "contact_implicit { slip_weight: 7.25 }\n"
      "nominal_foothold { nom_pos_lfoot_rt_rfoot_ewrt_heading_y: 0.45 }\n");
  ASSERT_TRUE(task.ok()) << task.status();
  const absl::StatusOr<ModelSettings> settings =
      ModelSettings::Create(*task, twoJointUrdf(), "test_", /*verbose=*/false);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_EQ(settings->footConstraintConfig.softConstraintWeight, 250.0);
  EXPECT_FALSE(settings->footConstraintConfig.constrainAttitude);
  EXPECT_EQ(settings->footConstraintConfig.positionErrorGain_z,
            ModelSettings::FootConstraintConfig{}.positionErrorGain_z);
  EXPECT_THAT(settings->contactNames6DoF,
              ::testing::ElementsAre("foot_l", "foot_r"));
  EXPECT_EQ(settings->posSurfaceRtWorldEwrtWorldZ, -0.5);
  EXPECT_EQ(settings->contactImplicitConfig.slipWeight, 7.25);
  EXPECT_EQ(settings->contactImplicitConfig.complementarityWeight,
            ModelSettings::ContactImplicitConfig{}.complementarityWeight);
  EXPECT_EQ(settings->nominalFootholdConfig.nomPosLfootRtRfootEwrtHeadingY,
            0.45);
}

TEST(ModelSettingsFromConfig, ThePointContactsComeBeforeTheFeet) {
  const absl::StatusOr<mpc_config::TaskFile> task = parseTask(
      "model_settings {\n"
      "  contact_names_6dof: \"foot_l\"\n"
      "  contact_names_6dof: \"foot_r\"\n"
      "  contacts_3dof {\n"
      "    name: \"hand\"\n"
      "    parent_joint_name: \"joint_a\"\n"
      "    pos_contact_rt_link_ewrt_link { x: 0.1 z: -0.2 }\n"
      "  }\n"
      "}\n");
  ASSERT_TRUE(task.ok()) << task.status();
  const absl::StatusOr<ModelSettings> settings =
      ModelSettings::Create(*task, twoJointUrdf(), "test_", /*verbose=*/false);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_THAT(settings->contactNames,
              ::testing::ElementsAre("hand", "foot_l", "foot_r"));
  EXPECT_THAT(settings->contactNames3DoF, ::testing::ElementsAre("hand"));
  ASSERT_EQ(settings->pointContacts.size(), 1U);
  EXPECT_EQ(settings->pointContacts[0].parentJointName, "joint_a");
  EXPECT_EQ(settings->pointContacts[0].posContactRtLinkEwrtLink,
            vector3_t(0.1, 0.0, -0.2));
  EXPECT_TRUE(settings->isPointContact(0));
  EXPECT_FALSE(settings->isPointContact(1));
  EXPECT_FALSE(settings->isBipedContactSet());
}

// contactParentJointNames has one entry per 6-DoF contact, while contactNames
// puts the 3-DoF contacts first. The MuJoCo backend resolves each contact's
// body from the parent joint at the contact's own index, so the parents have
// to be in contact order: paired by index, the hand took foot_l's joint and
// foot_r got none.
TEST(ModelSettingsFromConfig, TheParentJointsFollowTheContactOrder) {
  const absl::StatusOr<mpc_config::TaskFile> task = parseTask(
      "model_settings {\n"
      "  contact_names_6dof: \"foot_l\"\n"
      "  contact_names_6dof: \"foot_r\"\n"
      "  contact_parent_joint_names: \"hip\"\n"
      "  contact_parent_joint_names: \"shoulder\"\n"
      "  contacts_3dof {\n"
      "    name: \"hand\"\n"
      "    parent_joint_name: \"shoulder\"\n"
      "  }\n"
      "}\n");
  ASSERT_TRUE(task.ok()) << task.status();
  const absl::StatusOr<ModelSettings> settings =
      ModelSettings::Create(*task, twoJointUrdf(), "test_", /*verbose=*/false);
  ASSERT_TRUE(settings.ok()) << settings.status();
  ASSERT_THAT(settings->contactNames,
              ::testing::ElementsAre("hand", "foot_l", "foot_r"));
  EXPECT_THAT(settings->contactParentJointNamesInContactOrder(),
              ::testing::ElementsAre("shoulder", "hip", "shoulder"));
}

// A 6-DoF contact without an entry in contactParentJointNames gets an empty
// parent, which the MuJoCo backend resolves by the frame name instead.
TEST(ModelSettingsFromConfig, AMissingParentJointIsEmptyInContactOrder) {
  const absl::StatusOr<mpc_config::TaskFile> task = parseTask(
      "model_settings {\n"
      "  contact_names_6dof: \"foot_l\"\n"
      "  contact_names_6dof: \"foot_r\"\n"
      "  contact_parent_joint_names: \"hip\"\n"
      "  contacts_3dof {\n"
      "    name: \"hand\"\n"
      "    parent_joint_name: \"shoulder\"\n"
      "  }\n"
      "}\n");
  ASSERT_TRUE(task.ok()) << task.status();
  const absl::StatusOr<ModelSettings> settings =
      ModelSettings::Create(*task, twoJointUrdf(), "test_", /*verbose=*/false);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_THAT(settings->contactParentJointNamesInContactOrder(),
              ::testing::ElementsAre("shoulder", "hip", std::string()));
}

TEST(ModelSettingsFromConfig, APointContactNeedsANameAParentAndAFiniteOffset) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {R"(parent_joint_name: "joint_a")", "contacts_3dof[0] needs a name"},
      {R"(name: "hand")", "contacts_3dof[0] needs a name"},
      {R"(name: "foot_l" parent_joint_name: "joint_a")",
       "contacts_3dof[0].name is 'foot_l'"},
      {"name: \"hand\" parent_joint_name: \"joint_a\" "
       "pos_contact_rt_link_ewrt_link { y: nan }",
       "contacts_3dof[0].pos_contact_rt_link_ewrt_link.y"},
  };
  for (const std::pair<std::string, std::string>& entry : cases) {
    const absl::StatusOr<mpc_config::TaskFile> task = parseTask(
        absl::StrCat("model_settings {\n  contact_names_6dof: \"foot_l\"\n"
                     "  contact_names_6dof: \"foot_r\"\n  contacts_3dof { ",
                     entry.first, " }\n}\n"));
    ASSERT_TRUE(task.ok()) << task.status();
    const absl::StatusOr<ModelSettings> settings = ModelSettings::Create(
        *task, twoJointUrdf(), "test_", /*verbose=*/false);
    ASSERT_FALSE(settings.ok()) << entry.first;
    EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(settings.status().message(), HasSubstr(entry.second));
  }
}

/** A task file of the two-joint robot with two feet and `numPointContacts`
 * 3-DoF contacts on the hip. */
mpc_config::TaskFile taskWithPointContacts(size_t numPointContacts) {
  mpc_config::TaskFile task = twoJointTask();
  task.model_settings.contact_names_6dof = {"foot_l", "foot_r"};
  for (size_t i = 0; i < numPointContacts; ++i) {
    mpc_config::ModelSettingsConfig::PointContact contact;
    contact.name = absl::StrCat("point_", i);
    contact.parent_joint_name = "hip";
    task.model_settings.contacts_3dof.push_back(contact);
  }
  return task;
}

// A mode number has one bit per contact (common/ContactMask.h): a set of
// kMaxNumContacts contacts is built, one more is refused naming the count.
TEST(ModelSettingsFromConfig, MoreContactsThanAModeNumberHoldsAreRefused) {
  constexpr size_t kNumFeet = 2;
  const absl::StatusOr<ModelSettings> largest =
      ModelSettings::Create(taskWithPointContacts(kMaxNumContacts - kNumFeet),
                            twoJointUrdf(), "test_", /*verbose=*/false);
  ASSERT_TRUE(largest.ok()) << largest.status();
  EXPECT_EQ(largest->numContacts(), kMaxNumContacts);
  EXPECT_FALSE(largest->isBipedContactSet());

  const absl::StatusOr<ModelSettings> refused = ModelSettings::Create(
      taskWithPointContacts(kMaxNumContacts - kNumFeet + 1), twoJointUrdf(),
      "test_", /*verbose=*/false);
  ASSERT_FALSE(refused.ok());
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(refused.status().message(),
              HasSubstr("the contact set has 33 contacts"));
  EXPECT_THAT(refused.status().message(),
              HasSubstr("at most 32 are supported"));
}

// A set of point contacts alone, such as a point-footed quadruped's, is valid:
// the contact set needs no 6-DoF contact.
TEST(ModelSettingsFromConfig, ASetOfPointContactsAloneIsBuilt) {
  constexpr size_t kNumPoints = 4;
  mpc_config::TaskFile task = taskWithPointContacts(kNumPoints);
  task.model_settings.contact_names_6dof = {};
  const absl::StatusOr<ModelSettings> settings =
      ModelSettings::Create(task, twoJointUrdf(), "test_", /*verbose=*/false);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_EQ(settings->numContacts(), kNumPoints);
  EXPECT_THAT(
      settings->contactNames,
      ::testing::ElementsAre("point_0", "point_1", "point_2", "point_3"));
  EXPECT_TRUE(settings->contactNames6DoF.empty());
  for (size_t i = 0; i < kNumPoints; ++i) {
    EXPECT_TRUE(settings->isPointContact(i)) << i;
  }
  EXPECT_FALSE(settings->isBipedContactSet());
  EXPECT_THAT(settings->contactParentJointNamesInContactOrder(),
              ::testing::ElementsAre("hip", "hip", "hip", "hip"));
}

TEST(ModelSettingsFromConfig, NumContactsCountsEveryContactOfTheSet) {
  const absl::StatusOr<ModelSettings> biped = ModelSettings::Create(
      taskWithPointContacts(/*numPointContacts=*/0), twoJointUrdf(), "test_",
      /*verbose=*/false);
  ASSERT_TRUE(biped.ok()) << biped.status();
  EXPECT_EQ(biped->numContacts(), 2U);
  EXPECT_TRUE(biped->isBipedContactSet());
}

// The rolling contacts on the LimX Tron1 wheel-foot biped, whose wheels turn
// on the continuous joints wheel_L_Joint and wheel_R_Joint.
constexpr char kTron1UrdfFile[] =
    "robot_models/limx_tron1/tron1_description/urdf/tron1.urdf";
// [m] Tron1's wheel radius, and the wheel center's offset along the wheel
// joint's axis.
constexpr double kTron1WheelRadius = 0.127;
constexpr double kTron1PosWheelRtLinkEwrtLinkY = 0.00707;

/** `relativePath` in the test's runfiles. */
std::string runfilesPath(absl::string_view relativePath) {
  const char* absl_nullable directory = std::getenv("TEST_SRCDIR");
  return directory == nullptr
             ? std::string(relativePath)
             : absl::StrCat(directory, "/_main/", relativePath);
}

/** A rolling contact `name` on the wheel joint `joint`, with Tron1's wheel. */
mpc_config::ModelSettingsConfig::RollingContact tron1Wheel(
    absl::string_view name, absl::string_view joint, double side) {
  mpc_config::ModelSettingsConfig::RollingContact contact;
  contact.name = std::string(name);
  contact.parent_joint_name = std::string(joint);
  contact.pos_wheel_rt_link_ewrt_link.y = side * kTron1PosWheelRtLinkEwrtLinkY;
  contact.wheel_radius = kTron1WheelRadius;
  return contact;
}

/** A leg `name` ending in `contacts`. */
mpc_config::ModelSettingsConfig::Leg legOf(absl::string_view name,
                                           std::vector<std::string> contacts) {
  mpc_config::ModelSettingsConfig::Leg leg;
  leg.name = std::string(name);
  leg.contacts = std::move(contacts);
  return leg;
}

/** Tron1 on its two wheels, the legs "left" and "right". */
mpc_config::TaskFile tron1Task() {
  mpc_config::TaskFile task;
  task.model_settings.robot_name = "tron1";
  task.model_settings.contacts_rolling = {
      tron1Wheel("wheel_L_contact", "wheel_L_Joint", /*side=*/1.0),
      tron1Wheel("wheel_R_contact", "wheel_R_Joint", /*side=*/-1.0)};
  task.model_settings.legs = {legOf("left", {"wheel_L_contact"}),
                              legOf("right", {"wheel_R_contact"})};
  return task;
}

absl::StatusOr<ModelSettings> tron1Settings(const mpc_config::TaskFile& task) {
  return ModelSettings::Create(task, runfilesPath(kTron1UrdfFile), "test_",
                               /*verbose=*/false);
}

TEST(ModelSettingsFromConfig, TheWheelsOfTron1AreRollingForceOnlyContacts) {
  const absl::StatusOr<ModelSettings> settings = tron1Settings(tron1Task());
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_THAT(settings->contactNames,
              ::testing::ElementsAre("wheel_L_contact", "wheel_R_contact"));
  EXPECT_EQ(settings->contactNames3DoF, settings->contactNames);
  EXPECT_TRUE(settings->hasRollingContacts());
  EXPECT_FALSE(settings->isBipedContactSet());
  for (size_t i = 0; i < settings->numContacts(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_TRUE(settings->isPointContact(i)) << "a force-only contact";
    EXPECT_EQ(settings->contactKind(i), ModelSettings::ContactKind::kRolling);
    EXPECT_TRUE(settings->isRollingContact(i));
    EXPECT_EQ(settings->wheelRadius(i), kTron1WheelRadius);
    EXPECT_EQ(settings->pointContactEntry(i),
              absl::StrCat("model_settings.contacts_rolling[", i, "]"));
  }
  EXPECT_EQ(settings->pointContacts[0].posContactRtLinkEwrtLink,
            vector3_t(0.0, kTron1PosWheelRtLinkEwrtLinkY, 0.0));
  EXPECT_EQ(settings->pointContacts[1].parentJointName, "wheel_R_Joint");
  EXPECT_THAT(settings->rollingContactJointNames(),
              ::testing::ElementsAre("wheel_L_Joint", "wheel_R_Joint"));
  // The wheels are MPC joints, read off the URDF as any joint is.
  EXPECT_TRUE(settings->jointIndexMap.contains("wheel_L_Joint"));
  EXPECT_EQ(settings->mpc_joint_dim, 8U);

  // The legs end in the rolling contacts, their hips derived from the wheels'
  // joints.
  ASSERT_EQ(settings->legs.size(), 2U);
  EXPECT_TRUE(settings->isLeggedContactSet());
  EXPECT_EQ(settings->legs[0].hipFrameName, "abad_L_Joint");
  EXPECT_EQ(settings->legs[0].lateralSign, 1.0);
  EXPECT_EQ(settings->legs[1].hipFrameName, "abad_R_Joint");
  EXPECT_EQ(settings->legs[1].lateralSign, -1.0);
}

TEST(ModelSettingsFromConfig,
     TheRollingContactsFollowTheFixedPointsAndPrecedeTheSurfaces) {
  mpc_config::TaskFile task = tron1Task();
  mpc_config::ModelSettingsConfig::PointContact knee;
  knee.name = "knee_L_point";
  knee.parent_joint_name = "knee_L_Joint";
  task.model_settings.contacts_3dof = {knee};
  task.model_settings.contact_names_6dof = {"sole"};
  task.model_settings.contact_parent_joint_names = {"knee_R_Joint"};
  task.model_settings.legs = {};
  const absl::StatusOr<ModelSettings> settings = tron1Settings(task);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_THAT(settings->contactNames,
              ::testing::ElementsAre("knee_L_point", "wheel_L_contact",
                                     "wheel_R_contact", "sole"));
  EXPECT_EQ(settings->contactKind(0), ModelSettings::ContactKind::kPoint);
  EXPECT_EQ(settings->contactKind(1), ModelSettings::ContactKind::kRolling);
  EXPECT_EQ(settings->contactKind(2), ModelSettings::ContactKind::kRolling);
  EXPECT_EQ(settings->contactKind(3), ModelSettings::ContactKind::kSurface);
  EXPECT_EQ(settings->pointContacts[0].wheelRadius, 0.0);
  EXPECT_EQ(settings->pointContactEntry(0), "model_settings.contacts_3dof[0]");
  EXPECT_EQ(settings->pointContactEntry(2),
            "model_settings.contacts_rolling[1]");
  EXPECT_THAT(settings->contactParentJointNamesInContactOrder(),
              ::testing::ElementsAre("knee_L_Joint", "wheel_L_Joint",
                                     "wheel_R_Joint", "knee_R_Joint"));
}

TEST(ModelSettingsFromConfig, ASetWithoutRollingContactsHasNone) {
  const absl::StatusOr<ModelSettings> settings = ModelSettings::Create(
      taskWithPointContacts(/*numPointContacts=*/2), twoJointUrdf(), "test_",
      /*verbose=*/false);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_FALSE(settings->hasRollingContacts());
  EXPECT_TRUE(settings->rollingContactJointNames().empty());
  EXPECT_EQ(settings->contactKind(0), ModelSettings::ContactKind::kPoint);
  EXPECT_EQ(settings->contactKind(2), ModelSettings::ContactKind::kSurface);
  EXPECT_FALSE(settings->isRollingContact(0));
}

/** A change to Tron1's first wheel, and what the refusal says. */
struct RollingContactCase {
  const char* absl_nonnull description;
  void (*absl_nonnull change)(mpc_config::TaskFile& task);
  const char* absl_nonnull message;
};

TEST(ModelSettingsFromConfig, ARollingContactThatDoesNotHoldIsRefused) {
  const std::vector<RollingContactCase> cases = {
      {.description = "no name",
       .change =
           [](mpc_config::TaskFile& task) {
             task.model_settings.contacts_rolling[0].name = "";
           },
       .message = "contacts_rolling[0] needs a name and a parent_joint_name"},
      {.description = "no joint",
       .change =
           [](mpc_config::TaskFile& task) {
             task.model_settings.contacts_rolling[0].parent_joint_name = "";
           },
       .message = "contacts_rolling[0] needs a name and a parent_joint_name"},
      {.description = "the name of a 6-DoF contact",
       .change =
           [](mpc_config::TaskFile& task) {
             task.model_settings.contact_names_6dof = {"wheel_L_contact"};
           },
       .message = "contacts_rolling[0].name is 'wheel_L_contact', which "
                  "another contact has"},
      {.description = "the name of a 3-DoF contact",
       .change =
           [](mpc_config::TaskFile& task) {
             mpc_config::ModelSettingsConfig::PointContact point;
             point.name = "wheel_R_contact";
             point.parent_joint_name = "knee_R_Joint";
             task.model_settings.contacts_3dof = {point};
           },
       .message = "contacts_rolling[1].name is 'wheel_R_contact', which "
                  "another contact has"},
      {.description = "zero radius",
       .change =
           [](mpc_config::TaskFile& task) {
             task.model_settings.contacts_rolling[0].wheel_radius = 0.0;
           },
       .message = "contacts_rolling[0].wheel_radius is 0 m"},
      {.description = "negative radius",
       .change =
           [](mpc_config::TaskFile& task) {
             task.model_settings.contacts_rolling[0].wheel_radius = -0.1;
           },
       .message = "contacts_rolling[0].wheel_radius is -0.1 m"},
      {.description = "NaN radius",
       .change =
           [](mpc_config::TaskFile& task) {
             task.model_settings.contacts_rolling[1].wheel_radius =
                 std::numeric_limits<double>::quiet_NaN();
           },
       .message = "contacts_rolling[1].wheel_radius is nan m"},
      {.description = "infinite radius",
       .change =
           [](mpc_config::TaskFile& task) {
             task.model_settings.contacts_rolling[1].wheel_radius =
                 std::numeric_limits<double>::infinity();
           },
       .message = "contacts_rolling[1].wheel_radius is inf m"},
      {.description = "NaN wheel center",
       .change =
           [](mpc_config::TaskFile& task) {
             task.model_settings.contacts_rolling[0]
                 .pos_wheel_rt_link_ewrt_link.z =
                 std::numeric_limits<double>::quiet_NaN();
           },
       .message = "contacts_rolling[0].pos_wheel_rt_link_ewrt_link.z"},
      {.description = "a wheel center off the wheel joint's axis",
       .change =
           [](mpc_config::TaskFile& task) {
             task.model_settings.contacts_rolling[1]
                 .pos_wheel_rt_link_ewrt_link.x = 0.01;
           },
       .message = "contacts_rolling[1].pos_wheel_rt_link_ewrt_link must lie "
                  "on the wheel joint's axis: an offset off the axle turns "
                  "with the wheel"},
      {.description = "a joint the URDF does not have",
       .change =
           [](mpc_config::TaskFile& task) {
             task.model_settings.contacts_rolling[0].parent_joint_name = "axle";
           },
       .message = "contacts_rolling[0].parent_joint_name is 'axle', which is "
                  "not a joint of the URDF"},
      {.description = "a fixed URDF joint",
       .change =
           [](mpc_config::TaskFile& task) {
             task.model_settings.contacts_rolling[0].parent_joint_name =
                 "limx_imu_joint";
           },
       .message = "contacts_rolling[0].parent_joint_name is 'limx_imu_joint', "
                  "which is not a continuous joint"},
      {.description = "a joint the MPC fixes",
       .change =
           [](mpc_config::TaskFile& task) {
             task.model_settings.fixed_joint_names = {"wheel_R_Joint"};
           },
       .message = "contacts_rolling[1].parent_joint_name is 'wheel_R_Joint', "
                  "which model_settings.fixed_joint_names fixes"},
      {.description = "two contacts on one wheel",
       .change =
           [](mpc_config::TaskFile& task) {
             task.model_settings.contacts_rolling[1].parent_joint_name =
                 "wheel_L_Joint";
           },
       .message = "contacts_rolling[1].parent_joint_name is 'wheel_L_Joint', "
                  "the joint of another rolling contact"},
  };
  for (const RollingContactCase& entry : cases) {
    SCOPED_TRACE(entry.description);
    mpc_config::TaskFile task = tron1Task();
    entry.change(task);
    const absl::StatusOr<ModelSettings> settings = tron1Settings(task);
    ASSERT_FALSE(settings.ok());
    EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(settings.status().message(), HasSubstr(entry.message));
  }
}

/**
 * Tron1's URDF with wheel_L_Joint declared as the upstream description does, a
 * revolute joint with position limits, written to the test's scratch
 * directory; an empty path when the shipped URDF does not read as expected.
 */
std::string tron1UrdfWithLimitedRevoluteWheel() {
  std::ifstream input(runfilesPath(kTron1UrdfFile));
  std::string urdf((std::istreambuf_iterator<char>(input)),
                   std::istreambuf_iterator<char>());
  constexpr absl::string_view kContinuousWheel =
      R"(<joint name="wheel_L_Joint" type="continuous">)";
  const size_t joint = urdf.find(kContinuousWheel);
  if (joint == std::string::npos) return "";
  urdf.replace(joint, kContinuousWheel.size(),
               R"(<joint name="wheel_L_Joint" type="revolute">)");
  const size_t limitBegin = urdf.find("<limit ", joint);
  const size_t limitEnd = urdf.find("/>", limitBegin);
  if (limitBegin == std::string::npos || limitEnd == std::string::npos) {
    return "";
  }
  urdf.replace(
      limitBegin, limitEnd + 2 - limitBegin,
      R"(<limit lower="-100" upper="100" effort="40" velocity="40" />)");
  const std::string path =
      absl::StrCat(::testing::TempDir(), "/tron1_limited_wheel.urdf");
  std::ofstream(path) << urdf;
  return path;
}

TEST(ModelSettingsFromConfig, ARollingContactOnALimitedRevoluteJointIsRefused) {
  const std::string urdfFile = tron1UrdfWithLimitedRevoluteWheel();
  ASSERT_FALSE(urdfFile.empty());
  const absl::StatusOr<ModelSettings> settings =
      ModelSettings::Create(tron1Task(), urdfFile, "test_", /*verbose=*/false);
  ASSERT_FALSE(settings.ok());
  EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(settings.status().message(),
              HasSubstr("contacts_rolling[0].parent_joint_name is "
                        "'wheel_L_Joint', a revolute joint with position "
                        "limits [-100, 100]: a wheel joint is continuous"));
}

/** FNV-1a 64 over words and length-prefixed strings, written out as
 * ModelSettings.cpp's hash: add, then read value(). Not thread-safe. */
class OracleHash {
 public:
  void addWord(uint64_t word) {
    for (int octet = 0; octet < nproto::kFnvWordOctets; ++octet) {
      hash_ ^= (word >> (CHAR_BIT * octet)) & nproto::kFnvOctetMask;
      hash_ *= nproto::kFnvPrime;
    }
  }
  void addString(absl::string_view text) {
    addWord(text.size());
    for (const char character : text) {
      hash_ ^= static_cast<uint64_t>(static_cast<unsigned char>(character));
      hash_ *= nproto::kFnvPrime;
    }
  }
  void addStrings(absl::Span<const std::string> texts) {
    addWord(texts.size());
    for (const std::string& text : texts) addString(text);
  }
  uint64_t value() const { return hash_; }

 private:
  uint64_t hash_ = nproto::kFnvOffsetBasis;
};

/** The fingerprint ModelSettings had before rolling contacts: the state layout
 * tag, the robot name and the joint, 3-DoF and 6-DoF contact names. */
uint64_t fingerprintWithoutContactKinds(const ModelSettings& settings) {
  OracleHash hash;
  hash.addString(kStateLayoutTag);
  hash.addString(settings.robotName);
  hash.addStrings(settings.mpcModelJointNames);
  hash.addStrings(settings.contactNames3DoF);
  hash.addStrings(settings.contactNames6DoF);
  return hash.value();
}

TEST(ModelSettingsFromConfig,
     OnlyARollingSetMixesItsContactKindsIntoTheFingerprint) {
  const absl::StatusOr<ModelSettings> points = ModelSettings::Create(
      taskWithPointContacts(/*numPointContacts=*/2), twoJointUrdf(), "test_",
      /*verbose=*/false);
  ASSERT_TRUE(points.ok()) << points.status();
  EXPECT_EQ(modelLayoutFingerprint(*points),
            fingerprintWithoutContactKinds(*points));

  const absl::StatusOr<ModelSettings> wheels = tron1Settings(tron1Task());
  ASSERT_TRUE(wheels.ok()) << wheels.status();
  EXPECT_NE(modelLayoutFingerprint(*wheels),
            fingerprintWithoutContactKinds(*wheels));
  mpc_config::TaskFile otherRadius = tron1Task();
  otherRadius.model_settings.contacts_rolling[1].wheel_radius = 0.1;
  const absl::StatusOr<ModelSettings> otherWheels = tron1Settings(otherRadius);
  ASSERT_TRUE(otherWheels.ok()) << otherWheels.status();
  EXPECT_NE(modelLayoutFingerprint(*wheels),
            modelLayoutFingerprint(*otherWheels));
}

}  // namespace
}  // namespace ocs2::legged
