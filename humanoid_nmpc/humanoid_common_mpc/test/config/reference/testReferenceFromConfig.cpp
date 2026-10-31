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

// The reference file's conversion: every field reaches its member, the fields
// the MPC requires are required, the others default to what an absent key stood
// for, every value is finite, and the default posture is read by joint name in
// the model's order with every joint exactly once.

#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_mpc_config/joint_value.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"

namespace ocs2::humanoid {
namespace {

using RequiredField =
    std::optional<double> mpc_config::ReferenceFile::*absl_nonnull;
using DefaultedField = double mpc_config::ReferenceFile::*absl_nonnull;

// The fields the MPC cannot do without.
constexpr std::pair<absl::string_view, RequiredField> kRequiredFields[] = {
    {"ref_speed_base_rt_world",
     &mpc_config::ReferenceFile::ref_speed_base_rt_world},
    {"ref_rate_yaw_base_rt_world",
     &mpc_config::ReferenceFile::ref_rate_yaw_base_rt_world},
    {"cmd_max_linvel_base_rt_world_ewrt_heading_x",
     &mpc_config::ReferenceFile::cmd_max_linvel_base_rt_world_ewrt_heading_x},
    {"cmd_max_linvel_base_rt_world_ewrt_heading_y",
     &mpc_config::ReferenceFile::cmd_max_linvel_base_rt_world_ewrt_heading_y},
    {"cmd_max_err_pos_base_rt_world_ewrt_world_z",
     &mpc_config::ReferenceFile::cmd_max_err_pos_base_rt_world_ewrt_world_z},
    {"cmd_max_angvel_base_rt_world_ewrt_world_z",
     &mpc_config::ReferenceFile::cmd_max_angvel_base_rt_world_ewrt_world_z},
    {"nom_pos_base_rt_world_ewrt_world_z",
     &mpc_config::ReferenceFile::nom_pos_base_rt_world_ewrt_world_z},
};

// The fields an absent value of which stands for "off".
constexpr std::pair<absl::string_view, DefaultedField> kDefaultedFields[] = {
    {"cmd_max_linacc_base_rt_world_ewrt_heading_xy",
     &mpc_config::ReferenceFile::cmd_max_linacc_base_rt_world_ewrt_heading_xy},
    {"cmd_max_angacc_base_rt_world_ewrt_world_z",
     &mpc_config::ReferenceFile::cmd_max_angacc_base_rt_world_ewrt_world_z},
    {"velocity_command_filter_break_frequency",
     &mpc_config::ReferenceFile::velocity_command_filter_break_frequency},
};

/** A file with every scalar field set, each to a value of its own. */
mpc_config::ReferenceFile completeFile() {
  mpc_config::ReferenceFile file;
  file.ref_speed_base_rt_world = 1.25;
  file.ref_rate_yaw_base_rt_world = 2.25;
  file.cmd_max_linvel_base_rt_world_ewrt_heading_x = 3.25;
  file.cmd_max_linvel_base_rt_world_ewrt_heading_y = 4.25;
  file.cmd_max_err_pos_base_rt_world_ewrt_world_z = 5.25;
  file.cmd_max_angvel_base_rt_world_ewrt_world_z = 6.25;
  file.cmd_max_linacc_base_rt_world_ewrt_heading_xy = 7.25;
  file.cmd_max_angacc_base_rt_world_ewrt_world_z = 8.25;
  file.velocity_command_filter_break_frequency = 9.25;
  file.target_joint_state_interpolation_time_constant = 10.25;
  file.nom_pos_base_rt_world_ewrt_world_z = 11.25;
  return file;
}

/** A file with only the fields the MPC requires. */
mpc_config::ReferenceFile requiredOnly() {
  mpc_config::ReferenceFile file;
  for (const std::pair<absl::string_view, RequiredField>& field :
       kRequiredFields) {
    file.*field.second = 1.0;
  }
  return file;
}

TEST(ReferenceSettingsFromConfigTest, EveryFieldReachesItsMember) {
  const absl::StatusOr<ReferenceSettings> settings =
      referenceSettingsFromConfig(completeFile());
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_EQ(settings->refSpeedBaseRtWorld, 1.25);
  EXPECT_EQ(settings->refRateYawBaseRtWorld, 2.25);
  EXPECT_EQ(settings->cmdMaxLinvelBaseRtWorldEwrtHeadingX, 3.25);
  EXPECT_EQ(settings->cmdMaxLinvelBaseRtWorldEwrtHeadingY, 4.25);
  EXPECT_EQ(settings->cmdMaxErrPosBaseRtWorldEwrtWorldZ, 5.25);
  EXPECT_EQ(settings->cmdMaxAngvelBaseRtWorldEwrtWorldZ, 6.25);
  EXPECT_EQ(settings->cmdMaxLinaccBaseRtWorldEwrtHeadingXy, 7.25);
  EXPECT_EQ(settings->cmdMaxAngaccBaseRtWorldEwrtWorldZ, 8.25);
  EXPECT_EQ(settings->velocityCommandFilterBreakFrequency, 9.25);
  EXPECT_EQ(settings->targetJointStateInterpolationTimeConstant,
            std::optional<scalar_t>(10.25));
  EXPECT_EQ(settings->nomPosBaseRtWorldEwrtWorldZ, 11.25);
}

TEST(ReferenceSettingsFromConfigTest,
     EachFieldTheMpcRequiresIsRequiredAndNamed) {
  for (const std::pair<absl::string_view, RequiredField>& field :
       kRequiredFields) {
    mpc_config::ReferenceFile file = completeFile();
    file.*field.second = std::nullopt;
    const absl::StatusOr<ReferenceSettings> settings =
        referenceSettingsFromConfig(file);
    EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument)
        << field.first;
    EXPECT_TRUE(absl::StrContains(settings.status().message(), field.first))
        << settings.status();
  }
}

TEST(ReferenceSettingsFromConfigTest,
     AnAbsentOptionalFieldIsWhatTheSettingsDefaultTo) {
  const absl::StatusOr<ReferenceSettings> settings =
      referenceSettingsFromConfig(requiredOnly());
  ASSERT_TRUE(settings.ok()) << settings.status();
  const ReferenceSettings defaults;
  EXPECT_EQ(settings->cmdMaxLinaccBaseRtWorldEwrtHeadingXy,
            defaults.cmdMaxLinaccBaseRtWorldEwrtHeadingXy);
  EXPECT_EQ(settings->cmdMaxAngaccBaseRtWorldEwrtWorldZ,
            defaults.cmdMaxAngaccBaseRtWorldEwrtWorldZ);
  EXPECT_EQ(settings->velocityCommandFilterBreakFrequency,
            defaults.velocityCommandFilterBreakFrequency);
  EXPECT_FALSE(settings->targetJointStateInterpolationTimeConstant.has_value())
      << "the whole-body MPC's file has none";
  // The schema's defaults are the settings' (the file's struct defaults its
  // optional fields as the schema does).
  const mpc_config::ReferenceFile empty;
  for (const std::pair<absl::string_view, DefaultedField>& field :
       kDefaultedFields) {
    EXPECT_EQ(empty.*field.second, 0.0) << field.first;
  }
  EXPECT_EQ(defaults.cmdMaxLinaccBaseRtWorldEwrtHeadingXy, 0.0);
  EXPECT_EQ(defaults.cmdMaxAngaccBaseRtWorldEwrtWorldZ, 0.0);
  EXPECT_EQ(defaults.velocityCommandFilterBreakFrequency, 0.0);
}

TEST(ReferenceSettingsFromConfigTest,
     AValueThatIsNotAFiniteNumberIsRefusedNamingItsField) {
  for (const double bad : {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
    for (const std::pair<absl::string_view, RequiredField>& field :
         kRequiredFields) {
      mpc_config::ReferenceFile file = completeFile();
      file.*field.second = bad;
      const absl::StatusOr<ReferenceSettings> settings =
          referenceSettingsFromConfig(file);
      EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument)
          << field.first;
      EXPECT_TRUE(absl::StrContains(settings.status().message(), field.first))
          << settings.status();
    }
    for (const std::pair<absl::string_view, DefaultedField>& field :
         kDefaultedFields) {
      mpc_config::ReferenceFile file = completeFile();
      file.*field.second = bad;
      const absl::StatusOr<ReferenceSettings> settings =
          referenceSettingsFromConfig(file);
      EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument)
          << field.first;
      EXPECT_TRUE(absl::StrContains(settings.status().message(), field.first))
          << settings.status();
    }
    mpc_config::ReferenceFile file = completeFile();
    file.target_joint_state_interpolation_time_constant = bad;
    EXPECT_TRUE(
        absl::StrContains(referenceSettingsFromConfig(file).status().message(),
                          "target_joint_state_interpolation_time_constant"));
  }
}

TEST(ReferenceSettingsFromConfigTest,
     AValueOutOfItsRangeIsRefusedNamingItsField) {
  // A speed or a command limit something is divided by, and a height, are
  // positive; the symmetric clamp bound of the pelvis height and the
  // acceleration limits (0: no ramp) are at least 0. std::clamp(x, -b, b) with
  // b < 0 is undefined, and a negative velocity limit would flip the operator's
  // command. (An acceleration limit at or below 0 turns its ramp off:
  // TheCommandFilterIsOffAtZeroAndNeverNegative.)
  const std::vector<std::pair<absl::string_view, RequiredField>> positive = {
      {"ref_speed_base_rt_world",
       &mpc_config::ReferenceFile::ref_speed_base_rt_world},
      {"ref_rate_yaw_base_rt_world",
       &mpc_config::ReferenceFile::ref_rate_yaw_base_rt_world},
      {"cmd_max_linvel_base_rt_world_ewrt_heading_x",
       &mpc_config::ReferenceFile::cmd_max_linvel_base_rt_world_ewrt_heading_x},
      {"cmd_max_linvel_base_rt_world_ewrt_heading_y",
       &mpc_config::ReferenceFile::cmd_max_linvel_base_rt_world_ewrt_heading_y},
      {"cmd_max_angvel_base_rt_world_ewrt_world_z",
       &mpc_config::ReferenceFile::cmd_max_angvel_base_rt_world_ewrt_world_z},
      {"nom_pos_base_rt_world_ewrt_world_z",
       &mpc_config::ReferenceFile::nom_pos_base_rt_world_ewrt_world_z},
      {"target_joint_state_interpolation_time_constant",
       &mpc_config::ReferenceFile::
           target_joint_state_interpolation_time_constant},
  };
  for (const std::pair<absl::string_view, RequiredField>& field : positive) {
    for (const double bad : {0.0, -1.0}) {
      mpc_config::ReferenceFile file = completeFile();
      file.*field.second = bad;
      const absl::Status status = referenceSettingsFromConfig(file).status();
      EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument)
          << field.first << " = " << bad;
      EXPECT_TRUE(absl::StrContains(
          status.message(), absl::StrCat(field.first, ": ", bad,
                                         " is not a finite positive number")))
          << status;
    }
  }
  mpc_config::ReferenceFile height = completeFile();
  height.cmd_max_err_pos_base_rt_world_ewrt_world_z = -0.1;
  EXPECT_TRUE(
      absl::StrContains(referenceSettingsFromConfig(height).status().message(),
                        "cmd_max_err_pos_base_rt_world_ewrt_world_z: -0.1"));
  height.cmd_max_err_pos_base_rt_world_ewrt_world_z = 0.0;
  EXPECT_TRUE(referenceSettingsFromConfig(height).ok())
      << "a pelvis height that may not change";
}

TEST(ReferenceSettingsFromConfigTest,
     TheCommandFilterIsOffAtZeroAndNeverNegative) {
  mpc_config::ReferenceFile file = completeFile();
  file.velocity_command_filter_break_frequency = 0.0;
  EXPECT_TRUE(referenceSettingsFromConfig(file).ok());
  file.velocity_command_filter_break_frequency = -1.0;
  const absl::StatusOr<ReferenceSettings> negative =
      referenceSettingsFromConfig(file);
  EXPECT_EQ(negative.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(negative.status().message(),
                                "velocity_command_filter_break_frequency"))
      << negative.status();
  // A ramp at or below zero is off, as it always was: no sign rule.
  file = completeFile();
  file.cmd_max_linacc_base_rt_world_ewrt_heading_xy = -1.0;
  EXPECT_TRUE(referenceSettingsFromConfig(file).ok());
}

/** The joints of a model of the test's, in state order. */
std::vector<std::string> modelJoints() {
  return {"hip", "knee", "ankle"};
}

/** The joints the model's task file fixes. */
std::vector<std::string> fixedJoints() {
  return {"wrist"};
}

mpc_config::JointValue joint(absl::string_view name, double value) {
  return mpc_config::JointValue{.joint = std::string(name), .value = value};
}

TEST(DefaultJointStateFromConfigTest,
     ThePostureIsInTheModelsOrderWhateverTheFilesOrder) {
  mpc_config::ReferenceFile file;
  file.default_joint_state = {joint("ankle", /*value=*/-0.4),
                              joint("hip", /*value=*/-0.3),
                              joint("knee", /*value=*/0.7)};
  const absl::StatusOr<vector_t> state =
      defaultJointStateFromConfig(file, modelJoints(), fixedJoints());
  ASSERT_TRUE(state.ok()) << state.status();
  ASSERT_EQ(state->size(), 3);
  EXPECT_EQ((*state)(0), -0.3);
  EXPECT_EQ((*state)(1), 0.7);
  EXPECT_EQ((*state)(2), -0.4);
}

TEST(DefaultJointStateFromConfigTest,
     EveryJointThatIsMissingTwiceFixedOrUnknownIsListed) {
  mpc_config::ReferenceFile file;
  file.default_joint_state = {
      joint("hip", /*value=*/0.0), joint("hip", /*value=*/0.1),
      joint("wrist", /*value=*/0.0), joint("elbow", /*value=*/0.0)};
  const absl::StatusOr<vector_t> state =
      defaultJointStateFromConfig(file, modelJoints(), fixedJoints());
  ASSERT_EQ(state.status().code(), absl::StatusCode::kInvalidArgument);
  const absl::string_view message = state.status().message();
  EXPECT_TRUE(absl::StrContains(message, "misses the MPC joints knee, ankle"))
      << message;
  EXPECT_TRUE(absl::StrContains(message, "more than once hip")) << message;
  EXPECT_TRUE(absl::StrContains(message, "wrist, which the task file fixes"))
      << message;
  EXPECT_TRUE(
      absl::StrContains(message, "elbow, which the MPC model does not have"))
      << message;
  EXPECT_TRUE(absl::StartsWith(message, "default_joint_state ")) << message;
}

TEST(DefaultJointStateFromConfigTest, AnEmptyPostureMissesEveryJoint) {
  const absl::StatusOr<vector_t> state = defaultJointStateFromConfig(
      mpc_config::ReferenceFile{}, modelJoints(), fixedJoints());
  EXPECT_TRUE(absl::StrContains(state.status().message(),
                                "misses the MPC joints hip, knee, ankle"))
      << state.status();
}

TEST(DefaultJointStateFromConfigTest,
     APositionThatIsNotAFiniteNumberIsRefusedNamingTheJoint) {
  mpc_config::ReferenceFile file;
  file.default_joint_state = {
      joint("hip", /*value=*/0.0),
      joint("knee", /*value=*/std::numeric_limits<double>::quiet_NaN()),
      joint("ankle", /*value=*/0.0)};
  const absl::StatusOr<vector_t> state =
      defaultJointStateFromConfig(file, modelJoints(), fixedJoints());
  EXPECT_EQ(state.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(state.status().message(), "knee"))
      << state.status();
}

}  // namespace
}  // namespace ocs2::humanoid
