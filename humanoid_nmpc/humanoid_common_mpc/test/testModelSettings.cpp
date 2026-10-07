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

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"

/**
 * ModelSettings::Create() and GaitSchedule::Create(), the path forms of the roots the MPC interfaces build their model
 * from, on a two-joint robot and textprotos written by the test: what they accept, and the failures they report instead
 * of throwing. test/config/testConfigRoots.cpp tests them on the shipped files.
 */
namespace ocs2::humanoid {
namespace {

using ::testing::HasSubstr;

/** Writes `content` to `name` in the test's scratch directory and returns its path. */
std::string writeFile(const std::string& name, const std::string& content) {
  const std::string path = absl::StrCat(::testing::TempDir(), "/", name);
  std::ofstream(path) << content;
  return path;
}

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

/** The robot of kTwoJointUrdf, written to the scratch directory. */
std::string twoJointUrdf() {
  return writeFile("two_joints.urdf", absl::Substitute(kTwoJointUrdf, kInertial));
}

/** A task file whose model_settings block carries `extra` after its robot name. */
std::string taskFile(const std::string& name, const std::string& extra) {
  return writeFile(name,
                   absl::StrCat("model_settings {\n  robot_name: \"two_joints\"\n  phase_transition_stance_time: 0.25\n", extra, "}\n"));
}

TEST(ModelSettingsCreate, LoadsTheActiveJointsOfTheUrdf) {
  const absl::StatusOr<ModelSettings> settings =
      ModelSettings::Create(taskFile("plain.textproto", /*extra=*/""), twoJointUrdf(), "test_", /*verbose=*/false);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_EQ(settings->robotName, "two_joints");
  EXPECT_EQ(settings->mpc_joint_dim, 2U);
  EXPECT_EQ(settings->full_joint_dim, 2U);
  EXPECT_THAT(settings->mpcModelJointNames, ::testing::UnorderedElementsAre("shoulder", "hip"));
  EXPECT_FALSE(settings->hasArmSwingJoints) << "a file without arm_joint_names disables the arm swing";
  EXPECT_EQ(settings->modelFolderCppAd, "cppad_code_gen/cppad_test_two_joints");
}

TEST(ModelSettingsCreate, FixingEveryJointIsRefused) {
  const absl::StatusOr<ModelSettings> settings =
      ModelSettings::Create(taskFile("all_fixed.textproto", "  fixed_joint_names: \"shoulder\"\n  fixed_joint_names: \"hip\"\n"),
                            twoJointUrdf(), "test_", /*verbose=*/false);
  EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(settings.status().message(), HasSubstr("at least one active joint"));
}

TEST(ModelSettingsCreate, AnArmJointThatIsNotAnActiveJointIsRefusedNamingIt) {
  // A misspelled joint, and one that fixed_joint_names fixes: both used to fail a CHECK macro that threw.
  const std::string arms =
      "  arm_joint_names {\n    left_shoulder_y: \"shoulder\"\n    right_shoulder_y: \"shoulder\"\n    left_elbow_y: \"shoulder\"\n"
      "    right_elbow_y: \"elbow\"\n  }\n";
  const absl::StatusOr<ModelSettings> misspelled =
      ModelSettings::Create(taskFile("misspelled_arm.textproto", arms), twoJointUrdf(), "test_", /*verbose=*/false);
  EXPECT_EQ(misspelled.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(misspelled.status().message(), HasSubstr("right_elbow_y is 'elbow'"));

  const std::string fixedArm =
      "  fixed_joint_names: \"shoulder\"\n  arm_joint_names {\n    left_shoulder_y: \"shoulder\"\n    right_shoulder_y: \"hip\"\n"
      "    left_elbow_y: \"hip\"\n    right_elbow_y: \"hip\"\n  }\n";
  const absl::StatusOr<ModelSettings> fixed =
      ModelSettings::Create(taskFile("fixed_arm.textproto", fixedArm), twoJointUrdf(), "test_", /*verbose=*/false);
  EXPECT_EQ(fixed.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(fixed.status().message(), HasSubstr("left_shoulder_y"));

  const std::string valid =
      "  arm_joint_names {\n    left_shoulder_y: \"shoulder\"\n    right_shoulder_y: \"hip\"\n    left_elbow_y: \"shoulder\"\n"
      "    right_elbow_y: \"hip\"\n  }\n";
  const absl::StatusOr<ModelSettings> resolved =
      ModelSettings::Create(taskFile("valid_arm.textproto", valid), twoJointUrdf(), "test_", /*verbose=*/false);
  ASSERT_TRUE(resolved.ok()) << resolved.status();
  EXPECT_TRUE(resolved->hasArmSwingJoints);
  EXPECT_EQ(resolved->j_l_shoulder_y_index, resolved->jointIndexMap.at("shoulder"));
  EXPECT_EQ(resolved->j_r_shoulder_y_index, resolved->jointIndexMap.at("hip"));
}

TEST(ModelSettingsCreate, AFileThatCannotBeReadOrAValueThatDoesNotConvertIsRefused) {
  const std::string urdf = twoJointUrdf();
  const absl::StatusOr<ModelSettings> missing =
      ModelSettings::Create(::testing::TempDir() + "/no_such_task.textproto", urdf, "test_", /*verbose=*/false);
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(missing.status().message(), HasSubstr("no_such_task.textproto"));

  // A value that does not convert is refused by the parser, naming the file, its line and its column.
  const std::string badValueFile =
      writeFile("bad_value.textproto", "model_settings {\n  robot_name: \"two_joints\"\n  phase_transition_stance_time: soon\n}\n");
  const absl::StatusOr<ModelSettings> badValue = ModelSettings::Create(badValueFile, urdf, "test_", /*verbose=*/false);
  EXPECT_EQ(badValue.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(badValue.status().message(), HasSubstr(absl::StrCat(badValueFile, ":3:")));

  const std::string badFootGainFile = taskFile("bad_foot_gain.textproto", "  foot_constraint {\n    position_error_gain_z: high\n  }\n");
  const absl::StatusOr<ModelSettings> badFootGain = ModelSettings::Create(badFootGainFile, urdf, "test_", /*verbose=*/false);
  EXPECT_EQ(badFootGain.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(badFootGain.status().message(), HasSubstr(absl::StrCat(badFootGainFile, ":5:")));
}

TEST(ModelSettingsCreate, TheSettingsMoveWithEveryField) {
  // The MPC interfaces take the settings Create() returns over by moving them.
  absl::StatusOr<ModelSettings> created =
      ModelSettings::Create(taskFile("moved.textproto", /*extra=*/""), twoJointUrdf(), "test_", /*verbose=*/false);
  ASSERT_TRUE(created.ok()) << created.status();
  const ModelSettings moved = *std::move(created);
  EXPECT_EQ(moved.mpc_joint_dim, 2U);
  EXPECT_EQ(moved.jointIndexMap.size(), 2U);
  EXPECT_EQ(moved.modelFolderCppAd, "cppad_code_gen/cppad_test_two_joints");
}

TEST(GaitScheduleCreate, LoadsTheReferenceFileOrReportsWhyNot) {
  const absl::StatusOr<ModelSettings> settings =
      ModelSettings::Create(taskFile("gait_settings.textproto", /*extra=*/""), twoJointUrdf(), "test_", /*verbose=*/false);
  ASSERT_TRUE(settings.ok()) << settings.status();

  const std::string reference =
      writeFile("reference.textproto",
                "initial_mode_schedule {\n  mode_sequence: [\"STANCE\", \"STANCE\"]\n  event_times: 0.5\n}\n"
                "default_mode_sequence_template {\n  mode_sequence: \"STANCE\"\n  switching_times: [0.0, 1.0]\n}\n");
  const absl::StatusOr<std::shared_ptr<GaitSchedule>> schedule = GaitSchedule::Create(reference, *settings);
  ASSERT_TRUE(schedule.ok()) << schedule.status();
  EXPECT_EQ((*schedule)->getCurrentModeSchedule().modeSequence, (std::vector<size_t>{ModeNumber::kStance, ModeNumber::kStance}));

  const std::string noTemplate = writeFile("reference_without_template.textproto",
                                           "initial_mode_schedule {\n  mode_sequence: [\"STANCE\", \"STANCE\"]\n  event_times: 0.5\n}\n");
  const absl::StatusOr<std::shared_ptr<GaitSchedule>> refused = GaitSchedule::Create(noTemplate, *settings);
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(refused.status().message(), HasSubstr("default_mode_sequence_template"));

  EXPECT_EQ(GaitSchedule::Create(::testing::TempDir() + "/no_such_reference.textproto", *settings).status().code(),
            absl::StatusCode::kNotFound);
}

TEST(ProceduralMpcMotionManagerGaits, AGaitFileWithoutAGaitTheCommandSelectsIsRefused) {
  // The motion manager switches gaits by name as the commanded velocity changes; a gait file without one of them used
  // to load and then fail the solve the first time the command asked for it.
  const ModeSequenceTemplate stance({0.0, 1.0}, {ModeNumber::kStance});
  ProceduralMpcMotionManager::GaitModeStateConfig stanceState{};
  stanceState.gaitCommand = "stance";
  ProceduralMpcMotionManager::GaitModeStateConfig walkState{};
  walkState.gaitCommand = "walk";
  const std::vector<ProceduralMpcMotionManager::GaitModeStateConfig> states = {stanceState, walkState};
  const std::map<std::string, ModeSequenceTemplate> stanceOnly = {{"stance", stance}};
  const absl::Status refused = ProceduralMpcMotionManager::checkEveryGaitIsLoaded(stanceOnly, states, "gait.textproto");
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(refused.message(), HasSubstr("gait.textproto has no gait 'walk'"));

  const std::map<std::string, ModeSequenceTemplate> both = {{"stance", stance}, {"walk", stance}};
  EXPECT_TRUE(ProceduralMpcMotionManager::checkEveryGaitIsLoaded(both, states, "gait.textproto").ok());
}

}  // namespace
}  // namespace ocs2::humanoid
