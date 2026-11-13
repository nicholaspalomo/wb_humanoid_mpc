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

// The nproto structs of every legged_mpc_msgs message (ocs2::legged::msgs,
// tools/nproto/README.md): each one round-trips through its protobuf message
// both ways, and the test covers every .proto file of the package.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "Eigen/Core"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/status/status.h"
#include "gtest/gtest.h"

#include "legged_mpc_msgs/arrows.nproto.h"
#include "legged_mpc_msgs/arrows.pb.h"
#include "legged_mpc_msgs/color.nproto.h"
#include "legged_mpc_msgs/color.pb.h"
#include "legged_mpc_msgs/config_file_kind.nproto.h"
#include "legged_mpc_msgs/config_file_kind.pb.h"
#include "legged_mpc_msgs/config_file_save.nproto.h"
#include "legged_mpc_msgs/config_file_save.pb.h"
#include "legged_mpc_msgs/config_file_save_status.nproto.h"
#include "legged_mpc_msgs/config_file_save_status.pb.h"
#include "legged_mpc_msgs/controller_type.nproto.h"
#include "legged_mpc_msgs/controller_type.pb.h"
#include "legged_mpc_msgs/dodgeball_throw.nproto.h"
#include "legged_mpc_msgs/dodgeball_throw.pb.h"
#include "legged_mpc_msgs/fsm_command.nproto.h"
#include "legged_mpc_msgs/fsm_command.pb.h"
#include "legged_mpc_msgs/fsm_state.nproto.h"
#include "legged_mpc_msgs/fsm_state.pb.h"
#include "legged_mpc_msgs/joint_targets.nproto.h"
#include "legged_mpc_msgs/joint_targets.pb.h"
#include "legged_mpc_msgs/line_strip.nproto.h"
#include "legged_mpc_msgs/line_strip.pb.h"
#include "legged_mpc_msgs/line_strips.nproto.h"
#include "legged_mpc_msgs/line_strips.pb.h"
#include "legged_mpc_msgs/loop_timing.nproto.h"
#include "legged_mpc_msgs/loop_timing.pb.h"
#include "legged_mpc_msgs/mode_schedule.nproto.h"
#include "legged_mpc_msgs/mode_schedule.pb.h"
#include "legged_mpc_msgs/mpc_observation.nproto.h"
#include "legged_mpc_msgs/mpc_observation.pb.h"
#include "legged_mpc_msgs/mpc_policy.nproto.h"
#include "legged_mpc_msgs/mpc_policy.pb.h"
#include "legged_mpc_msgs/mpc_solver_status.nproto.h"
#include "legged_mpc_msgs/mpc_solver_status.pb.h"
#include "legged_mpc_msgs/mpc_status.nproto.h"
#include "legged_mpc_msgs/mpc_status.pb.h"
#include "legged_mpc_msgs/performance_index.nproto.h"
#include "legged_mpc_msgs/performance_index.pb.h"
#include "legged_mpc_msgs/pose.nproto.h"
#include "legged_mpc_msgs/pose.pb.h"
#include "legged_mpc_msgs/quaternion.nproto.h"
#include "legged_mpc_msgs/quaternion.pb.h"
#include "legged_mpc_msgs/reset_requests.nproto.h"
#include "legged_mpc_msgs/reset_requests.pb.h"
#include "legged_mpc_msgs/robot_model_instance.nproto.h"
#include "legged_mpc_msgs/robot_model_instance.pb.h"
#include "legged_mpc_msgs/robot_state_sample.nproto.h"
#include "legged_mpc_msgs/robot_state_sample.pb.h"
#include "legged_mpc_msgs/scalar_group.nproto.h"
#include "legged_mpc_msgs/scalar_group.pb.h"
#include "legged_mpc_msgs/spheres.nproto.h"
#include "legged_mpc_msgs/spheres.pb.h"
#include "legged_mpc_msgs/system_observation.nproto.h"
#include "legged_mpc_msgs/system_observation.pb.h"
#include "legged_mpc_msgs/target_contact_patch.nproto.h"
#include "legged_mpc_msgs/target_contact_patch.pb.h"
#include "legged_mpc_msgs/target_trajectories.nproto.h"
#include "legged_mpc_msgs/target_trajectories.pb.h"
#include "legged_mpc_msgs/telemetry_series.nproto.h"
#include "legged_mpc_msgs/telemetry_series.pb.h"
#include "legged_mpc_msgs/vector.nproto.h"
#include "legged_mpc_msgs/vector.pb.h"
#include "legged_mpc_msgs/vector3.nproto.h"
#include "legged_mpc_msgs/vector3.pb.h"
#include "legged_mpc_msgs/viewer_annotations.nproto.h"
#include "legged_mpc_msgs/viewer_annotations.pb.h"
#include "legged_mpc_msgs/visualization_scene.nproto.h"
#include "legged_mpc_msgs/visualization_scene.pb.h"
#include "legged_mpc_msgs/walking_velocity_command.nproto.h"
#include "legged_mpc_msgs/walking_velocity_command.pb.h"
#include "legged_mpc_msgs/wrench.nproto.h"
#include "legged_mpc_msgs/wrench.pb.h"
#include "tools/nproto/test/ProtoTestValues.h"

// The conversions, which ExpectRoundTrips() finds by argument-dependent lookup.
// NOLINTBEGIN(misc-include-cleaner): used from the templates of ProtoTestValues.h, which include-cleaner does not follow
#include "legged_mpc_msgs/arrows.nproto.pb.h"
#include "legged_mpc_msgs/color.nproto.pb.h"
#include "legged_mpc_msgs/config_file_kind.nproto.pb.h"
#include "legged_mpc_msgs/config_file_save.nproto.pb.h"
#include "legged_mpc_msgs/config_file_save_status.nproto.pb.h"
#include "legged_mpc_msgs/controller_type.nproto.pb.h"
#include "legged_mpc_msgs/dodgeball_throw.nproto.pb.h"
#include "legged_mpc_msgs/fsm_command.nproto.pb.h"
#include "legged_mpc_msgs/fsm_state.nproto.pb.h"
#include "legged_mpc_msgs/joint_targets.nproto.pb.h"
#include "legged_mpc_msgs/line_strip.nproto.pb.h"
#include "legged_mpc_msgs/line_strips.nproto.pb.h"
#include "legged_mpc_msgs/loop_timing.nproto.pb.h"
#include "legged_mpc_msgs/mode_schedule.nproto.pb.h"
#include "legged_mpc_msgs/mpc_observation.nproto.pb.h"
#include "legged_mpc_msgs/mpc_policy.nproto.pb.h"
#include "legged_mpc_msgs/mpc_solver_status.nproto.pb.h"
#include "legged_mpc_msgs/mpc_status.nproto.pb.h"
#include "legged_mpc_msgs/performance_index.nproto.pb.h"
#include "legged_mpc_msgs/pose.nproto.pb.h"
#include "legged_mpc_msgs/quaternion.nproto.pb.h"
#include "legged_mpc_msgs/reset_requests.nproto.pb.h"
#include "legged_mpc_msgs/robot_model_instance.nproto.pb.h"
#include "legged_mpc_msgs/robot_state_sample.nproto.pb.h"
#include "legged_mpc_msgs/scalar_group.nproto.pb.h"
#include "legged_mpc_msgs/spheres.nproto.pb.h"
#include "legged_mpc_msgs/system_observation.nproto.pb.h"
#include "legged_mpc_msgs/target_contact_patch.nproto.pb.h"
#include "legged_mpc_msgs/target_trajectories.nproto.pb.h"
#include "legged_mpc_msgs/telemetry_series.nproto.pb.h"
#include "legged_mpc_msgs/vector.nproto.pb.h"
#include "legged_mpc_msgs/vector3.nproto.pb.h"
#include "legged_mpc_msgs/viewer_annotations.nproto.pb.h"
#include "legged_mpc_msgs/visualization_scene.nproto.pb.h"
#include "legged_mpc_msgs/walking_velocity_command.nproto.pb.h"
#include "legged_mpc_msgs/wrench.nproto.pb.h"
// NOLINTEND(misc-include-cleaner)

namespace ocs2::legged::msgs {
namespace {

using nproto::test_support::ExpectRoundTrips;
using nproto::test_support::ProtoEquals;

// The mappings the realtime and IPC code relies on.
static_assert(
    std::is_same_v<decltype(MpcPolicy::time_trajectory), Eigen::VectorXd>);
static_assert(
    std::is_same_v<decltype(MpcPolicy::state_trajectory), std::vector<Vector>>);
static_assert(
    std::is_same_v<decltype(MpcPolicy::controller_type), ControllerType>);
static_assert(std::is_same_v<decltype(Vector::data), Eigen::VectorXd>);
static_assert(
    std::is_same_v<decltype(SystemObservation::state), Eigen::VectorXd>);
static_assert(std::is_same_v<decltype(SystemObservation::mode), uint64_t>);
static_assert(std::is_same_v<decltype(ModeSchedule::mode_sequence),
                             std::vector<uint64_t>>);
static_assert(std::is_same_v<decltype(RobotStateSample::joint_names),
                             std::vector<std::string>>);
static_assert(std::is_same_v<decltype(RobotStateSample::contact_flags),
                             std::vector<bool>>);
static_assert(
    std::is_same_v<decltype(RobotStateSample::meas_wrench_contacts_ewrt_world),
                   std::vector<Wrench>>);
static_assert(std::is_same_v<decltype(JointTargets::joint_positions),
                             absl::btree_map<std::string, double>>);
static_assert(std::is_same_v<decltype(Spheres::radii), Eigen::VectorXf>);
static_assert(std::is_same_v<decltype(TargetContactPatch::kind),
                             TargetContactPatch::Kind>);
// Presence, not a `bool valid`: a foot without a target has no pose.
static_assert(
    std::is_same_v<decltype(TargetContactPatch::des_pose_foot_rt_world),
                   std::optional<TargetContactPatch::Pose>>);
static_assert(std::is_enum_v<ControllerType> &&
              !std::is_convertible_v<ControllerType, int>);

template <typename StructType, typename ProtoType>
struct Conversion {
  using Struct = StructType;
  using Proto = ProtoType;
};

// Every message of the package; controller_type.proto and
// config_file_kind.proto are its enums (EnumTest below).
using AllMessages = ::testing::Types<
    Conversion<Arrows, legged_mpc_msgs::Arrows>,
    Conversion<Color, legged_mpc_msgs::Color>,
    Conversion<ConfigFileSave, legged_mpc_msgs::ConfigFileSave>,
    Conversion<ConfigFileSaveStatus, legged_mpc_msgs::ConfigFileSaveStatus>,
    Conversion<DodgeballThrow, legged_mpc_msgs::DodgeballThrow>,
    Conversion<FsmCommand, legged_mpc_msgs::FsmCommand>,
    Conversion<FsmState, legged_mpc_msgs::FsmState>,
    Conversion<JointTargets, legged_mpc_msgs::JointTargets>,
    Conversion<LineStrip, legged_mpc_msgs::LineStrip>,
    Conversion<LineStrips, legged_mpc_msgs::LineStrips>,
    Conversion<LoopTiming, legged_mpc_msgs::LoopTiming>,
    Conversion<ModeSchedule, legged_mpc_msgs::ModeSchedule>,
    Conversion<MpcObservation, legged_mpc_msgs::MpcObservation>,
    Conversion<MpcPolicy, legged_mpc_msgs::MpcPolicy>,
    Conversion<MpcSolverStatus, legged_mpc_msgs::MpcSolverStatus>,
    Conversion<MpcStatus, legged_mpc_msgs::MpcStatus>,
    Conversion<PerformanceIndex, legged_mpc_msgs::PerformanceIndex>,
    Conversion<Pose, legged_mpc_msgs::Pose>,
    Conversion<Quaternion, legged_mpc_msgs::Quaternion>,
    Conversion<ResetRequests, legged_mpc_msgs::ResetRequests>,
    Conversion<RobotModelInstance, legged_mpc_msgs::RobotModelInstance>,
    Conversion<RobotStateSample, legged_mpc_msgs::RobotStateSample>,
    Conversion<ScalarGroup, legged_mpc_msgs::ScalarGroup>,
    Conversion<Spheres, legged_mpc_msgs::Spheres>,
    Conversion<SystemObservation, legged_mpc_msgs::SystemObservation>,
    Conversion<TargetContactPatch, legged_mpc_msgs::TargetContactPatch>,
    Conversion<TargetTrajectories, legged_mpc_msgs::TargetTrajectories>,
    Conversion<TelemetrySeries, legged_mpc_msgs::TelemetrySeries>,
    Conversion<Vector, legged_mpc_msgs::Vector>,
    Conversion<Vector3, legged_mpc_msgs::Vector3>,
    Conversion<ViewerAnnotations, legged_mpc_msgs::ViewerAnnotations>,
    Conversion<VisualizationScene, legged_mpc_msgs::VisualizationScene>,
    Conversion<WalkingVelocityCommand, legged_mpc_msgs::WalkingVelocityCommand>,
    Conversion<Wrench, legged_mpc_msgs::Wrench>>;

template <typename T>
class MessageRoundTripTest : public ::testing::Test {};
TYPED_TEST_SUITE(MessageRoundTripTest, AllMessages);

TYPED_TEST(MessageRoundTripTest, ConvertsBothWaysWithoutLosingAnything) {
  ExpectRoundTrips<typename TypeParam::Struct, typename TypeParam::Proto>();
}

template <typename... Conversions>
absl::btree_set<std::string> filesOf(
    ::testing::Types<Conversions...> /*types*/) {
  return {std::string(Conversions::Proto::descriptor()->file()->name())...};
}

TEST(CoverageTest, EveryProtoFileOfThePackageIsTested) {
  absl::btree_set<std::string> covered = filesOf(AllMessages());
  covered.insert(std::string(
      legged_mpc_msgs::ControllerType_descriptor()->file()->name()));
  covered.insert(std::string(
      legged_mpc_msgs::ConfigFileKind_descriptor()->file()->name()));
  const std::filesystem::path directory =
      std::filesystem::path(std::getenv("TEST_SRCDIR")) / "_main" /
      "legged_nmpc" / "legged_mpc_msgs";
  size_t files = 0;
  for (const std::filesystem::directory_entry& entry :
       std::filesystem::directory_iterator(directory)) {
    if (entry.path().extension() != ".proto") {
      continue;
    }
    ++files;
    const std::string importPath =
        "legged_mpc_msgs/" + entry.path().filename().string();
    EXPECT_EQ(covered.count(importPath), 1u)
        << importPath << " has no round-trip test here";
  }
  EXPECT_EQ(files, covered.size());
}

TEST(EnumTest, ControllerTypeConvertsBothWaysAndRejectsUndefinedNumbers) {
  for (const ControllerType type :
       {ControllerType::kUnknown, ControllerType::kFeedforward,
        ControllerType::kLinear, ControllerType::kManifoldLinear}) {
    legged_mpc_msgs::ControllerType proto =
        legged_mpc_msgs::CONTROLLER_TYPE_UNKNOWN;
    ToProto(type, &proto);
    ControllerType back = ControllerType::kUnknown;
    ASSERT_TRUE(FromProto(proto, &back).ok());
    EXPECT_EQ(back, type);
  }
  legged_mpc_msgs::ControllerType linear =
      legged_mpc_msgs::CONTROLLER_TYPE_UNKNOWN;
  ToProto(ControllerType::kLinear, &linear);
  EXPECT_EQ(linear, legged_mpc_msgs::CONTROLLER_TYPE_LINEAR);
  // The wire codes are fixed: a robot and an MPC node built from other commits
  // still agree on them.
  legged_mpc_msgs::ControllerType manifoldLinear =
      legged_mpc_msgs::CONTROLLER_TYPE_UNKNOWN;
  ToProto(ControllerType::kManifoldLinear, &manifoldLinear);
  EXPECT_EQ(manifoldLinear, legged_mpc_msgs::CONTROLLER_TYPE_MANIFOLD_LINEAR);
  EXPECT_EQ(static_cast<int>(legged_mpc_msgs::CONTROLLER_TYPE_FEEDFORWARD), 1);
  EXPECT_EQ(static_cast<int>(legged_mpc_msgs::CONTROLLER_TYPE_LINEAR), 2);
  EXPECT_EQ(static_cast<int>(legged_mpc_msgs::CONTROLLER_TYPE_MANIFOLD_LINEAR),
            3);

  ControllerType value = ControllerType::kUnknown;
  const absl::Status status =
      FromProto(static_cast<legged_mpc_msgs::ControllerType>(17), &value);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(),
            "17 is not a value of legged_mpc_msgs.ControllerType");
}

TEST(EnumTest, ConfigFileKindConvertsBothWaysAndRejectsUndefinedNumbers) {
  for (const ConfigFileKind kind :
       {ConfigFileKind::kUnspecified, ConfigFileKind::kTask,
        ConfigFileKind::kReference, ConfigFileKind::kJointPdGains}) {
    legged_mpc_msgs::ConfigFileKind proto =
        legged_mpc_msgs::CONFIG_FILE_KIND_UNSPECIFIED;
    ToProto(kind, &proto);
    ConfigFileKind back = ConfigFileKind::kUnspecified;
    ASSERT_TRUE(FromProto(proto, &back).ok());
    EXPECT_EQ(back, kind);
  }
  ConfigFileKind value = ConfigFileKind::kUnspecified;
  EXPECT_EQ(
      FromProto(static_cast<legged_mpc_msgs::ConfigFileKind>(9), &value).code(),
      absl::StatusCode::kInvalidArgument);
}

// A save as the GUI sends it, and the robot's answer: the text travels byte for
// byte, the result by its name.
TEST(ConfigFileSaveTest, TheTextAndTheResultTravelUnchanged) {
  ConfigFileSave save;
  save.kind = ConfigFileKind::kJointPdGains;
  save.robot_name = "g1";
  save.config_path =
      "robot_models/unitree_g1/g1_wb_mpc/config/controller/"
      "joint_pd_gains.textproto";
  save.text = "# proto-file: x\ndefault_gains { kp: 100.0 }\n";
  save.sequence = 1'759'680'000'000'000'000ULL;
  legged_mpc_msgs::ConfigFileSave proto;
  ToProto(save, &proto);
  EXPECT_EQ(proto.kind(), legged_mpc_msgs::CONFIG_FILE_KIND_JOINT_PD_GAINS);
  EXPECT_EQ(proto.text(), save.text);
  legged_mpc_msgs::ConfigFileSave parsed;
  ASSERT_TRUE(parsed.ParseFromString(proto.SerializeAsString()));
  ConfigFileSave back;
  ASSERT_TRUE(FromProto(parsed, &back).ok());
  EXPECT_TRUE(back == save);

  ConfigFileSaveStatus status;
  status.sequence = save.sequence;
  status.kind = save.kind;
  status.result = ConfigFileSaveStatus::Result::kSaved;
  legged_mpc_msgs::ConfigFileSaveStatus statusProto;
  ToProto(status, &statusProto);
  EXPECT_EQ(statusProto.result(),
            legged_mpc_msgs::ConfigFileSaveStatus::RESULT_SAVED);
}

TEST(ErrorTest, AnUndefinedEnumNumberDeepInAPolicyNamesItsPath) {
  legged_mpc_msgs::MpcPolicy proto;
  proto.mutable_annotations()->add_target_contact_patches();
  proto.mutable_annotations()->add_target_contact_patches()->set_kind(
      static_cast<legged_mpc_msgs::TargetContactPatch_Kind>(9));
  MpcPolicy value;
  const absl::Status status = FromProto(proto, &value);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(),
            "annotations.target_contact_patches[1].kind: 9 is not a value of "
            "legged_mpc_msgs.TargetContactPatch.Kind");
}

// A policy written by hand: the members land in the fields of the same names,
// element by element.
TEST(MpcPolicyTest, AHandWrittenPolicyLandsInTheMessageFields) {
  constexpr int kNodes = 4;
  constexpr int kStateDim = 3;
  MpcPolicy policy;
  policy.resets_served = 3;
  policy.solver_status.healthy = true;
  policy.solver_status.last_error =
      "a message long enough to leave the small-string buffer";
  policy.init_observation.time = 1.25;
  policy.init_observation.state =
      Eigen::VectorXd::LinSpaced(kStateDim, 0.0, 2.0);
  policy.time_trajectory = Eigen::VectorXd::LinSpaced(kNodes, 1.25, 1.28);
  for (int node = 0; node < kNodes; ++node) {
    Vector state;
    state.data =
        Eigen::VectorXd::Constant(kStateDim, static_cast<double>(node));
    policy.state_trajectory.push_back(state);
  }
  policy.post_event_indices = {2};
  policy.mode_schedule.event_times = Eigen::VectorXd::Constant(1, 1.27);
  policy.mode_schedule.mode_sequence = {3, 1};
  policy.controller_type = ControllerType::kLinear;
  TargetContactPatch patch;
  patch.kind = TargetContactPatch::Kind::kNextSwing;
  patch.des_pose_foot_rt_world =
      TargetContactPatch::Pose{.x = 0.25,
                               .y = -0.1,
                               .z = 0.0,
                               .eulerzyx_foot_rt_world_yaw = 0.5,
                               .yaw_planned = true};
  policy.annotations.target_contact_patches.push_back(patch);
  // A foot without a target: no pose.
  policy.annotations.target_contact_patches.push_back(
      TargetContactPatch{.kind = TargetContactPatch::Kind::kStance,
                         .des_pose_foot_rt_world = std::nullopt});

  legged_mpc_msgs::MpcPolicy proto;
  ToProto(policy, &proto);
  EXPECT_EQ(proto.resets_served(), 3u);
  EXPECT_TRUE(proto.solver_status().healthy());
  EXPECT_EQ(proto.solver_status().last_error(),
            policy.solver_status.last_error);
  ASSERT_EQ(proto.time_trajectory_size(), kNodes);
  EXPECT_EQ(proto.time_trajectory(kNodes - 1),
            policy.time_trajectory[kNodes - 1]);
  ASSERT_EQ(proto.state_trajectory_size(), kNodes);
  ASSERT_EQ(proto.state_trajectory(2).data_size(), kStateDim);
  EXPECT_EQ(proto.state_trajectory(2).data(1), 2.0);
  EXPECT_EQ(proto.mode_schedule().mode_sequence(1), 1u);
  EXPECT_EQ(proto.controller_type(), legged_mpc_msgs::CONTROLLER_TYPE_LINEAR);
  ASSERT_EQ(proto.annotations().target_contact_patches_size(), 2);
  EXPECT_EQ(proto.annotations().target_contact_patches(0).kind(),
            legged_mpc_msgs::TargetContactPatch::KIND_NEXT_SWING);
  ASSERT_TRUE(proto.annotations()
                  .target_contact_patches(0)
                  .has_des_pose_foot_rt_world());
  EXPECT_EQ(proto.annotations()
                .target_contact_patches(0)
                .des_pose_foot_rt_world()
                .x(),
            0.25);
  EXPECT_FALSE(proto.annotations()
                   .target_contact_patches(1)
                   .has_des_pose_foot_rt_world());

  MpcPolicy back;
  ASSERT_TRUE(FromProto(proto, &back).ok());
  EXPECT_TRUE(back == policy);

  // Through the wire format too.
  legged_mpc_msgs::MpcPolicy parsed;
  ASSERT_TRUE(parsed.ParseFromString(proto.SerializeAsString()));
  EXPECT_TRUE(ProtoEquals(proto, parsed));
  MpcPolicy fromWire;
  ASSERT_TRUE(FromProto(parsed, &fromWire).ok());
  EXPECT_TRUE(fromWire == policy);
}

TEST(RobotStateSampleTest, AHandWrittenSampleLandsInTheMessageFields) {
  RobotStateSample sample;
  sample.time = 2.5;
  sample.control_mode = "WB_MPC";
  sample.quat_base_rt_world.w = 1.0;
  sample.joint_names = {"left_knee", "right_knee"};
  sample.joint_positions = Eigen::Vector2d(0.1, -0.2);
  sample.contact_flags = {true, false};
  Wrench measWrenchContactEwrtWorld;
  // NOLINTNEXTLINE(frame-naming): a component of a frame-generic value
  measWrenchContactEwrtWorld.force.z = 400.0;
  sample.meas_wrench_contacts_ewrt_world = {measWrenchContactEwrtWorld,
                                            Wrench()};

  legged_mpc_msgs::RobotStateSample proto;
  ToProto(sample, &proto);
  EXPECT_EQ(proto.control_mode(), "WB_MPC");
  EXPECT_EQ(proto.quat_base_rt_world().w(), 1.0);
  ASSERT_EQ(proto.joint_names_size(), 2);
  EXPECT_EQ(proto.joint_names(1), "right_knee");
  ASSERT_EQ(proto.joint_positions_size(), 2);
  EXPECT_EQ(proto.joint_positions(1), -0.2);
  ASSERT_EQ(proto.contact_flags_size(), 2);
  EXPECT_TRUE(proto.contact_flags(0));
  EXPECT_FALSE(proto.contact_flags(1));
  // NOLINTNEXTLINE(frame-naming): a component of a frame-generic value
  EXPECT_EQ(proto.meas_wrench_contacts_ewrt_world(0).force().z(), 400.0);

  RobotStateSample back;
  ASSERT_TRUE(FromProto(proto, &back).ok());
  EXPECT_TRUE(back == sample);
}

}  // namespace
}  // namespace ocs2::legged::msgs
