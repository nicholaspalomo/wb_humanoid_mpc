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

// The nproto structs of every humanoid_mpc_msgs message (ocs2::humanoid::msgs, tools/nproto/README.md): each one
// round-trips through its protobuf message both ways, and the test covers every .proto file of the package.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include <Eigen/Core>

#include "absl/status/status.h"

#include "humanoid_mpc_msgs/arrows.nproto.pb.h"
#include "humanoid_mpc_msgs/color.nproto.pb.h"
#include "humanoid_mpc_msgs/controller_type.nproto.pb.h"
#include "humanoid_mpc_msgs/fsm_command.nproto.pb.h"
#include "humanoid_mpc_msgs/fsm_state.nproto.pb.h"
#include "humanoid_mpc_msgs/joint_targets.nproto.pb.h"
#include "humanoid_mpc_msgs/line_strip.nproto.pb.h"
#include "humanoid_mpc_msgs/line_strips.nproto.pb.h"
#include "humanoid_mpc_msgs/loop_timing.nproto.pb.h"
#include "humanoid_mpc_msgs/mode_schedule.nproto.pb.h"
#include "humanoid_mpc_msgs/mpc_observation.nproto.pb.h"
#include "humanoid_mpc_msgs/mpc_policy.nproto.pb.h"
#include "humanoid_mpc_msgs/mpc_solver_status.nproto.pb.h"
#include "humanoid_mpc_msgs/mpc_status.nproto.pb.h"
#include "humanoid_mpc_msgs/performance_index.nproto.pb.h"
#include "humanoid_mpc_msgs/pose.nproto.pb.h"
#include "humanoid_mpc_msgs/quaternion.nproto.pb.h"
#include "humanoid_mpc_msgs/reset_requests.nproto.pb.h"
#include "humanoid_mpc_msgs/robot_model_instance.nproto.pb.h"
#include "humanoid_mpc_msgs/robot_state_sample.nproto.pb.h"
#include "humanoid_mpc_msgs/scalar_group.nproto.pb.h"
#include "humanoid_mpc_msgs/spheres.nproto.pb.h"
#include "humanoid_mpc_msgs/system_observation.nproto.pb.h"
#include "humanoid_mpc_msgs/target_contact_patch.nproto.pb.h"
#include "humanoid_mpc_msgs/target_trajectories.nproto.pb.h"
#include "humanoid_mpc_msgs/telemetry_series.nproto.pb.h"
#include "humanoid_mpc_msgs/vector.nproto.pb.h"
#include "humanoid_mpc_msgs/vector3.nproto.pb.h"
#include "humanoid_mpc_msgs/viewer_annotations.nproto.pb.h"
#include "humanoid_mpc_msgs/visualization_scene.nproto.pb.h"
#include "humanoid_mpc_msgs/walking_velocity_command.nproto.pb.h"
#include "humanoid_mpc_msgs/wrench.nproto.pb.h"
#include "humanoid_mpc_msgs/yaml_document.nproto.pb.h"
#include "tools/nproto/test/ProtoTestValues.h"

namespace ocs2::humanoid::msgs {
namespace {

using nproto::test_support::ExpectRoundTrips;
using nproto::test_support::ProtoEquals;

// The mappings the realtime and IPC code relies on.
static_assert(std::is_same_v<decltype(MpcPolicy::time_trajectory), Eigen::VectorXd>);
static_assert(std::is_same_v<decltype(MpcPolicy::state_trajectory), std::vector<Vector>>);
static_assert(std::is_same_v<decltype(MpcPolicy::controller_type), ControllerType>);
static_assert(std::is_same_v<decltype(Vector::data), Eigen::VectorXd>);
static_assert(std::is_same_v<decltype(SystemObservation::state), Eigen::VectorXd>);
static_assert(std::is_same_v<decltype(SystemObservation::mode), std::uint64_t>);
static_assert(std::is_same_v<decltype(ModeSchedule::mode_sequence), std::vector<std::uint64_t>>);
static_assert(std::is_same_v<decltype(RobotStateSample::joint_names), std::vector<std::string>>);
static_assert(std::is_same_v<decltype(RobotStateSample::contact_flags), std::vector<bool>>);
static_assert(std::is_same_v<decltype(RobotStateSample::measured_contact_wrenches), std::vector<Wrench>>);
static_assert(std::is_same_v<decltype(JointTargets::positions), std::map<std::string, double>>);
static_assert(std::is_same_v<decltype(Spheres::radii), Eigen::VectorXf>);
static_assert(std::is_same_v<decltype(TargetContactPatch::kind), TargetContactPatch::Kind>);
static_assert(std::is_enum_v<ControllerType> && !std::is_convertible_v<ControllerType, int>);

template <typename StructType, typename ProtoType>
struct Conversion {
  using Struct = StructType;
  using Proto = ProtoType;
};

// Every message of the package; controller_type.proto is the one enum (EnumTest below).
using AllMessages = ::testing::Types<Conversion<Arrows, humanoid_mpc_msgs::Arrows>,
                                     Conversion<Color, humanoid_mpc_msgs::Color>,
                                     Conversion<FsmCommand, humanoid_mpc_msgs::FsmCommand>,
                                     Conversion<FsmState, humanoid_mpc_msgs::FsmState>,
                                     Conversion<JointTargets, humanoid_mpc_msgs::JointTargets>,
                                     Conversion<LineStrip, humanoid_mpc_msgs::LineStrip>,
                                     Conversion<LineStrips, humanoid_mpc_msgs::LineStrips>,
                                     Conversion<LoopTiming, humanoid_mpc_msgs::LoopTiming>,
                                     Conversion<ModeSchedule, humanoid_mpc_msgs::ModeSchedule>,
                                     Conversion<MpcObservation, humanoid_mpc_msgs::MpcObservation>,
                                     Conversion<MpcPolicy, humanoid_mpc_msgs::MpcPolicy>,
                                     Conversion<MpcSolverStatus, humanoid_mpc_msgs::MpcSolverStatus>,
                                     Conversion<MpcStatus, humanoid_mpc_msgs::MpcStatus>,
                                     Conversion<PerformanceIndex, humanoid_mpc_msgs::PerformanceIndex>,
                                     Conversion<Pose, humanoid_mpc_msgs::Pose>,
                                     Conversion<Quaternion, humanoid_mpc_msgs::Quaternion>,
                                     Conversion<ResetRequests, humanoid_mpc_msgs::ResetRequests>,
                                     Conversion<RobotModelInstance, humanoid_mpc_msgs::RobotModelInstance>,
                                     Conversion<RobotStateSample, humanoid_mpc_msgs::RobotStateSample>,
                                     Conversion<ScalarGroup, humanoid_mpc_msgs::ScalarGroup>,
                                     Conversion<Spheres, humanoid_mpc_msgs::Spheres>,
                                     Conversion<SystemObservation, humanoid_mpc_msgs::SystemObservation>,
                                     Conversion<TargetContactPatch, humanoid_mpc_msgs::TargetContactPatch>,
                                     Conversion<TargetTrajectories, humanoid_mpc_msgs::TargetTrajectories>,
                                     Conversion<TelemetrySeries, humanoid_mpc_msgs::TelemetrySeries>,
                                     Conversion<Vector, humanoid_mpc_msgs::Vector>,
                                     Conversion<Vector3, humanoid_mpc_msgs::Vector3>,
                                     Conversion<ViewerAnnotations, humanoid_mpc_msgs::ViewerAnnotations>,
                                     Conversion<VisualizationScene, humanoid_mpc_msgs::VisualizationScene>,
                                     Conversion<WalkingVelocityCommand, humanoid_mpc_msgs::WalkingVelocityCommand>,
                                     Conversion<Wrench, humanoid_mpc_msgs::Wrench>,
                                     Conversion<YamlDocument, humanoid_mpc_msgs::YamlDocument>>;

template <typename T>
class MessageRoundTripTest : public ::testing::Test {};
TYPED_TEST_SUITE(MessageRoundTripTest, AllMessages);

TYPED_TEST(MessageRoundTripTest, ConvertsBothWaysWithoutLosingAnything) {
  ExpectRoundTrips<typename TypeParam::Struct, typename TypeParam::Proto>();
}

template <typename... Conversions>
std::set<std::string> filesOf(::testing::Types<Conversions...> /*types*/) {
  return {std::string(Conversions::Proto::descriptor()->file()->name())...};
}

TEST(CoverageTest, EveryProtoFileOfThePackageIsTested) {
  std::set<std::string> covered = filesOf(AllMessages());
  covered.insert(std::string(humanoid_mpc_msgs::ControllerType_descriptor()->file()->name()));
  const std::filesystem::path directory =
      std::filesystem::path(std::getenv("TEST_SRCDIR")) / "_main" / "humanoid_nmpc" / "humanoid_mpc_msgs";
  std::size_t files = 0;
  for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(directory)) {
    if (entry.path().extension() != ".proto") {
      continue;
    }
    ++files;
    const std::string importPath = "humanoid_mpc_msgs/" + entry.path().filename().string();
    EXPECT_EQ(covered.count(importPath), 1u) << importPath << " has no round-trip test here";
  }
  EXPECT_EQ(files, covered.size());
}

TEST(EnumTest, ControllerTypeConvertsBothWaysAndRejectsUndefinedNumbers) {
  for (const ControllerType type : {ControllerType::kUnknown, ControllerType::kFeedforward, ControllerType::kLinear}) {
    humanoid_mpc_msgs::ControllerType proto = humanoid_mpc_msgs::CONTROLLER_TYPE_UNKNOWN;
    ToProto(type, &proto);
    ControllerType back = ControllerType::kUnknown;
    ASSERT_TRUE(FromProto(proto, &back).ok());
    EXPECT_EQ(back, type);
  }
  humanoid_mpc_msgs::ControllerType linear = humanoid_mpc_msgs::CONTROLLER_TYPE_UNKNOWN;
  ToProto(ControllerType::kLinear, &linear);
  EXPECT_EQ(linear, humanoid_mpc_msgs::CONTROLLER_TYPE_LINEAR);

  ControllerType value = ControllerType::kUnknown;
  const absl::Status status = FromProto(static_cast<humanoid_mpc_msgs::ControllerType>(17), &value);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(), "17 is not a value of humanoid_mpc_msgs.ControllerType");
}

TEST(ErrorTest, AnUndefinedEnumNumberDeepInAPolicyNamesItsPath) {
  humanoid_mpc_msgs::MpcPolicy proto;
  proto.mutable_annotations()->add_target_contact_patches();
  proto.mutable_annotations()->add_target_contact_patches()->set_kind(static_cast<humanoid_mpc_msgs::TargetContactPatch_Kind>(9));
  MpcPolicy value;
  const absl::Status status = FromProto(proto, &value);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(), "annotations.target_contact_patches[1].kind: 9 is not a value of humanoid_mpc_msgs.TargetContactPatch.Kind");
}

// A policy written by hand: the members land in the fields of the same names, element by element.
TEST(MpcPolicyTest, AHandWrittenPolicyLandsInTheMessageFields) {
  constexpr int kNodes = 4;
  constexpr int kStateDim = 3;
  MpcPolicy policy;
  policy.resets_served = 3;
  policy.solver_status.healthy = true;
  policy.solver_status.last_error = "a message long enough to leave the small-string buffer";
  policy.init_observation.time = 1.25;
  policy.init_observation.state = Eigen::VectorXd::LinSpaced(kStateDim, 0.0, 2.0);
  policy.time_trajectory = Eigen::VectorXd::LinSpaced(kNodes, 1.25, 1.28);
  for (int node = 0; node < kNodes; ++node) {
    Vector state;
    state.data = Eigen::VectorXd::Constant(kStateDim, static_cast<double>(node));
    policy.state_trajectory.push_back(state);
  }
  policy.post_event_indices = {2};
  policy.mode_schedule.event_times = Eigen::VectorXd::Constant(1, 1.27);
  policy.mode_schedule.mode_sequence = {3, 1};
  policy.controller_type = ControllerType::kLinear;
  TargetContactPatch patch;
  patch.valid = true;
  patch.kind = TargetContactPatch::Kind::kNextSwing;
  policy.annotations.target_contact_patches.push_back(patch);

  humanoid_mpc_msgs::MpcPolicy proto;
  ToProto(policy, &proto);
  EXPECT_EQ(proto.resets_served(), 3u);
  EXPECT_TRUE(proto.solver_status().healthy());
  EXPECT_EQ(proto.solver_status().last_error(), policy.solver_status.last_error);
  ASSERT_EQ(proto.time_trajectory_size(), kNodes);
  EXPECT_EQ(proto.time_trajectory(kNodes - 1), policy.time_trajectory[kNodes - 1]);
  ASSERT_EQ(proto.state_trajectory_size(), kNodes);
  ASSERT_EQ(proto.state_trajectory(2).data_size(), kStateDim);
  EXPECT_EQ(proto.state_trajectory(2).data(1), 2.0);
  EXPECT_EQ(proto.mode_schedule().mode_sequence(1), 1u);
  EXPECT_EQ(proto.controller_type(), humanoid_mpc_msgs::CONTROLLER_TYPE_LINEAR);
  ASSERT_EQ(proto.annotations().target_contact_patches_size(), 1);
  EXPECT_EQ(proto.annotations().target_contact_patches(0).kind(), humanoid_mpc_msgs::TargetContactPatch::KIND_NEXT_SWING);

  MpcPolicy back;
  ASSERT_TRUE(FromProto(proto, &back).ok());
  EXPECT_TRUE(back == policy);

  // Through the wire format too.
  humanoid_mpc_msgs::MpcPolicy parsed;
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
  sample.base_orientation_world.w = 1.0;
  sample.joint_names = {"left_knee", "right_knee"};
  sample.joint_positions = Eigen::Vector2d(0.1, -0.2);
  sample.contact_flags = {true, false};
  Wrench wrench;
  wrench.force.z = 400.0;
  sample.measured_contact_wrenches = {wrench, Wrench()};

  humanoid_mpc_msgs::RobotStateSample proto;
  ToProto(sample, &proto);
  EXPECT_EQ(proto.control_mode(), "WB_MPC");
  EXPECT_EQ(proto.base_orientation_world().w(), 1.0);
  ASSERT_EQ(proto.joint_names_size(), 2);
  EXPECT_EQ(proto.joint_names(1), "right_knee");
  ASSERT_EQ(proto.joint_positions_size(), 2);
  EXPECT_EQ(proto.joint_positions(1), -0.2);
  ASSERT_EQ(proto.contact_flags_size(), 2);
  EXPECT_TRUE(proto.contact_flags(0));
  EXPECT_FALSE(proto.contact_flags(1));
  EXPECT_EQ(proto.measured_contact_wrenches(0).force().z(), 400.0);

  RobotStateSample back;
  ASSERT_TRUE(FromProto(proto, &back).ok());
  EXPECT_TRUE(back == sample);
}

}  // namespace
}  // namespace ocs2::humanoid::msgs
