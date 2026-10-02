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

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <ocs2_mpc/CommandData.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

#include "VisualizationTestRobot.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc_app/visualization/PolicySnapshot.h"
#include "humanoid_common_mpc_app/visualization/RobotStateDecoder.h"
#include "humanoid_common_mpc_app/visualization/TelemetryBuilder.h"
#include "humanoid_common_mpc_app/visualization/VisualizationConfig.h"
#include "humanoid_mpc_msgs/robot_state_sample.nproto.pb.h"

namespace ocs2::humanoid::visualization {
namespace {

constexpr scalar_t kTolerance = 1e-9;

const humanoid_mpc_msgs::ScalarGroup* findGroup(const humanoid_mpc_msgs::TelemetrySeries& series, const std::string& path) {
  for (const humanoid_mpc_msgs::ScalarGroup& group : series.groups()) {
    if (group.path() == path) {
      return &group;
    }
  }
  return nullptr;
}

/** The value `name` of the group at `path`. */
double valueOf(const humanoid_mpc_msgs::TelemetrySeries& series, const std::string& path, const std::string& name) {
  const humanoid_mpc_msgs::ScalarGroup* group = findGroup(series, path);
  if (group == nullptr) {
    ADD_FAILURE() << "no group " << path;
    return NAN;
  }
  for (int index = 0; index < group->names_size(); ++index) {
    if (group->names(index) == name) {
      return group->values(index);
    }
  }
  ADD_FAILURE() << "no series " << name << " in " << path;
  return NAN;
}

/** Every value of the group at `path`. */
vector_t valuesOf(const humanoid_mpc_msgs::TelemetrySeries& series, const std::string& path) {
  const humanoid_mpc_msgs::ScalarGroup* group = findGroup(series, path);
  if (group == nullptr) {
    ADD_FAILURE() << "no group " << path;
    return vector_t();
  }
  return Eigen::Map<const vector_t>(group->values().data(), group->values_size());
}

/**
 * The paths of the telemetry contract (humanoid_nmpc/humanoid_rerun_viewer/README.md) for `frames`, written out here
 * independently of the builder; test_end_to_end compares them with the bridge's contract module itself.
 */
std::vector<std::string> contractPaths(const std::vector<std::string>& frames) {
  std::vector<std::string> paths = {"base_pose/position_x",
                                    "base_pose/position_y",
                                    "base_pose/position_z",
                                    "base_pose/roll",
                                    "base_pose/pitch",
                                    "base_pose/yaw",
                                    "base_twist/linear_x",
                                    "base_twist/linear_y",
                                    "base_twist/linear_z",
                                    "base_twist/angular_x",
                                    "base_twist/angular_y",
                                    "base_twist/angular_z",
                                    "contact_forces/left_normal",
                                    "contact_forces/right_normal",
                                    "contact_forces/left_tangential",
                                    "contact_forces/right_tangential",
                                    "generalized_base/position_z",
                                    "generalized_base/pitch",
                                    "generalized_base/roll",
                                    "generalized_base/velocity_z",
                                    "generalized_base/pitch_rate",
                                    "generalized_base/roll_rate",
                                    "foot_kinematics/left_acceleration_z",
                                    "foot_kinematics/right_acceleration_z",
                                    "foot_kinematics/left_velocity_z",
                                    "foot_kinematics/right_velocity_z",
                                    "joints/position/measured",
                                    "joints/position/target",
                                    "joints/velocity/measured",
                                    "joints/velocity/target",
                                    "joints/effort/measured",
                                    "joints/effort/target",
                                    "dofs/position/measured",
                                    "dofs/position/reference",
                                    "dofs/position/plan",
                                    "dofs/velocity/measured",
                                    "dofs/velocity/reference",
                                    "dofs/velocity/plan",
                                    "dofs/force/measured",
                                    "dofs/force/reference",
                                    "contact_wrenches/left/mpc",
                                    "contact_wrenches/left/measured",
                                    "contact_wrenches/right/mpc",
                                    "contact_wrenches/right/measured",
                                    "mpc_observation/state",
                                    "mpc_observation/input",
                                    "mpc_observation/mode"};
  for (const std::string& frame : frames) {
    for (const char* kind : {"pose", "twist", "acceleration", "wrench"}) {
      for (const char* source : {"measured", "reference", "plan"}) {
        paths.push_back(absl::StrCat("frames/", kind, "/", frame, "/", source));
      }
    }
  }
  return paths;
}

class TelemetryBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override { load(test::g1CentroidalFiles(), test::Formulation::kCentroidal); }

  void load(const test::RobotFiles& files, test::Formulation formulation) {
    robot_ = test::TestRobot::load(files, formulation);
    const absl::StatusOr<VisualizationConfig> config = loadVisualizationConfig(robot_->taskFile(), robot_->modelSettings());
    ASSERT_TRUE(config.ok()) << config.status();
    config_ = *config;
    absl::StatusOr<std::unique_ptr<TelemetryBuilder>> builder = TelemetryBuilder::Create(robot_->model(), config_);
    ASSERT_TRUE(builder.ok()) << builder.status();
    builder_ = std::move(*builder);
    decoder_ = std::make_unique<RobotStateDecoder>(robot_->modelSettings());

    CommandData command;
    PrimalSolution solution;
    robot_->makePolicy(/*startTime=*/0.0, /*nodes=*/21, /*normalForce=*/300.0, vector2_t(0.02, -0.01), &command, &solution);
    policy_.assign(command, solution);
    observation_ = robot_->observation(/*time=*/0.0, ModeNumber::STANCE);
  }

  DecodedRobotState decode(const humanoid_mpc_msgs::RobotStateSample& proto) {
    msgs::RobotStateSample sample;
    EXPECT_TRUE(msgs::FromProto(proto, &sample).ok());
    DecodedRobotState state;
    EXPECT_TRUE(decoder_->decode(sample, &state).ok());
    return state;
  }

  std::unique_ptr<test::TestRobot> robot_;
  VisualizationConfig config_;
  std::unique_ptr<TelemetryBuilder> builder_;
  std::unique_ptr<RobotStateDecoder> decoder_;
  PolicySnapshot policy_;
  SystemObservation observation_;
};

TEST_F(TelemetryBuilderTest, TheGroupsAreTheContractsInItsOrder) {
  EXPECT_EQ(builder_->groupPaths(), contractPaths(config_.telemetryFrames));
  // 26 panel groups, 21 complete groups of the robot and 12 per tracked frame.
  EXPECT_EQ(builder_->groupPaths().size(), 26 + 21 + 12 * config_.telemetryFrames.size());
  const humanoid_mpc_msgs::TelemetrySeries& series =
      builder_->build(decode(robot_->robotState(/*time=*/0.1, vector3_t::Zero(), vector3_t::Zero())), &observation_, &policy_);
  absl::flat_hash_set<std::string> paths;
  for (const humanoid_mpc_msgs::ScalarGroup& group : series.groups()) {
    EXPECT_TRUE(paths.insert(group.path()).second) << group.path();
    EXPECT_EQ(group.names_size(), group.values_size()) << group.path();
    EXPECT_GT(group.names_size(), 0) << group.path();
  }
  const std::vector<std::string>& joints = robot_->modelSettings().fullJointNames;
  const humanoid_mpc_msgs::ScalarGroup* jointPositions = findGroup(series, "joints/position/measured");
  ASSERT_NE(jointPositions, nullptr);
  EXPECT_EQ(std::vector<std::string>(jointPositions->names().begin(), jointPositions->names().end()), joints);
  const humanoid_mpc_msgs::ScalarGroup* dofs = findGroup(series, "dofs/position/plan");
  ASSERT_NE(dofs, nullptr);
  EXPECT_EQ(dofs->names(0), "base_x");
  EXPECT_EQ(dofs->names(3), "base_yaw");
  EXPECT_EQ(dofs->names(6), robot_->modelSettings().mpcModelJointNames.front());
  EXPECT_EQ(findGroup(series, "mpc_observation/state")->names(0), "x0");
  EXPECT_EQ(findGroup(series, "mpc_observation/input")->names_size(), static_cast<int>(robot_->robotModel().getInputDim()));
}

TEST_F(TelemetryBuilderTest, TheBaseIsMeasuredInTheWorldFrameAsRollPitchYaw) {
  humanoid_mpc_msgs::RobotStateSample proto = robot_->robotState(/*time=*/0.1, vector3_t(1.0, 2.0, 3.0), vector3_t(0.1, 0.2, 0.3));
  proto.mutable_base_linear_velocity_local()->set_x(1.0);
  proto.mutable_base_angular_velocity_local()->set_z(0.5);
  const humanoid_mpc_msgs::TelemetrySeries& series = builder_->build(decode(proto), /*observation=*/nullptr, /*policy=*/nullptr);
  const matrix3_t rotation = test::quaternionFromRollPitchYaw(vector3_t(0.1, 0.2, 0.3)).toRotationMatrix();
  const vector3_t linear = rotation * vector3_t(1.0, 0.0, 0.0);
  const vector3_t angular = rotation * vector3_t(0.0, 0.0, 0.5);
  EXPECT_EQ(series.time(), 0.1);
  EXPECT_NEAR(valueOf(series, "base_pose/position_x", "measured"), 1.0, kTolerance);
  EXPECT_NEAR(valueOf(series, "base_pose/position_y", "measured"), 2.0, kTolerance);
  EXPECT_NEAR(valueOf(series, "base_pose/position_z", "measured"), 3.0, kTolerance);
  EXPECT_NEAR(valueOf(series, "base_pose/roll", "measured"), 0.1, kTolerance);
  EXPECT_NEAR(valueOf(series, "base_pose/pitch", "measured"), 0.2, kTolerance);
  EXPECT_NEAR(valueOf(series, "base_pose/yaw", "measured"), 0.3, kTolerance);
  EXPECT_NEAR(valueOf(series, "base_twist/linear_x", "measured"), linear.x(), kTolerance);
  EXPECT_NEAR(valueOf(series, "base_twist/linear_y", "measured"), linear.y(), kTolerance);
  EXPECT_NEAR(valueOf(series, "base_twist/linear_z", "measured"), linear.z(), kTolerance);
  EXPECT_NEAR(valueOf(series, "base_twist/angular_x", "measured"), angular.x(), kTolerance);
  EXPECT_NEAR(valueOf(series, "base_twist/angular_z", "measured"), angular.z(), kTolerance);
  // The generalized coordinates: z, then the Euler ZYX angles by their names.
  EXPECT_NEAR(valueOf(series, "generalized_base/position_z", "measured"), 3.0, kTolerance);
  EXPECT_NEAR(valueOf(series, "generalized_base/pitch", "measured"), 0.2, kTolerance);
  EXPECT_NEAR(valueOf(series, "generalized_base/roll", "measured"), 0.1, kTolerance);
  EXPECT_NEAR(valueOf(series, "generalized_base/velocity_z", "measured"), linear.z(), kTolerance);
  EXPECT_NEAR(valueOf(series, "dofs/position/measured", "base_yaw"), 0.3, kTolerance);
  // The root link's pose is the base's.
  const std::string root = "pelvis";
  ASSERT_NE(findGroup(series, "frames/pose/pelvis/measured"), nullptr);
  const vector_t pose = valuesOf(series, "frames/pose/" + root + "/measured");
  EXPECT_TRUE(pose.isApprox((vector_t(6) << 1.0, 2.0, 3.0, 0.1, 0.2, 0.3).finished(), kTolerance)) << pose.transpose();
  const vector_t twist = valuesOf(series, "frames/twist/" + root + "/measured");
  EXPECT_TRUE(twist.head<3>().isApprox(linear, kTolerance));
  EXPECT_TRUE(twist.tail<3>().isApprox(angular, kTolerance));
}

TEST_F(TelemetryBuilderTest, WithoutAPolicyTheReferenceAndThePlanAreTheMeasuredRobotAtRest) {
  humanoid_mpc_msgs::RobotStateSample proto = robot_->robotState(/*time=*/0.1, vector3_t(1.0, 2.0, 3.0), vector3_t(0.1, 0.2, 0.3));
  proto.mutable_base_linear_velocity_local()->set_x(1.0);
  const humanoid_mpc_msgs::TelemetrySeries& series = builder_->build(decode(proto), /*observation=*/nullptr, /*policy=*/nullptr);
  EXPECT_EQ(valueOf(series, "base_pose/position_x", "reference"), 1.0);
  EXPECT_NEAR(valueOf(series, "base_pose/yaw", "reference"), 0.3, kTolerance);
  EXPECT_EQ(valueOf(series, "base_twist/linear_x", "reference"), 0.0);
  EXPECT_EQ(valuesOf(series, "dofs/position/reference"), valuesOf(series, "dofs/position/measured"));
  EXPECT_EQ(valuesOf(series, "dofs/position/plan"), valuesOf(series, "dofs/position/measured"));
  EXPECT_TRUE(valuesOf(series, "dofs/velocity/reference").isZero());
  EXPECT_TRUE(valuesOf(series, "dofs/velocity/plan").isZero());
  EXPECT_TRUE(valuesOf(series, "contact_wrenches/left/mpc").isZero());
  EXPECT_EQ(valueOf(series, "contact_forces/right_normal", "mpc"), 0.0);
  EXPECT_TRUE(valuesOf(series, "mpc_observation/state").isZero());
  EXPECT_TRUE(valuesOf(series, "mpc_observation/input").isZero());
}

TEST_F(TelemetryBuilderTest, TheReferenceIsTheTargetAtTheSamplesTime) {
  const humanoid_mpc_msgs::TelemetrySeries& series =
      builder_->build(decode(robot_->robotState(/*time=*/0.5, vector3_t::Zero(), vector3_t::Zero())), &observation_, &policy_);
  // The target runs linearly from the nominal state at 0 s to the plan's last state at 1 s.
  const vector_t target = 0.5 * (policy_.target.stateTrajectory.front() + policy_.target.stateTrajectory.back());
  const MpcRobotModelBase<scalar_t>& model = robot_->robotModel();
  EXPECT_TRUE(valuesOf(series, "dofs/position/reference").isApprox(model.getGeneralizedCoordinates(target), kTolerance));
  EXPECT_NEAR(valueOf(series, "base_pose/position_x", "reference"), model.getBasePosition(target).x(), kTolerance);
  EXPECT_NEAR(valueOf(series, "base_pose/yaw", "reference"), model.getBaseOrientationEulerZYX(target)(0), kTolerance);
  EXPECT_NEAR(valueOf(series, "base_pose/roll", "reference"), model.getBaseOrientationEulerZYX(target)(2), kTolerance);
  EXPECT_NEAR(valueOf(series, "base_twist/linear_x", "reference"), model.getBaseComLinearVelocity(target).x(), kTolerance);
  EXPECT_EQ(valueOf(series, "base_twist/angular_z", "reference"), 0.0);
}

TEST_F(TelemetryBuilderTest, ThePlanIsThePolicyAtTheSamplesTime) {
  const scalar_t time = 0.1;
  const humanoid_mpc_msgs::TelemetrySeries& series =
      builder_->build(decode(robot_->robotState(time, vector3_t::Zero(), vector3_t::Zero())), &observation_, &policy_);
  vector_t state;
  vector_t input;
  samplePlan(policy_, time, &state, &input);
  const MpcRobotModelBase<scalar_t>& model = robot_->robotModel();
  EXPECT_TRUE(valuesOf(series, "dofs/position/plan").isApprox(model.getGeneralizedCoordinates(state), kTolerance));
  const vector6_t left = model.getContactWrenchInWorldFrame(state, input, CONTACT_LEFT_INDEX);
  EXPECT_TRUE(valuesOf(series, "contact_wrenches/left/mpc").isApprox(left, kTolerance));
  // Written as 300 N up and 15 N forward on each stance foot (world-frame wrench inputs).
  EXPECT_NEAR(valueOf(series, "contact_forces/left_normal", "mpc"), 300.0, 1e-9);
  EXPECT_NEAR(valueOf(series, "contact_forces/left_tangential", "mpc_x"), 15.0, 1e-9);
  EXPECT_TRUE(valuesOf(series, "frames/wrench/foot_l_contact/plan").isApprox(left, kTolerance));
  EXPECT_TRUE(valuesOf(series, "frames/wrench/pelvis/plan").isZero());
}

TEST_F(TelemetryBuilderTest, TheMeasuredWrenchesAreTheSensors) {
  humanoid_mpc_msgs::RobotStateSample proto = robot_->robotState(/*time=*/0.1, vector3_t::Zero(), vector3_t::Zero());
  humanoid_mpc_msgs::Wrench* left = proto.mutable_measured_contact_wrenches(0);
  left->mutable_force()->set_x(5.0);
  left->mutable_force()->set_y(-6.0);
  left->mutable_force()->set_z(310.0);
  left->mutable_torque()->set_y(2.0);
  const humanoid_mpc_msgs::TelemetrySeries& series = builder_->build(decode(proto), &observation_, &policy_);
  EXPECT_EQ(valuesOf(series, "contact_wrenches/left/measured"), (vector_t(3) << 5.0, -6.0, 310.0).finished());
  EXPECT_TRUE(valuesOf(series, "contact_wrenches/right/measured").isZero());
  EXPECT_EQ(valueOf(series, "contact_forces/left_normal", "measured"), 310.0);
  EXPECT_EQ(valueOf(series, "contact_forces/left_tangential", "measured_x"), 5.0);
  EXPECT_EQ(valueOf(series, "contact_forces/left_tangential", "measured_y"), -6.0);
  EXPECT_EQ(valuesOf(series, "frames/wrench/foot_l_contact/measured"), (vector_t(6) << 5.0, -6.0, 310.0, 0.0, 2.0, 0.0).finished());
  EXPECT_TRUE(valuesOf(series, "frames/wrench/pelvis/measured").isZero());
}

TEST_F(TelemetryBuilderTest, TheMeasuredAccelerationIsTheDifferenceOfConsecutiveSamples) {
  builder_->build(decode(robot_->robotState(/*time=*/1.0, vector3_t::Zero(), vector3_t::Zero())), /*observation=*/nullptr,
                  /*policy=*/nullptr);
  humanoid_mpc_msgs::RobotStateSample proto = robot_->robotState(/*time=*/1.01, vector3_t::Zero(), vector3_t::Zero());
  proto.mutable_base_linear_velocity_local()->set_z(0.1);
  const humanoid_mpc_msgs::TelemetrySeries& series = builder_->build(decode(proto), /*observation=*/nullptr, /*policy=*/nullptr);
  // 0.1 m/s more in 10 ms, at the root link, which does not rotate.
  EXPECT_NEAR(valueOf(series, "frames/acceleration/pelvis/measured", "linear_z"), 10.0, 1e-6);
  EXPECT_NEAR(valueOf(series, "frames/acceleration/pelvis/measured", "linear_x"), 0.0, 1e-9);

  // The robot's clock went back (a simulation restarted): no acceleration from a sample of the previous run.
  proto.set_time(0.5);
  proto.mutable_base_linear_velocity_local()->set_z(-3.0);
  const humanoid_mpc_msgs::TelemetrySeries& rewound = builder_->build(decode(proto), /*observation=*/nullptr, /*policy=*/nullptr);
  EXPECT_EQ(valueOf(rewound, "frames/acceleration/pelvis/measured", "linear_z"), 0.0);
  builder_->reset();
  proto.set_time(0.51);
  const humanoid_mpc_msgs::TelemetrySeries& afterReset = builder_->build(decode(proto), /*observation=*/nullptr, /*policy=*/nullptr);
  EXPECT_EQ(valueOf(afterReset, "frames/acceleration/pelvis/measured", "linear_z"), 0.0);
}

TEST_F(TelemetryBuilderTest, TheObservationGroupsAreTheObservationAndThePolicysInputThere) {
  observation_.time = 0.2;
  observation_.mode = ModeNumber::LF;
  const humanoid_mpc_msgs::TelemetrySeries& series =
      builder_->build(decode(robot_->robotState(/*time=*/0.25, vector3_t::Zero(), vector3_t::Zero())), &observation_, &policy_);
  vector_t state;
  vector_t input;
  samplePlan(policy_, observation_.time, &state, &input);
  EXPECT_EQ(valuesOf(series, "mpc_observation/state"), observation_.state);
  EXPECT_TRUE(valuesOf(series, "mpc_observation/input").isApprox(input, kTolerance));
  EXPECT_EQ(valueOf(series, "mpc_observation/mode", "mode"), static_cast<double>(ModeNumber::LF));
}

TEST_F(TelemetryBuilderTest, TheJointGroupsAreTheDecodedJoints) {
  humanoid_mpc_msgs::RobotStateSample proto = robot_->robotState(/*time=*/0.1, vector3_t::Zero(), vector3_t::Zero());
  for (int joint = 0; joint < proto.joint_names_size(); ++joint) {
    proto.add_joint_position_targets(0.3);
    proto.add_joint_velocity_targets(0.0);
    proto.add_joint_kp(50.0);
    proto.add_joint_kd(1.0);
    proto.add_joint_feed_forward_efforts(2.0);
  }
  const DecodedRobotState state = decode(proto);
  const humanoid_mpc_msgs::TelemetrySeries& series = builder_->build(state, &observation_, &policy_);
  EXPECT_EQ(valuesOf(series, "joints/position/measured"), state.jointPositions);
  EXPECT_EQ(valuesOf(series, "joints/position/target"), state.jointPositionTargets);
  EXPECT_EQ(valuesOf(series, "joints/velocity/measured"), state.jointVelocities);
  EXPECT_EQ(valuesOf(series, "joints/effort/measured"), state.jointEfforts);
  EXPECT_EQ(valuesOf(series, "joints/effort/target"), state.jointFeedForwardEfforts);
  EXPECT_EQ(valuesOf(series, "dofs/force/measured"), state.generalizedForces);
  const vector_t referenceForces = valuesOf(series, "dofs/force/reference");
  EXPECT_TRUE(referenceForces.head<6>().isZero());
  EXPECT_TRUE((referenceForces.tail(referenceForces.size() - 6).array() == 2.0).all());
}

TEST_F(TelemetryBuilderTest, EveryValueIsFinite) {
  for (const PolicySnapshot* policy : {static_cast<const PolicySnapshot*>(nullptr), static_cast<const PolicySnapshot*>(&policy_)}) {
    for (const scalar_t time : {-1.0, 0.0, 0.3, 0.6, 2.0}) {
      const humanoid_mpc_msgs::TelemetrySeries& series =
          builder_->build(decode(robot_->robotState(time, vector3_t(0.0, 0.0, 0.7), vector3_t(0.0, 0.1, 0.0))), &observation_, policy);
      for (const humanoid_mpc_msgs::ScalarGroup& group : series.groups()) {
        for (const double value : group.values()) {
          ASSERT_TRUE(std::isfinite(value)) << group.path() << " at " << time;
        }
      }
    }
  }
}

TEST_F(TelemetryBuilderTest, EveryFormulationGivesTheContractsGroups) {
  for (const std::pair<test::RobotFiles, test::Formulation>& configuration :
       {std::make_pair(test::atlasFiles(), test::Formulation::kCentroidalBasisVectors),
        std::make_pair(test::g1WholeBodyFiles(), test::Formulation::kWholeBody)}) {
    SCOPED_TRACE(configuration.first.taskFile);
    load(configuration.first, configuration.second);
    EXPECT_EQ(builder_->groupPaths(), contractPaths(config_.telemetryFrames));
    const humanoid_mpc_msgs::TelemetrySeries& series =
        builder_->build(decode(robot_->robotState(/*time=*/0.1, vector3_t(0.0, 0.0, 0.8), vector3_t::Zero())), &observation_, &policy_);
    vector_t state;
    vector_t input;
    samplePlan(policy_, /*time=*/0.1, &state, &input);
    EXPECT_TRUE(valuesOf(series, "contact_wrenches/right/mpc")
                    .isApprox(robot_->robotModel().getContactWrenchInWorldFrame(state, input, CONTACT_RIGHT_INDEX), kTolerance));
    for (const humanoid_mpc_msgs::ScalarGroup& group : series.groups()) {
      for (const double value : group.values()) {
        ASSERT_TRUE(std::isfinite(value)) << group.path();
      }
    }
  }
}

}  // namespace
}  // namespace ocs2::humanoid::visualization
