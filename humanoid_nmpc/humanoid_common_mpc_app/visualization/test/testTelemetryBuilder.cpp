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

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_mpc/CommandData.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc_app/visualization/PolicySnapshot.h"
#include "humanoid_common_mpc_app/visualization/RobotStateDecoder.h"
#include "humanoid_common_mpc_app/visualization/SolutionTermEvaluator.h"
#include "humanoid_common_mpc_app/visualization/TelemetryBuilder.h"
#include "humanoid_common_mpc_app/visualization/VisualizationConfig.h"
#include "humanoid_mpc_msgs/robot_state_sample.nproto.pb.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/visualization/test/VisualizationTestRobot.h"

namespace ocs2::humanoid::visualization {
namespace {

constexpr scalar_t kTolerance = 1.0e-9;

const humanoid_mpc_msgs::ScalarGroup* absl_nullable findGroup(
    const humanoid_mpc_msgs::TelemetrySeries& series, const std::string& path) {
  for (const humanoid_mpc_msgs::ScalarGroup& group : series.groups()) {
    if (group.path() == path) {
      return &group;
    }
  }
  return nullptr;
}

/** The value `name` of the group at `path`. */
double valueOf(const humanoid_mpc_msgs::TelemetrySeries& series,
               const std::string& path,
               const std::string& name) {
  const humanoid_mpc_msgs::ScalarGroup* absl_nullable group =
      findGroup(series, path);
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
vector_t valuesOf(const humanoid_mpc_msgs::TelemetrySeries& series,
                  const std::string& path) {
  const humanoid_mpc_msgs::ScalarGroup* absl_nullable group =
      findGroup(series, path);
  if (group == nullptr) {
    ADD_FAILURE() << "no group " << path;
    return vector_t();
  }
  return Eigen::Map<const vector_t>(group->values().data(),
                                    group->values_size());
}

/**
 * The display coordinates of `state`, [p_W, (yaw, pitch, roll), q_j], from the
 * robot model's accessors, independently of the builder's display buffer.
 */
vector_t displayCoordinatesOf(const MpcRobotModelBase<scalar_t>& model,
                              const vector_t& state) {
  const vector_t joints = model.getJointAngles(state);
  vector_t display(
      static_cast<Eigen::Index>(kBaseTranslationDim + kBaseRotationDim) +
      joints.size());
  display << model.getPosBaseRtWorldEwrtWorld(state),
      model.getEulerzyxBaseRtWorld(state), joints;
  return display;
}

/**
 * The paths of the telemetry contract
 * (humanoid_nmpc/humanoid_rerun_viewer/README.md) for `frames`, written out
 * here independently of the builder; test_end_to_end compares them with the
 * bridge's contract module itself.
 */
std::vector<std::string> contractPaths(const std::vector<std::string>& frames) {
  std::vector<std::string> paths = {"base_pose/position_x",
                                    "base_pose/position_y",
                                    "base_pose/position_z",
                                    "base_pose/roll",
                                    "base_pose/pitch",
                                    "base_pose/yaw",
                                    "base_pose/roll_rate",
                                    "base_pose/pitch_rate",
                                    "base_pose/yaw_rate",
                                    "base_pose/angular_velocity_x",
                                    "base_pose/angular_velocity_y",
                                    "base_pose/angular_velocity_z",
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
                                    "com/position_x",
                                    "com/position_y",
                                    "com/position_z",
                                    "dcm/x",
                                    "dcm/y",
                                    "zmp/x",
                                    "zmp/y",
                                    "acom/roll",
                                    "acom/pitch",
                                    "acom/yaw",
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
                                    "mpc_observation/mode",
                                    "mpc_solution/performance",
                                    "mpc_solution/costs",
                                    "mpc_solution/constraints"};
  for (const std::string& frame : frames) {
    for (const char* absl_nonnull kind :
         {"pose", "twist", "acceleration", "wrench"}) {
      for (const char* absl_nonnull source :
           {"measured", "reference", "plan"}) {
        paths.push_back(absl::StrCat("frames/", kind, "/", frame, "/", source));
      }
    }
  }
  return paths;
}

/** `proto` with each foot's force sensor carrying `normalForce` [N] straight
 * up. */
humanoid_mpc_msgs::RobotStateSample withFootForces(
    humanoid_mpc_msgs::RobotStateSample proto, scalar_t normalForce) {
  for (int contact = 0; contact < proto.meas_wrench_contacts_ewrt_world_size();
       ++contact) {
    proto.mutable_meas_wrench_contacts_ewrt_world(contact)
        ->mutable_force()
        ->set_z(normalForce);
  }
  return proto;
}

/** The groups that are NaN by contract in the tests' data: the reference's
 * target carries no contact forces, so its ZMP. */
bool isNotANumberByContract(const std::string& path, bool hasAcom) {
  return path == "zmp/x" || path == "zmp/y" ||
         (!hasAcom &&
          (path == "acom/roll" || path == "acom/pitch" || path == "acom/yaw"));
}

/** The net contact moment about `point` of world wrenches about `positions`. */
vector3_t momentAbout(const vector3_t& point,
                      const feet_array_t<vector3_t>& positions,
                      const feet_array_t<vector6_t>& wrenches) {
  vector3_t moment = vector3_t::Zero();
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    moment += (positions[contact] - point)
                  .cross(vector3_t(wrenches[contact].head<kForceDim>())) +
              vector3_t(wrenches[contact].tail<kTorqueDim>());
  }
  return moment;
}

class TelemetryBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    load(test::g1CentroidalFiles(), test::Formulation::kCentroidal);
  }

  void load(const test::RobotFiles& files, test::Formulation formulation) {
    robot_ = test::TestRobot::load(files, formulation);
    const absl::StatusOr<VisualizationConfig> config =
        loadVisualizationConfig(robot_->taskFile(), robot_->modelSettings());
    ASSERT_TRUE(config.ok()) << config.status();
    config_ = *config;
    absl::StatusOr<std::unique_ptr<TelemetryBuilder>> builder =
        TelemetryBuilder::Create(robot_->model(), config_);
    ASSERT_TRUE(builder.ok()) << builder.status();
    builder_ = std::move(*builder);
    decoder_ = std::make_unique<RobotStateDecoder>(robot_->robotModel());

    CommandData command;
    PrimalSolution solution;
    robot_->makePolicy(/*startTime=*/0.0, /*nodes=*/21, /*normalForce=*/300.0,
                       vector2_t(0.02, -0.01), &command, &solution);
    policy_.assign(command, solution);
    observation_ = robot_->observation(/*time=*/0.0, ModeNumber::kStance);
  }

  DecodedRobotState decode(const humanoid_mpc_msgs::RobotStateSample& proto) {
    msgs::RobotStateSample sample;
    EXPECT_TRUE(msgs::FromProto(proto, &sample).ok());
    DecodedRobotState state;
    EXPECT_TRUE(decoder_->decode(sample, &state).ok());
    return state;
  }

  /** Whether the robot has an ACoM network for the MPC model's joints. */
  bool hasAcom() const {
    return AngularCenterOfMass::Create(
               robot_->modelSettings().robotName,
               robot_->modelSettings().mpcModelJointNames)
        .ok();
  }

  /** [1/s] sqrt(g / z0) of the nominal state, from a model of its own: its CoM
   * above the mean of its contact frames. */
  scalar_t nominalPendulumFrequency() const {
    PinocchioInterface reference = robot_->makeReferencePinocchioInterface();
    const vector_t q =
        robot_->robotModel().getGeneralizedCoordinates(robot_->nominalState());
    const vector3_t com =
        pinocchio::centerOfMass(reference.getModel(), reference.getData(), q,
                                /*computeSubtreeComs=*/false);
    pinocchio::updateFramePlacements(reference.getModel(), reference.getData());
    scalar_t footHeight = 0.0;
    for (size_t contact = 0; contact < kNumContacts; ++contact) {
      footHeight += reference.getData()
                        .oMf[reference.getModel().getFrameId(
                            robot_->modelSettings().contactNames[contact])]
                        .translation()
                        .z() /
                    static_cast<scalar_t>(kNumContacts);
    }
    return std::sqrt(reference.getModel().gravity.linear().norm() /
                     (com.z() - footHeight));
  }

  /**
   * The plan's balance groups at `time` against a model of their own: the CoM
   * and its velocity of the policy's state and input, the DCM of the nominal
   * pendulum, the ZMP as the point about which the plan's contact moment is
   * vertical, and the ACoM attitude through the robot model's accessors.
   */
  void expectBalanceOfThePlan(const humanoid_mpc_msgs::TelemetrySeries& series,
                              scalar_t time) const {
    vector_t state;
    vector_t input;
    samplePlan(policy_, robot_->stateManifold(), time, &state, &input);
    const MpcRobotModelBase<scalar_t>& model = robot_->robotModel();
    PinocchioInterface reference = robot_->makeReferencePinocchioInterface();
    const vector_t q = model.getGeneralizedCoordinates(state);
    // getGeneralizedVelocities() is not const: a copy of the model of its own.
    const std::unique_ptr<MpcRobotModelBase<scalar_t>> velocityModel(
        model.clone());
    const vector_t v = velocityModel->getGeneralizedVelocities(state, input);
    pinocchio::centerOfMass(reference.getModel(), reference.getData(), q, v,
                            /*computeSubtreeComs=*/false);
    const vector3_t com = reference.getData().com[0];
    const vector3_t comVelocity = reference.getData().vcom[0];
    EXPECT_NEAR(valueOf(series, "com/position_x", "plan"), com.x(), kTolerance);
    EXPECT_NEAR(valueOf(series, "com/position_y", "plan"), com.y(), kTolerance);
    EXPECT_NEAR(valueOf(series, "com/position_z", "plan"), com.z(), kTolerance);
    const scalar_t omega = nominalPendulumFrequency();
    EXPECT_NEAR(valueOf(series, "dcm/x", "plan"),
                com.x() + comVelocity.x() / omega, 1.0e-8);
    EXPECT_NEAR(valueOf(series, "dcm/y", "plan"),
                com.y() + comVelocity.y() / omega, 1.0e-8);

    pinocchio::forwardKinematics(reference.getModel(), reference.getData(), q);
    pinocchio::updateFramePlacements(reference.getModel(), reference.getData());
    feet_array_t<vector3_t> positions;
    feet_array_t<vector6_t> wrenches;
    scalar_t normalForce = 0.0;
    scalar_t weightedHeight = 0.0;
    for (size_t contact = 0; contact < kNumContacts; ++contact) {
      positions[contact] =
          reference.getData()
              .oMf[reference.getModel().getFrameId(
                  robot_->modelSettings().contactNames[contact])]
              .translation();
      wrenches[contact] =
          model.getWrenchContactEwrtWorld(state, input, contact);
      normalForce += wrenches[contact][kWrenchForceZIndex];
      weightedHeight +=
          wrenches[contact][kWrenchForceZIndex] * positions[contact].z();
    }
    const vector3_t zmp(valueOf(series, "zmp/x", "plan"),
                        valueOf(series, "zmp/y", "plan"),
                        weightedHeight / normalForce);
    const vector3_t moment = momentAbout(zmp, positions, wrenches);
    EXPECT_NEAR(moment.x(), 0.0, 1.0e-7);
    EXPECT_NEAR(moment.y(), 0.0, 1.0e-7);

    if (!hasAcom()) {
      EXPECT_TRUE(std::isnan(valueOf(series, "acom/yaw", "plan")));
      return;
    }
    const std::unique_ptr<AngularCenterOfMass> acom =
        *AngularCenterOfMass::Create(
            robot_->modelSettings().robotName,
            robot_->modelSettings().mpcModelJointNames);
    const vector3_t yawPitchRoll = eulerZyxFromQuaternion(acom->acomAttitude(
        model.getQuatBaseRtWorldXyzw(state), model.getJointAngles(state)));
    EXPECT_NEAR(valueOf(series, "acom/yaw", "plan"),
                yawPitchRoll[kBaseRotYawIndex], kTolerance);
    EXPECT_NEAR(valueOf(series, "acom/pitch", "plan"),
                yawPitchRoll[kBaseRotPitchIndex], kTolerance);
    EXPECT_NEAR(valueOf(series, "acom/roll", "plan"),
                yawPitchRoll[kBaseRotRollIndex], kTolerance);
  }

  std::unique_ptr<test::TestRobot> robot_;
  VisualizationConfig config_;
  std::unique_ptr<TelemetryBuilder> builder_;
  std::unique_ptr<RobotStateDecoder> decoder_;
  PolicySnapshot policy_;
  SystemObservation observation_;
  /** The terms of a solution: a performance index only, as for a problem
   * without terms. */
  SolutionTerms solutionTerms_;
};

TEST_F(TelemetryBuilderTest, TheGroupsAreTheContractsInItsOrder) {
  EXPECT_EQ(builder_->groupPaths(), contractPaths(config_.telemetryFrames));
  // 42 panel groups, 24 complete groups of the robot and 12 per tracked frame.
  EXPECT_EQ(builder_->groupPaths().size(),
            42 + 24 + 12 * config_.telemetryFrames.size());
  const humanoid_mpc_msgs::TelemetrySeries& series =
      builder_->build(decode(robot_->robotState(/*time=*/0.1, vector3_t::Zero(),
                                                vector3_t::Zero())),
                      &observation_, &policy_);
  absl::flat_hash_set<std::string> paths;
  for (const humanoid_mpc_msgs::ScalarGroup& group : series.groups()) {
    EXPECT_TRUE(paths.insert(group.path()).second) << group.path();
    EXPECT_EQ(group.names_size(), group.values_size()) << group.path();
    // The term groups have a series per term of the solver's problem, which
    // this builder was not given.
    if (group.path() != "mpc_solution/costs" &&
        group.path() != "mpc_solution/constraints") {
      EXPECT_GT(group.names_size(), 0) << group.path();
    }
  }
  const std::vector<std::string>& joints =
      robot_->modelSettings().fullJointNames;
  const humanoid_mpc_msgs::ScalarGroup* absl_nullable jointPositions =
      findGroup(series, "joints/position/measured");
  ASSERT_NE(jointPositions, nullptr);
  EXPECT_EQ(std::vector<std::string>(jointPositions->names().begin(),
                                     jointPositions->names().end()),
            joints);
  // The coordinates are displayed with the base's Euler angles, the velocities
  // and forces as v: one entry fewer than q.
  const std::vector<std::string>& mpcJoints =
      robot_->modelSettings().mpcModelJointNames;
  const int displayDim = static_cast<int>(kBaseTranslationDim +
                                          kBaseRotationDim + mpcJoints.size());
  for (const char* absl_nonnull path :
       {"dofs/position/measured", "dofs/position/reference",
        "dofs/position/plan"}) {
    const humanoid_mpc_msgs::ScalarGroup* absl_nullable dofs =
        findGroup(series, path);
    ASSERT_NE(dofs, nullptr);
    ASSERT_EQ(dofs->names_size(), displayDim) << path;
    EXPECT_EQ(dofs->names(0), "base_x");
    EXPECT_EQ(dofs->names(3), "base_yaw");
    EXPECT_EQ(dofs->names(5), "base_roll");
    EXPECT_EQ(dofs->names(6), mpcJoints.front());
  }
  for (const char* absl_nonnull path :
       {"dofs/velocity/measured", "dofs/velocity/reference",
        "dofs/velocity/plan", "dofs/force/measured", "dofs/force/reference"}) {
    const humanoid_mpc_msgs::ScalarGroup* absl_nullable dofs =
        findGroup(series, path);
    ASSERT_NE(dofs, nullptr);
    ASSERT_EQ(dofs->names_size(),
              static_cast<int>(robot_->robotModel().getVelocityDim()))
        << path;
    for (size_t dof = 0; dof < kBaseVelocityDim; ++dof) {
      EXPECT_EQ(dofs->names(static_cast<int>(dof)),
                getBaseVelocityDofNames()[dof])
          << path;
    }
    EXPECT_EQ(dofs->names(6), mpcJoints.front());
  }
  // The MPC state stays generic: x0 ... x{nx - 1}, the quaternion's four
  // entries included.
  EXPECT_EQ(findGroup(series, "mpc_observation/state")->names(0), "x0");
  EXPECT_EQ(findGroup(series, "mpc_observation/state")->names_size(),
            static_cast<int>(robot_->robotModel().getStateDim()));
  EXPECT_EQ(findGroup(series, "mpc_observation/input")->names_size(),
            static_cast<int>(robot_->robotModel().getInputDim()));
}

TEST_F(TelemetryBuilderTest, TheBaseIsMeasuredInTheWorldFrameAsRollPitchYaw) {
  humanoid_mpc_msgs::RobotStateSample proto = robot_->robotState(
      /*time=*/0.1, vector3_t(1.0, 2.0, 3.0), vector3_t(0.1, 0.2, 0.3));
  proto.mutable_linvel_base_rt_world_ewrt_base()->set_x(1.0);
  proto.mutable_angvel_base_rt_world_ewrt_base()->set_z(0.5);
  const humanoid_mpc_msgs::TelemetrySeries& series = builder_->build(
      decode(proto), /*observation=*/nullptr, /*policy=*/nullptr);
  const matrix3_t rotation =
      test::quaternionFromRollPitchYaw(vector3_t(0.1, 0.2, 0.3))
          .toRotationMatrix();
  const vector3_t linear = rotation * vector3_t(1.0, 0.0, 0.0);
  const vector3_t angular = rotation * vector3_t(0.0, 0.0, 0.5);
  EXPECT_EQ(series.time(), 0.1);
  EXPECT_NEAR(valueOf(series, "base_pose/position_x", "measured"), 1.0,
              kTolerance);
  EXPECT_NEAR(valueOf(series, "base_pose/position_y", "measured"), 2.0,
              kTolerance);
  EXPECT_NEAR(valueOf(series, "base_pose/position_z", "measured"), 3.0,
              kTolerance);
  EXPECT_NEAR(valueOf(series, "base_pose/roll", "measured"), 0.1, kTolerance);
  EXPECT_NEAR(valueOf(series, "base_pose/pitch", "measured"), 0.2, kTolerance);
  EXPECT_NEAR(valueOf(series, "base_pose/yaw", "measured"), 0.3, kTolerance);
  EXPECT_NEAR(valueOf(series, "base_twist/linear_x", "measured"), linear.x(),
              kTolerance);
  EXPECT_NEAR(valueOf(series, "base_twist/linear_y", "measured"), linear.y(),
              kTolerance);
  EXPECT_NEAR(valueOf(series, "base_twist/linear_z", "measured"), linear.z(),
              kTolerance);
  EXPECT_NEAR(valueOf(series, "base_twist/angular_x", "measured"), angular.x(),
              kTolerance);
  EXPECT_NEAR(valueOf(series, "base_twist/angular_z", "measured"), angular.z(),
              kTolerance);
  // The body angular velocity as sent, and the Euler rates of the decoder,
  // which are the display's.
  const DecodedRobotState decoded = decode(proto);
  EXPECT_EQ(valueOf(series, "base_pose/angular_velocity_x", "measured"), 0.0);
  EXPECT_EQ(valueOf(series, "base_pose/angular_velocity_y", "measured"), 0.0);
  EXPECT_EQ(valueOf(series, "base_pose/angular_velocity_z", "measured"), 0.5);
  EXPECT_EQ(valueOf(series, "base_pose/yaw_rate", "measured"),
            decoded.baseEulerZyxRates[kBaseRotYawIndex]);
  EXPECT_EQ(valueOf(series, "base_pose/pitch_rate", "measured"),
            decoded.baseEulerZyxRates[kBaseRotPitchIndex]);
  EXPECT_EQ(valueOf(series, "base_pose/roll_rate", "measured"),
            decoded.baseEulerZyxRates[kBaseRotRollIndex]);
  EXPECT_NEAR(valueOf(series, "base_pose/yaw_rate", "measured"),
              0.5 * std::cos(0.1) / std::cos(0.2), kTolerance);
  // The generalized coordinates: z, then the Euler ZYX angles by their names;
  // the velocities: v_z and the Euler rates.
  EXPECT_NEAR(valueOf(series, "generalized_base/position_z", "measured"), 3.0,
              kTolerance);
  EXPECT_NEAR(valueOf(series, "generalized_base/pitch", "measured"), 0.2,
              kTolerance);
  EXPECT_NEAR(valueOf(series, "generalized_base/roll", "measured"), 0.1,
              kTolerance);
  EXPECT_NEAR(valueOf(series, "generalized_base/velocity_z", "measured"),
              linear.z(), kTolerance);
  EXPECT_EQ(valueOf(series, "generalized_base/pitch_rate", "measured"),
            valueOf(series, "base_pose/pitch_rate", "measured"));
  EXPECT_EQ(valueOf(series, "generalized_base/roll_rate", "measured"),
            valueOf(series, "base_pose/roll_rate", "measured"));
  EXPECT_NEAR(valueOf(series, "dofs/position/measured", "base_yaw"), 0.3,
              kTolerance);
  EXPECT_NEAR(valueOf(series, "dofs/position/measured", "base_roll"), 0.1,
              kTolerance);
  EXPECT_EQ(valuesOf(series, "dofs/velocity/measured"),
            decoded.generalizedVelocities);
  EXPECT_EQ(valueOf(series, "dofs/velocity/measured", "base_wz"), 0.5);
  const vector_t joints =
      valuesOf(series, "dofs/position/measured")
          .tail(static_cast<Eigen::Index>(robot_->robotModel().getJointDim()));
  EXPECT_EQ(joints, decoded.generalizedCoordinates.tail(joints.size()));
  // The root link's pose is the base's.
  const std::string root = "pelvis";
  ASSERT_NE(findGroup(series, "frames/pose/pelvis/measured"), nullptr);
  const vector_t pose = valuesOf(series, "frames/pose/" + root + "/measured");
  EXPECT_TRUE(pose.isApprox(
      (vector_t(6) << 1.0, 2.0, 3.0, 0.1, 0.2, 0.3).finished(), kTolerance))
      << pose.transpose();
  const vector_t twist = valuesOf(series, "frames/twist/" + root + "/measured");
  EXPECT_TRUE(twist.head<3>().isApprox(linear, kTolerance));
  EXPECT_TRUE(twist.tail<3>().isApprox(angular, kTolerance));
}

TEST_F(TelemetryBuilderTest,
       WithoutAPolicyTheReferenceAndThePlanAreTheMeasuredRobotAtRest) {
  humanoid_mpc_msgs::RobotStateSample proto = robot_->robotState(
      /*time=*/0.1, vector3_t(1.0, 2.0, 3.0), vector3_t(0.1, 0.2, 0.3));
  proto.mutable_linvel_base_rt_world_ewrt_base()->set_x(1.0);
  const humanoid_mpc_msgs::TelemetrySeries& series = builder_->build(
      decode(proto), /*observation=*/nullptr, /*policy=*/nullptr);
  EXPECT_EQ(valueOf(series, "base_pose/position_x", "reference"), 1.0);
  EXPECT_NEAR(valueOf(series, "base_pose/yaw", "reference"), 0.3, kTolerance);
  EXPECT_EQ(valueOf(series, "base_twist/linear_x", "reference"), 0.0);
  for (const char* absl_nonnull path :
       {"base_pose/roll_rate", "base_pose/yaw_rate",
        "base_pose/angular_velocity_z", "base_twist/angular_z"}) {
    EXPECT_EQ(valueOf(series, path, "reference"), 0.0) << path;
  }
  EXPECT_EQ(valuesOf(series, "dofs/position/reference"),
            valuesOf(series, "dofs/position/measured"));
  EXPECT_EQ(valuesOf(series, "dofs/position/plan"),
            valuesOf(series, "dofs/position/measured"));
  EXPECT_TRUE(valuesOf(series, "dofs/velocity/reference").isZero());
  EXPECT_TRUE(valuesOf(series, "dofs/velocity/plan").isZero());
  EXPECT_TRUE(valuesOf(series, "contact_wrenches/left/mpc").isZero());
  EXPECT_EQ(valueOf(series, "contact_forces/right_normal", "mpc"), 0.0);
  EXPECT_TRUE(valuesOf(series, "mpc_observation/state").isZero());
  EXPECT_TRUE(valuesOf(series, "mpc_observation/input").isZero());
}

TEST_F(TelemetryBuilderTest, TheReferenceIsTheTargetAtTheSamplesTime) {
  const humanoid_mpc_msgs::TelemetrySeries& series =
      builder_->build(decode(robot_->robotState(/*time=*/0.5, vector3_t::Zero(),
                                                vector3_t::Zero())),
                      &observation_, &policy_);
  // The target runs from the nominal state at 0 s to the plan's last state at 1
  // s, interpolated in the model's layout.
  const MpcRobotModelBase<scalar_t>& model = robot_->robotModel();
  const vector_t target = model.getDesiredState(policy_.target, /*time=*/0.5);
  EXPECT_TRUE(valuesOf(series, "dofs/position/reference")
                  .isApprox(displayCoordinatesOf(model, target), kTolerance));
  EXPECT_NEAR(valueOf(series, "base_pose/position_x", "reference"),
              model.getPosBaseRtWorldEwrtWorld(target).x(), kTolerance);
  EXPECT_NEAR(valueOf(series, "base_pose/yaw", "reference"),
              model.getEulerzyxBaseRtWorld(target)(0), kTolerance);
  EXPECT_NEAR(valueOf(series, "base_pose/roll", "reference"),
              model.getEulerzyxBaseRtWorld(target)(2), kTolerance);
  EXPECT_NEAR(valueOf(series, "base_twist/linear_x", "reference"),
              model.getBaseComLinearVelocity(target).x(), kTolerance);
  // The display coordinates of the reference: q with the Euler angles of its
  // quaternion.
  const vector_t q = model.getGeneralizedCoordinates(target);
  const vector_t display = valuesOf(series, "dofs/position/reference");
  EXPECT_TRUE(display.head<3>().isApprox(q.head<3>(), kTolerance));
  EXPECT_TRUE(display.segment<3>(3).isApprox(
      model.getEulerzyxBaseRtWorld(target), kTolerance));
  EXPECT_EQ(display.tail(display.size() - 6), q.tail(display.size() - 6));
}

// The reference's base rates: the body angular velocity w_B of the target
// trajectory (Pinocchio v[3:6]), its Euler rates for display and its
// world-frame angular velocity R(xi) w_B, on the whole-body MPC, whose state
// holds w_B.
TEST_F(TelemetryBuilderTest,
       TheReferencesAngularVelocityIsTheTargetsInEveryFrame) {
  load(test::g1WholeBodyFiles(), test::Formulation::kWholeBody);
  const MpcRobotModelBase<scalar_t>& model = robot_->robotModel();
  const vector3_t angularVelocityLocal(0.2, -0.1, 0.4);
  for (vector_t& state : policy_.target.stateTrajectory) {
    model.setEulerzyxBaseRtWorld(state, vector3_t(0.5, 0.3, -0.2));
    model.setAngvelBaseRtWorldEwrtBase(state, angularVelocityLocal);
  }
  const humanoid_mpc_msgs::TelemetrySeries& series =
      builder_->build(decode(robot_->robotState(/*time=*/0.5, vector3_t::Zero(),
                                                vector3_t::Zero())),
                      &observation_, &policy_);
  const vector_t target = model.getDesiredState(policy_.target, /*time=*/0.5);
  const vector3_t eulerZyx = model.getEulerzyxBaseRtWorld(target);
  const vector3_t eulerZyxRates =
      eulerZyxRatesForDisplay(eulerZyx, angularVelocityLocal);
  vector3_t angularVelocityWorld = vector3_t::Zero();
  {
    quaternion_t orientation;
    orientation.coeffs() = model.getQuatBaseRtWorldXyzw(target);
    angularVelocityWorld =
        orientation.toRotationMatrix() * angularVelocityLocal;
  }
  const std::array<const char* absl_nonnull, 3> axes = {"x", "y", "z"};
  for (size_t axis = 0; axis < 3; ++axis) {
    EXPECT_NEAR(
        valueOf(series, absl::StrCat("base_pose/angular_velocity_", axes[axis]),
                "reference"),
        angularVelocityLocal[axis], kTolerance);
    EXPECT_NEAR(valueOf(series, absl::StrCat("base_twist/angular_", axes[axis]),
                        "reference"),
                angularVelocityWorld[axis], kTolerance);
  }
  EXPECT_NEAR(valueOf(series, "base_pose/yaw_rate", "reference"),
              eulerZyxRates[kBaseRotYawIndex], kTolerance);
  EXPECT_NEAR(valueOf(series, "base_pose/pitch_rate", "reference"),
              eulerZyxRates[kBaseRotPitchIndex], kTolerance);
  EXPECT_NEAR(valueOf(series, "base_pose/roll_rate", "reference"),
              eulerZyxRates[kBaseRotRollIndex], kTolerance);
  EXPECT_EQ(valueOf(series, "generalized_base/pitch_rate", "reference"),
            valueOf(series, "base_pose/pitch_rate", "reference"));
  EXPECT_EQ(valueOf(series, "generalized_base/roll_rate", "reference"),
            valueOf(series, "base_pose/roll_rate", "reference"));
  EXPECT_NEAR(valueOf(series, "base_pose/roll", "reference"), -0.2, kTolerance);
  EXPECT_NEAR(valueOf(series, "dofs/velocity/reference", "base_wy"),
              angularVelocityLocal.y(), kTolerance);
}

// The same on the centroidal MPC, whose state holds the centroidal momentum:
// the reference's base twist is the mapping's A_b^-1 (m hbar - A_j qd_j) at the
// target's posture, so a target whose momentum is that of a turn about the
// world z axis at a yaw rate r, with the joints still, shows (0, 0, r) in the
// world frame and R' (0, 0, r) in the body frame. (The velocity-command knots
// write L = I_G R_z(dpsi) e_z r, the same turn up to the posture difference
// between the knot and the measured state the calculator's I_G is taken at.)
TEST_F(TelemetryBuilderTest,
       TheCentroidalReferencesAngularVelocityIsTheTurnItsMomentumCarries) {
  const MpcRobotModelBase<scalar_t>& model = robot_->robotModel();
  PinocchioInterface pinocchioInterface = robot_->pinocchioInterface();
  const scalar_t mass =
      pinocchio::computeTotalMass(pinocchioInterface.getModel());
  constexpr scalar_t kYawRate = 0.6;
  ASSERT_EQ(policy_.target.inputTrajectory.size(),
            policy_.target.stateTrajectory.size());
  const Eigen::Index joints = static_cast<Eigen::Index>(model.getJointDim());
  // One knot held over the target, so that the reference between the knots is
  // the knot itself: the momentum of a turn at one posture interpolated towards
  // another posture's is no posture's turn.
  vector_t knot = policy_.target.stateTrajectory.front();
  model.setEulerzyxBaseRtWorld(knot, vector3_t(0.5, 0.3, -0.2));
  quaternion_t knotOrientation;
  knotOrientation.coeffs() = model.getQuatBaseRtWorldXyzw(knot);
  vector_t velocity =
      vector_t::Zero(static_cast<Eigen::Index>(model.getVelocityDim()));
  velocity.segment<3>(3) = knotOrientation.toRotationMatrix().transpose() *
                           vector3_t(0.0, 0.0, kYawRate);
  const matrix_t momentumMatrix = pinocchio::computeCentroidalMap(
      pinocchioInterface.getModel(), pinocchioInterface.getData(),
      model.getGeneralizedCoordinates(knot));
  // The centroidal state opens with the normalized momentum [m c_dot; L_G] / m.
  knot.head<6>() = momentumMatrix * velocity / mass;
  for (size_t k = 0; k < policy_.target.stateTrajectory.size(); ++k) {
    policy_.target.stateTrajectory[k] = knot;
    policy_.target.inputTrajectory[k].tail(joints).setZero();
  }
  const humanoid_mpc_msgs::TelemetrySeries& series =
      builder_->build(decode(robot_->robotState(/*time=*/0.5, vector3_t::Zero(),
                                                vector3_t::Zero())),
                      &observation_, &policy_);
  const vector_t target = model.getDesiredState(policy_.target, /*time=*/0.5);
  quaternion_t orientation;
  orientation.coeffs() = model.getQuatBaseRtWorldXyzw(target);
  const vector3_t turnLocal = orientation.toRotationMatrix().transpose() *
                              vector3_t(0.0, 0.0, kYawRate);
  const std::array<const char* absl_nonnull, 3> axes = {"x", "y", "z"};
  for (size_t axis = 0; axis < 3; ++axis) {
    EXPECT_NEAR(valueOf(series, absl::StrCat("base_twist/angular_", axes[axis]),
                        "reference"),
                axis == 2 ? kYawRate : 0.0, 1.0e-9);
    EXPECT_NEAR(
        valueOf(series, absl::StrCat("base_pose/angular_velocity_", axes[axis]),
                "reference"),
        turnLocal[axis], 1.0e-9);
  }
}

// Near the pitch singularity the Euler rates are not a number, a gap in the
// plots, and every other value is finite.
TEST_F(TelemetryBuilderTest,
       AtThePitchSingularityOnlyTheEulerRatesAreNotANumber) {
  humanoid_mpc_msgs::RobotStateSample proto = robot_->robotState(
      /*time=*/0.1, vector3_t(0.0, 0.0, 0.7), vector3_t(0.2, M_PI / 2.0, 0.3));
  proto.mutable_angvel_base_rt_world_ewrt_base()->set_y(0.4);
  const humanoid_mpc_msgs::TelemetrySeries& series =
      builder_->build(decode(withFootForces(proto, /*normalForce=*/300.0)),
                      &observation_, &policy_, &solutionTerms_);
  for (const humanoid_mpc_msgs::ScalarGroup& group : series.groups()) {
    if (isNotANumberByContract(group.path(), hasAcom())) continue;
    for (int index = 0; index < group.values_size(); ++index) {
      const bool eulerRate = group.path() == "base_pose/roll_rate" ||
                             group.path() == "base_pose/pitch_rate" ||
                             group.path() == "base_pose/yaw_rate" ||
                             group.path() == "generalized_base/pitch_rate" ||
                             group.path() == "generalized_base/roll_rate";
      if (eulerRate && group.names(index) == "measured") {
        EXPECT_TRUE(std::isnan(group.values(index))) << group.path();
      } else {
        EXPECT_TRUE(std::isfinite(group.values(index)))
            << group.path() << " " << group.names(index);
      }
    }
  }
  EXPECT_NEAR(valueOf(series, "base_pose/pitch", "measured"), M_PI / 2.0,
              1.0e-9);
  EXPECT_EQ(valueOf(series, "base_pose/angular_velocity_y", "measured"), 0.4);
}

TEST_F(TelemetryBuilderTest, ThePlanIsThePolicyAtTheSamplesTime) {
  const scalar_t time = 0.1;
  const humanoid_mpc_msgs::TelemetrySeries& series = builder_->build(
      decode(robot_->robotState(time, vector3_t::Zero(), vector3_t::Zero())),
      &observation_, &policy_);
  vector_t state;
  vector_t input;
  samplePlan(policy_, robot_->stateManifold(), time, &state, &input);
  const MpcRobotModelBase<scalar_t>& model = robot_->robotModel();
  EXPECT_TRUE(valuesOf(series, "dofs/position/plan")
                  .isApprox(displayCoordinatesOf(model, state), kTolerance));
  const vector6_t left =
      model.getWrenchContactEwrtWorld(state, input, kContactLeftIndex);
  EXPECT_TRUE(
      valuesOf(series, "contact_wrenches/left/mpc").isApprox(left, kTolerance));
  // Written as 300 N up and 15 N forward on each stance foot (world-frame
  // wrench inputs).
  EXPECT_NEAR(valueOf(series, "contact_forces/left_normal", "mpc"), 300.0,
              1.0e-9);
  EXPECT_NEAR(valueOf(series, "contact_forces/left_tangential", "mpc_x"), 15.0,
              1.0e-9);
  EXPECT_TRUE(valuesOf(series, "frames/wrench/foot_l_contact/plan")
                  .isApprox(left, kTolerance));
  EXPECT_TRUE(valuesOf(series, "frames/wrench/pelvis/plan").isZero());
}

TEST_F(TelemetryBuilderTest, TheMeasuredWrenchesAreTheSensors) {
  humanoid_mpc_msgs::RobotStateSample proto =
      robot_->robotState(/*time=*/0.1, vector3_t::Zero(), vector3_t::Zero());
  humanoid_mpc_msgs::Wrench* absl_nonnull left =
      proto.mutable_meas_wrench_contacts_ewrt_world(0);
  left->mutable_force()->set_x(5.0);
  left->mutable_force()->set_y(-6.0);
  left->mutable_force()->set_z(310.0);
  left->mutable_torque()->set_y(2.0);
  const humanoid_mpc_msgs::TelemetrySeries& series =
      builder_->build(decode(proto), &observation_, &policy_);
  EXPECT_EQ(valuesOf(series, "contact_wrenches/left/measured"),
            (vector_t(3) << 5.0, -6.0, 310.0).finished());
  EXPECT_TRUE(valuesOf(series, "contact_wrenches/right/measured").isZero());
  EXPECT_EQ(valueOf(series, "contact_forces/left_normal", "measured"), 310.0);
  EXPECT_EQ(valueOf(series, "contact_forces/left_tangential", "measured_x"),
            5.0);
  EXPECT_EQ(valueOf(series, "contact_forces/left_tangential", "measured_y"),
            -6.0);
  EXPECT_EQ(valuesOf(series, "frames/wrench/foot_l_contact/measured"),
            (vector_t(6) << 5.0, -6.0, 310.0, 0.0, 2.0, 0.0).finished());
  EXPECT_TRUE(valuesOf(series, "frames/wrench/pelvis/measured").isZero());
}

TEST_F(TelemetryBuilderTest,
       TheMeasuredAccelerationIsTheDifferenceOfConsecutiveSamples) {
  builder_->build(decode(robot_->robotState(/*time=*/1.0, vector3_t::Zero(),
                                            vector3_t::Zero())),
                  /*observation=*/nullptr,
                  /*policy=*/nullptr);
  humanoid_mpc_msgs::RobotStateSample proto =
      robot_->robotState(/*time=*/1.01, vector3_t::Zero(), vector3_t::Zero());
  proto.mutable_linvel_base_rt_world_ewrt_base()->set_z(0.1);
  const humanoid_mpc_msgs::TelemetrySeries& series = builder_->build(
      decode(proto), /*observation=*/nullptr, /*policy=*/nullptr);
  // 0.1 m/s more in 10 ms, at the root link, which does not rotate.
  EXPECT_NEAR(
      valueOf(series, "frames/acceleration/pelvis/measured", "linear_z"), 10.0,
      1.0e-6);
  EXPECT_NEAR(
      valueOf(series, "frames/acceleration/pelvis/measured", "linear_x"), 0.0,
      1.0e-9);

  // The robot's clock went back (a simulation restarted): no acceleration from
  // a sample of the previous run.
  proto.set_time(0.5);
  proto.mutable_linvel_base_rt_world_ewrt_base()->set_z(-3.0);
  const humanoid_mpc_msgs::TelemetrySeries& rewound = builder_->build(
      decode(proto), /*observation=*/nullptr, /*policy=*/nullptr);
  EXPECT_EQ(valueOf(rewound, "frames/acceleration/pelvis/measured", "linear_z"),
            0.0);
  builder_->reset();
  proto.set_time(0.51);
  const humanoid_mpc_msgs::TelemetrySeries& afterReset = builder_->build(
      decode(proto), /*observation=*/nullptr, /*policy=*/nullptr);
  EXPECT_EQ(
      valueOf(afterReset, "frames/acceleration/pelvis/measured", "linear_z"),
      0.0);
}

TEST_F(TelemetryBuilderTest,
       TheObservationGroupsAreTheObservationAndThePolicysInputThere) {
  observation_.time = 0.2;
  observation_.mode = ModeNumber::kLf;
  const humanoid_mpc_msgs::TelemetrySeries& series =
      builder_->build(decode(robot_->robotState(
                          /*time=*/0.25, vector3_t::Zero(), vector3_t::Zero())),
                      &observation_, &policy_);
  vector_t state;
  vector_t input;
  samplePlan(policy_, robot_->stateManifold(), observation_.time, &state,
             &input);
  EXPECT_EQ(valuesOf(series, "mpc_observation/state"), observation_.state);
  EXPECT_TRUE(
      valuesOf(series, "mpc_observation/input").isApprox(input, kTolerance));
  EXPECT_EQ(valueOf(series, "mpc_observation/mode", "mode"),
            static_cast<double>(ModeNumber::kLf));
}

TEST_F(TelemetryBuilderTest, TheJointGroupsAreTheDecodedJoints) {
  humanoid_mpc_msgs::RobotStateSample proto =
      robot_->robotState(/*time=*/0.1, vector3_t::Zero(), vector3_t::Zero());
  for (int joint = 0; joint < proto.joint_names_size(); ++joint) {
    proto.add_joint_position_targets(0.3);
    proto.add_joint_velocity_targets(0.0);
    proto.add_joint_kp(50.0);
    proto.add_joint_kd(1.0);
    proto.add_joint_feed_forward_efforts(2.0);
  }
  const DecodedRobotState state = decode(proto);
  const humanoid_mpc_msgs::TelemetrySeries& series =
      builder_->build(state, &observation_, &policy_);
  EXPECT_EQ(valuesOf(series, "joints/position/measured"), state.jointPositions);
  EXPECT_EQ(valuesOf(series, "joints/position/target"),
            state.jointPositionTargets);
  EXPECT_EQ(valuesOf(series, "joints/velocity/measured"),
            state.jointVelocities);
  EXPECT_EQ(valuesOf(series, "joints/effort/measured"), state.jointEfforts);
  EXPECT_EQ(valuesOf(series, "joints/effort/target"),
            state.jointFeedForwardEfforts);
  EXPECT_EQ(valuesOf(series, "dofs/force/measured"), state.generalizedForces);
  // The forces are dual to the velocity v: the base's entries, then the
  // joints'.
  const vector_t referenceForces = valuesOf(series, "dofs/force/reference");
  const Eigen::Index baseVelocityDim =
      static_cast<Eigen::Index>(kBaseVelocityDim);
  EXPECT_TRUE(referenceForces.head(baseVelocityDim).isZero());
  EXPECT_TRUE(
      (referenceForces.tail(referenceForces.size() - baseVelocityDim).array() ==
       2.0)
          .all());
}

TEST_F(TelemetryBuilderTest, TheBalanceOfThePlanIsItsComDcmZmpAndAcom) {
  for (const scalar_t time :
       {0.1, 0.45}) {  // in double stance, and with the left foot in swing
    SCOPED_TRACE(time);
    expectBalanceOfThePlan(
        builder_->build(decode(robot_->robotState(time, vector3_t::Zero(),
                                                  vector3_t::Zero())),
                        &observation_, &policy_),
        time);
  }
}

TEST_F(TelemetryBuilderTest,
       TheMeasuredBalanceIsTheSamplesComAndTheSensorsZmp) {
  const DecodedRobotState measured = decode(withFootForces(
      robot_->robotState(/*time=*/0.1, vector3_t(0.2, -0.1, 0.75),
                         vector3_t(0.05, -0.02, 0.3)),
      /*normalForce=*/250.0));
  const humanoid_mpc_msgs::TelemetrySeries& series =
      builder_->build(measured, &observation_, &policy_);
  PinocchioInterface reference = robot_->makeReferencePinocchioInterface();
  pinocchio::centerOfMass(reference.getModel(), reference.getData(),
                          measured.generalizedCoordinates,
                          measured.generalizedVelocities,
                          /*computeSubtreeComs=*/false);
  const vector3_t com = reference.getData().com[0];
  const vector3_t comVelocity = reference.getData().vcom[0];
  // The joints move at 0.1 rad/s, so the CoM moves and the DCM is not the CoM.
  ASSERT_GT(comVelocity.head<kHorizontalPositionDim>().norm(), 1.0e-4);
  EXPECT_NEAR(valueOf(series, "com/position_z", "measured"), com.z(),
              kTolerance);
  const scalar_t omega = nominalPendulumFrequency();
  EXPECT_NEAR(valueOf(series, "dcm/x", "measured"),
              com.x() + comVelocity.x() / omega, 1.0e-8);
  EXPECT_NEAR(valueOf(series, "dcm/y", "measured"),
              com.y() + comVelocity.y() / omega, 1.0e-8);
  // Equal forces without torques: the ZMP is the midpoint of the measured
  // contact frames.
  pinocchio::forwardKinematics(reference.getModel(), reference.getData(),
                               measured.generalizedCoordinates);
  pinocchio::updateFramePlacements(reference.getModel(), reference.getData());
  vector3_t midpoint = vector3_t::Zero();
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    midpoint += reference.getData()
                    .oMf[reference.getModel().getFrameId(
                        robot_->modelSettings().contactNames[contact])]
                    .translation() /
                static_cast<scalar_t>(kNumContacts);
  }
  EXPECT_NEAR(valueOf(series, "zmp/x", "measured"), midpoint.x(), kTolerance);
  EXPECT_NEAR(valueOf(series, "zmp/y", "measured"), midpoint.y(), kTolerance);
  // Without force on the sensors there is no measured ZMP.
  const humanoid_mpc_msgs::TelemetrySeries& unloaded =
      builder_->build(decode(robot_->robotState(/*time=*/0.2, vector3_t::Zero(),
                                                vector3_t::Zero())),
                      &observation_, &policy_);
  EXPECT_TRUE(std::isnan(valueOf(unloaded, "zmp/x", "measured")));
}

// A rotation of the whole robot about the vertical turns the ACoM with it: its
// yaw by the same angle, its tilt not at all (AngularCenterOfMass: xi_a(q (x)
// xi, q_j) = q (x) xi_a(xi, q_j)).
TEST_F(TelemetryBuilderTest, TheMeasuredAcomTurnsWithTheRobot) {
  ASSERT_TRUE(hasAcom()) << "the G1 has an ACoM network";
  constexpr scalar_t kTurn = 0.4;
  const vector3_t tilt(0.05, -0.08, 0.1);
  const humanoid_mpc_msgs::TelemetrySeries first = builder_->build(
      decode(robot_->robotState(/*time=*/0.1, vector3_t::Zero(), tilt)),
      &observation_, &policy_);
  const humanoid_mpc_msgs::TelemetrySeries turned = builder_->build(
      decode(robot_->robotState(/*time=*/0.2, vector3_t::Zero(),
                                tilt + vector3_t(0.0, 0.0, kTurn))),
      &observation_, &policy_);
  EXPECT_NEAR(std::remainder(valueOf(turned, "acom/yaw", "measured") -
                                 valueOf(first, "acom/yaw", "measured") - kTurn,
                             2.0 * M_PI),
              0.0, 1.0e-9);
  EXPECT_NEAR(valueOf(turned, "acom/pitch", "measured"),
              valueOf(first, "acom/pitch", "measured"), 1.0e-9);
  EXPECT_NEAR(valueOf(turned, "acom/roll", "measured"),
              valueOf(first, "acom/roll", "measured"), 1.0e-9);
  // The joints bend the ACoM away from the base: it is not the base's attitude.
  EXPECT_GT(std::abs(valueOf(first, "acom/pitch", "measured") -
                     valueOf(first, "base_pose/pitch", "measured")) +
                std::abs(valueOf(first, "acom/roll", "measured") -
                         valueOf(first, "base_pose/roll", "measured")),
            1.0e-4);
}

TEST_F(TelemetryBuilderTest, WithoutANominalStateTheDcmIsNotANumber) {
  VisualizationModel model = robot_->model();
  model.nominalState = vector_t();
  std::unique_ptr<TelemetryBuilder> builder =
      *TelemetryBuilder::Create(model, config_);
  const humanoid_mpc_msgs::TelemetrySeries& series =
      builder->build(decode(robot_->robotState(/*time=*/0.1, vector3_t::Zero(),
                                               vector3_t::Zero())),
                     &observation_, &policy_);
  EXPECT_TRUE(std::isnan(valueOf(series, "dcm/x", "plan")));
  EXPECT_TRUE(std::isfinite(valueOf(series, "com/position_x", "plan")));
}

TEST_F(TelemetryBuilderTest,
       TheSolutionGroupsAreTheLatestTermsAndNotANumberWithout) {
  const TelemetryBuilder::SolutionTermNames names{
      .costs = {"intermediate/tracking", "final/terminal"},
      .constraints = {"equality/eq"}};
  std::unique_ptr<TelemetryBuilder> builder =
      *TelemetryBuilder::Create(robot_->model(), config_, names);
  const DecodedRobotState measured = decode(
      robot_->robotState(/*time=*/0.1, vector3_t::Zero(), vector3_t::Zero()));
  const humanoid_mpc_msgs::TelemetrySeries& before =
      builder->build(measured, &observation_, &policy_);
  EXPECT_TRUE(std::isnan(valueOf(before, "mpc_solution/performance", "merit")));
  EXPECT_TRUE(
      std::isnan(valueOf(before, "mpc_solution/costs", "final/terminal")));
  EXPECT_TRUE(
      std::isnan(valueOf(before, "mpc_solution/constraints", "equality/eq")));

  SolutionTerms terms;
  terms.performance.merit = 4.0;
  terms.performance.inequalityConstraintsSSE = 0.25;
  terms.costs = (vector_t(2) << 1.5, 2.5).finished();
  terms.constraintViolations = (vector_t(1) << 0.125).finished();
  const humanoid_mpc_msgs::TelemetrySeries& after =
      builder->build(measured, &observation_, &policy_, &terms);
  EXPECT_EQ(valueOf(after, "mpc_solution/performance", "merit"), 4.0);
  EXPECT_EQ(
      valueOf(after, "mpc_solution/performance", "inequality_constraints_sse"),
      0.25);
  EXPECT_EQ(valueOf(after, "mpc_solution/costs", "intermediate/tracking"), 1.5);
  EXPECT_EQ(valueOf(after, "mpc_solution/costs", "final/terminal"), 2.5);
  EXPECT_EQ(valueOf(after, "mpc_solution/constraints", "equality/eq"), 0.125);

  // Terms of another layout are not plotted against the wrong names.
  terms.costs = (vector_t(3) << 1.0, 2.0, 3.0).finished();
  const humanoid_mpc_msgs::TelemetrySeries& mismatched =
      builder->build(measured, &observation_, &policy_, &terms);
  EXPECT_TRUE(std::isnan(
      valueOf(mismatched, "mpc_solution/costs", "intermediate/tracking")));
  EXPECT_EQ(valueOf(mismatched, "mpc_solution/constraints", "equality/eq"),
            0.125);
}

TEST_F(TelemetryBuilderTest, EveryValueIsFinite) {
  for (const PolicySnapshot* absl_nullable policy :
       {static_cast<const PolicySnapshot*>(nullptr),
        static_cast<const PolicySnapshot*>(&policy_)}) {
    for (const scalar_t time : {-1.0, 0.0, 0.3, 0.6, 2.0}) {
      const humanoid_mpc_msgs::TelemetrySeries& series =
          builder_->build(decode(withFootForces(
                              robot_->robotState(time, vector3_t(0.0, 0.0, 0.7),
                                                 vector3_t(0.0, 0.1, 0.0)),
                              /*normalForce=*/300.0)),
                          &observation_, policy, &solutionTerms_);
      for (const humanoid_mpc_msgs::ScalarGroup& group : series.groups()) {
        if (isNotANumberByContract(group.path(), hasAcom())) continue;
        for (const double value : group.values()) {
          ASSERT_TRUE(std::isfinite(value)) << group.path() << " at " << time;
        }
      }
      // The measured ZMP (the sensors carry 300 N each) and the plan's (a
      // stance foot carries 300 N at every time).
      EXPECT_TRUE(std::isfinite(valueOf(series, "zmp/x", "measured")));
      if (policy != nullptr)
        EXPECT_TRUE(std::isfinite(valueOf(series, "zmp/y", "plan")));
    }
  }
}

TEST_F(TelemetryBuilderTest, EveryFormulationGivesTheContractsGroups) {
  for (const std::pair<test::RobotFiles, test::Formulation>& configuration :
       {std::make_pair(test::atlasFiles(),
                       test::Formulation::kCentroidalBasisVectors),
        std::make_pair(test::g1WholeBodyFiles(),
                       test::Formulation::kWholeBody)}) {
    SCOPED_TRACE(configuration.first.taskFile);
    load(configuration.first, configuration.second);
    EXPECT_EQ(builder_->groupPaths(), contractPaths(config_.telemetryFrames));
    const humanoid_mpc_msgs::TelemetrySeries& series = builder_->build(
        decode(withFootForces(
            robot_->robotState(/*time=*/0.1, vector3_t(0.0, 0.0, 0.8),
                               vector3_t::Zero()),
            /*normalForce=*/300.0)),
        &observation_, &policy_, &solutionTerms_);
    vector_t state;
    vector_t input;
    samplePlan(policy_, robot_->stateManifold(), /*time=*/0.1, &state, &input);
    EXPECT_TRUE(valuesOf(series, "contact_wrenches/right/mpc")
                    .isApprox(robot_->robotModel().getWrenchContactEwrtWorld(
                                  state, input, kContactRightIndex),
                              kTolerance));
    expectBalanceOfThePlan(series, /*time=*/0.1);
    for (const humanoid_mpc_msgs::ScalarGroup& group : series.groups()) {
      if (isNotANumberByContract(group.path(), hasAcom())) continue;
      for (const double value : group.values()) {
        ASSERT_TRUE(std::isfinite(value)) << group.path();
      }
    }
  }
}

}  // namespace
}  // namespace ocs2::humanoid::visualization
