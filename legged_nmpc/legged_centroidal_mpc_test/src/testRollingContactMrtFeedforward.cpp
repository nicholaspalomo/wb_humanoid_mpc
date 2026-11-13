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

// The feedforward of the centroidal MRT controller on the rolling contacts of
// the LimX Tron1 (legged_nmpc/docs/rolling_contact/README.md): a planned
// wheel force acts at the wheel's lowest point, so the controller applies it at
// the wheel-center frame with the moment (-r n) x f, and the wheel joint
// carries r times its traction.

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/control/ManifoldLinearController.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/multibody/fwd.hpp"

#include "legged_centroidal_mpc/CentroidalMpcInterface.h"
#include "legged_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h"
#include "legged_common_mpc/common/ContactArray.h"
#include "legged_common_mpc/common/ContactMask.h"
#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/MpcRobotModelBase.h"
#include "legged_common_mpc/common/StateManifoldFactory.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/config/ConfigFiles.h"
#include "legged_common_mpc/contact/RollingContact.h"
#include "legged_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "legged_nmpc/legged_common_mpc/test/ScriptedPolicyMpcLink.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"
#include "support/TypedConfigFiles.h"

namespace ocs2::legged {
namespace {

constexpr int kCycles = 200;  // past the entry ramp of WB_MPC
constexpr scalar_t kTolerance = 1.0e-8;
// [N] The planned traction of each wheel along world x, on top of its share
// of the weight: forward on the left wheel, backward on the right one.
constexpr scalar_t kTractionLeft = 8.0;
constexpr scalar_t kTractionRight = -5.0;

/** Runs `count` control cycles in `mode`, 10 ms apart from `time`, which it
 * advances. */
void runCycles(CentroidalMpcMrtJointController& controller,
               robot::model::RobotState& state,
               robot::model::RobotJointAction& action,
               scalar_t& time,
               const char* absl_nonnull mode,
               int count) {
  for (int cycle = 0; cycle < count; ++cycle) {
    controller.setControlMode(mode);
    time += 0.01;
    state.setTime(time);
    controller.computeJointControlAction(time, state, action);
  }
}

/** Where a wheel force of wrenchWheelsEwrtWorld() acts. */
enum class WheelForcePoint {
  // The wheel's lowest point, the rolling contact point.
  kContactPoint,
  // The wheel center, as a fixed point contact would apply it.
  kWheelCenter,
};

/** Returns the world-frame wrench at each wheel-center frame of the forces
 * `forceContactsEwrtWorld` acting at `point`: with the moment (-r n) x f at the
 * contact point, with a zero moment at the wheel center. */
ContactArray<vector6_t> wrenchWheelsEwrtWorld(
    const ModelSettings& settings,
    const ContactArray<vector3_t>& forceContactsEwrtWorld,
    WheelForcePoint point) {
  ContactArray<vector6_t> wrenchContactsEwrtWorld(forceContactsEwrtWorld.size(),
                                                  vector6_t::Zero());
  for (size_t i = 0; i < forceContactsEwrtWorld.size(); ++i) {
    wrenchContactsEwrtWorld[i].head<kForceDim>() = forceContactsEwrtWorld[i];
    if (point == WheelForcePoint::kContactPoint) {
      wrenchContactsEwrtWorld[i].tail<kTorqueDim>() = torqueWheelEwrtWorld(
          forceContactsEwrtWorld[i], settings.wheelRadius(i),
          normalTerrainEwrtWorld<scalar_t>());
    }
  }
  return wrenchContactsEwrtWorld;
}

TEST(RollingContactMrtFeedforward,
     TheWheelForcesActAtTheContactPointsAndTheWheelsCarryTheirTraction) {
  const CentroidalRobotFiles files = tron1Files();
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interfaceOrStatus =
      CentroidalMpcInterface::CreateControllerModels(
          files.taskFile, files.urdfFile, files.referenceFile);
  ASSERT_TRUE(interfaceOrStatus.ok()) << interfaceOrStatus.status();
  CentroidalMpcInterface& interface = **interfaceOrStatus;
  const ModelSettings& settings = interface.modelSettings();
  const size_t numContacts = settings.numContacts();
  ASSERT_EQ(numContacts, 2u);
  ASSERT_TRUE(settings.isRollingContact(0));
  ASSERT_TRUE(settings.isRollingContact(1));

  absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus =
      robot::model::RobotDescription::Create(files.urdfFile);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  test_support::ScriptedPolicyMpcLink* absl_nullable link = nullptr;
  absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>>
      controllerOrStatus = CentroidalMpcMrtJointController::Create(
          description, settings, interface.getMpcRobotModel(),
          test_support::ScriptedPolicyMpcLink::factory(&link),
          interface.getPinocchioInterface(),
          jointPdGainsFileBeside(files.taskFile),
          &interface.getEffectiveMpcRobotModel());
  ASSERT_TRUE(controllerOrStatus.ok()) << controllerOrStatus.status();
  ASSERT_NE(link, nullptr);
  CentroidalMpcMrtJointController& controller = **controllerOrStatus;

  // Tron1 standing on both wheels at its default posture, at rest.
  robot::model::RobotState state(description, numContacts);
  state.setPosBaseRtWorldEwrtWorld(vector3_t(0.0, 0.0, 0.7485));
  const std::vector<std::pair<std::string, scalar_t>> posture = {
      {"hip_L_Joint", 0.35},
      {"hip_R_Joint", -0.35},
      {"knee_L_Joint", 0.8},
      {"knee_R_Joint", -0.8}};
  for (const std::pair<std::string, scalar_t>& joint : posture) {
    const absl::StatusOr<std::vector<size_t>> index =
        description.findJointIndices(
            absl::MakeConstSpan(&joint.first, /*size=*/1));
    ASSERT_TRUE(index.ok()) << index.status();
    state.setJointPosition(index->front(), joint.second);
  }
  for (size_t contact = 0; contact < numContacts; ++contact) {
    state.setContactFlag(contact, /*contactFlag=*/true);
  }
  robot::model::RobotJointAction action(description);
  scalar_t time = 0.0;
  runCycles(controller, state, action, time, "JOINT_PD", /*count=*/20);
  runCycles(controller, state, action, time, "WB_MPC", /*count=*/20);

  // A standing policy whose input is the weight shared by the wheels plus a
  // traction along x on each, without feedback.
  const MpcRobotModelBase<scalar_t>& effective =
      interface.getEffectiveMpcRobotModel();
  const std::shared_ptr<const StateManifold> manifold =
      stateManifoldOf(interface.getMpcRobotModel()).value();
  const Eigen::Index inputDim =
      static_cast<Eigen::Index>(effective.getInputDim());
  const Eigen::Index tangentDim =
      static_cast<Eigen::Index>(manifold->getTangentDim());
  const vector_t start = link->lastObservation().state;
  PinocchioInterface reference = interface.getPinocchioInterface();
  const ContactMask stance = ContactMask::allActive(numContacts);
  vector_t input = weightCompensatingInput(reference, stance, effective, start);
  ContactArray<vector3_t> forceContactsEwrtWorld(numContacts,
                                                 vector3_t::Zero());
  ContactArray<vector3_t> initForceContactsEwrtWorld(numContacts,
                                                     vector3_t::Zero());
  for (size_t i = 0; i < numContacts; ++i) {
    initForceContactsEwrtWorld[i] =
        effective.getForceContactEwrtParam(input, i);
    forceContactsEwrtWorld[i] = initForceContactsEwrtWorld[i];
    forceContactsEwrtWorld[i].x() += i == 0 ? kTractionLeft : kTractionRight;
    effective.setForceContactEwrtParam(input, forceContactsEwrtWorld[i], i);
  }
  PrimalSolution policy;
  policy.timeTrajectory_ = {time, time + 10.0};
  policy.stateTrajectory_ = {start, start};
  policy.inputTrajectory_ = {input, input};
  policy.modeSchedule_ =
      ModeSchedule(/*eventTimesInput=*/{},
                   /*modeSequenceInput=*/{stance.modeNumber()});
  policy.controllerPtr_ = std::make_unique<ManifoldLinearController>(
      policy.timeTrajectory_, policy.stateTrajectory_, policy.inputTrajectory_,
      matrix_array_t{matrix_t::Zero(inputDim, tangentDim),
                     matrix_t::Zero(inputDim, tangentDim)},
      manifold);
  link->deliver(policy);
  runCycles(controller, state, action, time, "WB_MPC", kCycles);
  ASSERT_FALSE(controller.isEnteringMpc()) << "the policy is executed";

  // The feedforward is the free-base inverse dynamics of the forces moved to
  // the wheel-center frames, at the measured state the observation carries.
  const SystemObservation& observation = link->lastObservation();
  // A copy, as getGeneralizedVelocities() updates the model's own workspace.
  const std::unique_ptr<MpcRobotModelBase<scalar_t>> effectiveCopy(
      effective.clone());
  const vector_t q =
      effectiveCopy->getGeneralizedCoordinates(observation.state);
  const vector_t v = effectiveCopy->getGeneralizedVelocities(observation.state,
                                                             observation.input);
  const ContactArray<pinocchio::FrameIndex> frames =
      getContactFrameIndexArray<scalar_t>(reference,
                                          interface.getMpcRobotModel());
  const vector_t zeroJointAccelerations =
      vector_t::Zero(effective.getJointDim());
  const vector_t expected = computeJointTorques<scalar_t>(
      q, v, zeroJointAccelerations,
      wrenchWheelsEwrtWorld(settings, forceContactsEwrtWorld,
                            WheelForcePoint::kContactPoint),
      frames, reference);
  const vector_t feedforwardAtWheelCenters = computeJointTorques<scalar_t>(
      q, v, zeroJointAccelerations,
      wrenchWheelsEwrtWorld(settings, forceContactsEwrtWorld,
                            WheelForcePoint::kWheelCenter),
      frames, reference);
  const std::vector<size_t> mpcJoints =
      description.getJointIndices(settings.mpcModelJointNames);
  for (size_t i = 0; i < mpcJoints.size(); ++i) {
    const scalar_t limit = controller.getPdGains().mpcJointTorqueLimit[i];
    const std::optional<robot::model::JointAction>& jointAction =
        action[mpcJoints[i]];
    if (!jointAction.has_value()) {
      GTEST_FAIL() << settings.mpcModelJointNames[i];
    }
    EXPECT_NEAR(
        jointAction->feed_forward_effort,
        std::clamp(expected[static_cast<Eigen::Index>(i)], -limit, limit),
        kTolerance)
        << settings.mpcModelJointNames[i];
  }

  // The moment is what turns the wheels: on the wheels' own rows of the
  // generalized force, the contact wrench at the wheel center is r times the
  // traction along the rolling direction t = a x n (a the wheel axle in world
  // axes), which the base-held inverse dynamics at rest returns as the wheel
  // torque, and which a force applied at the wheel center leaves out.
  const pinocchio::Model& model = reference.getModel();
  pinocchio::Data data(model);
  pinocchio::computeJointJacobians(model, data, q);
  pinocchio::updateFramePlacements(model, data);
  const vector_t vAtRest = vector_t::Zero(v.size());
  const vector_t heldWithTraction = computeBaseHeldJointTorques<scalar_t>(
      q, vAtRest, zeroJointAccelerations,
      wrenchWheelsEwrtWorld(settings, forceContactsEwrtWorld,
                            WheelForcePoint::kContactPoint),
      frames, reference);
  const vector_t heldWithoutTraction = computeBaseHeldJointTorques<scalar_t>(
      q, vAtRest, zeroJointAccelerations,
      wrenchWheelsEwrtWorld(settings, initForceContactsEwrtWorld,
                            WheelForcePoint::kContactPoint),
      frames, reference);
  const std::vector<std::string> wheelJoints =
      settings.rollingContactJointNames();
  ASSERT_EQ(wheelJoints.size(), numContacts);
  for (size_t contact = 0; contact < numContacts; ++contact) {
    const std::vector<std::string>::const_iterator found =
        std::find(settings.mpcModelJointNames.begin(),
                  settings.mpcModelJointNames.end(), wheelJoints[contact]);
    ASSERT_NE(found, settings.mpcModelJointNames.end());
    const Eigen::Index wheel = static_cast<Eigen::Index>(
        std::distance(settings.mpcModelJointNames.begin(), found));
    const int velocityIndex =
        model.joints[model.getJointId(wheelJoints[contact])].idx_v();
    matrix_t jacobian = matrix_t::Zero(kTwistDim, model.nv);
    pinocchio::getFrameJacobian(model, data, frames[contact],
                                pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                jacobian);
    const vector3_t axle =
        jacobian.block<kRotationTangentDim, 1>(kPositionDim, velocityIndex);
    const vector3_t rollingDirection =
        axle.cross(normalTerrainEwrtWorld<scalar_t>()).normalized();
    const scalar_t traction =
        (forceContactsEwrtWorld[contact] - initForceContactsEwrtWorld[contact])
            .dot(rollingDirection);
    const scalar_t radius = settings.wheelRadius(contact);
    EXPECT_NEAR(heldWithTraction[wheel] - heldWithoutTraction[wheel],
                radius * traction, kTolerance)
        << wheelJoints[contact];
    // On Tron1 the axles are world y at rest, so t is +x and a forward
    // traction asks for a positive wheel torque.
    EXPECT_GT(rollingDirection.x(), 0.99) << wheelJoints[contact];
    EXPECT_GT(std::abs(expected[wheel] - feedforwardAtWheelCenters[wheel]),
              0.5 * radius * std::abs(traction))
        << wheelJoints[contact];
  }
}

}  // namespace
}  // namespace ocs2::legged
