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

#include <chrono>
#include <cmath>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/AccessHelperFunctions.h"
#include "ocs2_mpc/MPC_Settings.h"
#include "ocs2_mpc_test/ScriptedMpc.h"

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h"
#include "humanoid_centroidal_mpc/mrt/CentroidalMpcResetTarget.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/mrt/InProcessMpcLink.h"
#include "humanoid_common_mpc/mrt/MpcLink.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"
#include "support/AtlasReferenceStack.h"

/*
 * centroidalMpcResetTargetTrajectories(), the one reset target of the
 * centroidal MPC: the observation held still and upright at the twist heading
 * of its attitude with the weight on both feet, and exactly what the MRT joint
 * controller hands its MPC link, so that the in-process link and the MPC node
 * (which calls the function itself) reset to the same target.
 */

namespace ocs2::humanoid {
namespace {

/** The DRC Atlas standing state, moving and tilted, at `time`. */
SystemObservation movingObservation(const AtlasReferenceStack& stack,
                                    scalar_t time) {
  SystemObservation observation;
  observation.time = time;
  observation.state = stack.initialState();
  for (Eigen::Index i = 0; i < 6; ++i)
    observation.state(i) =
        0.1 * static_cast<scalar_t>(i + 1);  // normalized momentum
  const CentroidalMpcRobotModel<scalar_t>& model = stack.model();
  model.setPosBaseRtWorldEwrtWorld(
      observation.state,
      vector3_t(model.getPosBaseRtWorldEwrtWorld(observation.state) +
                vector3_t(0.3, 0.0, 0.0)));
  // Turned by 0.4 rad, pitched and rolled.
  model.setEulerzyxBaseRtWorld(
      observation.state,
      vector3_t(model.getYawBaseRtWorld(observation.state) + 0.4, 0.05, -0.03));
  observation.state.tail(3).array() += 0.2;
  observation.input = vector_t::Zero(stack.model().getInputDim());
  observation.mode = ModeNumber::kStance;
  return observation;
}

void expectSameTarget(const TargetTrajectories& expected,
                      const TargetTrajectories& actual) {
  ASSERT_EQ(expected.timeTrajectory, actual.timeTrajectory);
  ASSERT_EQ(expected.stateTrajectory.size(), actual.stateTrajectory.size());
  ASSERT_EQ(expected.inputTrajectory.size(), actual.inputTrajectory.size());
  for (size_t i = 0; i < expected.stateTrajectory.size(); ++i) {
    EXPECT_EQ(expected.stateTrajectory[i], actual.stateTrajectory[i])
        << "node " << i;
    EXPECT_EQ(expected.inputTrajectory[i], actual.inputTrajectory[i])
        << "node " << i;
  }
}

TEST(CentroidalMpcResetTarget,
     HoldsTheObservedConfigurationStillAndUprightWithTheWeightOnBothFeet) {
  const AtlasReferenceStack stack;
  const SystemObservation observation = movingObservation(stack, /*time=*/3.5);
  const TargetTrajectories target = centroidalMpcResetTargetTrajectories(
      observation, stack.centroidalModelInfo(), stack.model(),
      stack.pinocchioInterface());

  ASSERT_EQ(target.timeTrajectory.size(), 2);
  EXPECT_EQ(target.timeTrajectory.front(), observation.time);
  EXPECT_GT(target.timeTrajectory.back(), target.timeTrajectory.front());
  ASSERT_EQ(target.stateTrajectory.size(), 2);
  ASSERT_EQ(target.inputTrajectory.size(), 2);
  EXPECT_EQ(target.stateTrajectory.front(), target.stateTrajectory.back());
  EXPECT_EQ(target.inputTrajectory.front(), target.inputTrajectory.back());

  const vector_t& state = target.stateTrajectory.front();
  vector_t momentum = state;
  EXPECT_TRUE(centroidal_model::getMnormCmomComRtWorldEwrtWorld(
                  momentum, stack.centroidalModelInfo())
                  .isZero(0.0));
  // Upright at the observed heading, the twist of the observed attitude (design
  // decision D9); everything else is observed: the position and the joints.
  const CentroidalMpcRobotModel<scalar_t>& model = stack.model();
  const vector3_t eulerZyx = model.getEulerzyxBaseRtWorld(state);
  EXPECT_NEAR(eulerZyx(0), model.getYawBaseRtWorld(observation.state), 1.0e-12);
  EXPECT_EQ(eulerZyx(1), 0.0);
  EXPECT_EQ(eulerZyx(2), 0.0);
  EXPECT_EQ(model.getPosBaseRtWorldEwrtWorld(state),
            model.getPosBaseRtWorldEwrtWorld(observation.state));
  EXPECT_EQ(model.getJointAngles(state),
            model.getJointAngles(observation.state));

  // The robot's weight, carried by both feet in the world frame.
  const vector_t& input = target.inputTrajectory.front();
  const vector3_t left =
      stack.model().getForceContactEwrtWorld(state, input, /*contactIndex=*/0);
  const vector3_t right =
      stack.model().getForceContactEwrtWorld(state, input, /*contactIndex=*/1);
  const scalar_t weight = computeRobotWeight(stack.pinocchioInterface());
  EXPECT_NEAR(left.z() + right.z(), weight, 1.0e-9 * weight);
  EXPECT_NEAR(left.z(), right.z(), 1.0e-9 * weight);
}

TEST(CentroidalMpcResetTarget,
     IsUprightAtTheTwistHeadingOfTheObservedAttitude) {
  const AtlasReferenceStack stack;
  const CentroidalMpcRobotModel<scalar_t>& model = stack.model();
  // Stored yaws on both sides of +-pi and beyond it, tilted: the heading of the
  // target is the twist of the observed attitude, wrapped into (-pi, pi], which
  // differs from the stored yaw by the wrap and by -2 atan(tan(theta / 2)
  // tan(phi / 2)).
  for (const vector3_t& eulerZyx :
       {vector3_t(3.5, 0.15, 0.15), vector3_t(-3.1, -0.2, 0.1),
        vector3_t(7.0, 0.05, -0.3)}) {
    SCOPED_TRACE(eulerZyx.transpose());
    SystemObservation observation = movingObservation(stack, /*time=*/1.0);
    model.setEulerzyxBaseRtWorld(observation.state, eulerZyx);
    const vector_t target = centroidalMpcResetTargetTrajectories(
                                observation, stack.centroidalModelInfo(), model,
                                stack.pinocchioInterface())
                                .stateTrajectory.front();

    const scalar_t heading = model.getYawBaseRtWorld(observation.state);
    const scalar_t expectedHeading = std::remainder(
        eulerZyx(0) - 2.0 * std::atan(std::tan(eulerZyx(1) / 2.0) *
                                      std::tan(eulerZyx(2) / 2.0)),
        2.0 * M_PI);
    EXPECT_NEAR(heading, expectedHeading, 1.0e-12);
    const vector3_t targetEulerZyx = model.getEulerzyxBaseRtWorld(target);
    EXPECT_NEAR(targetEulerZyx(0), heading, 1.0e-12);
    EXPECT_GT(targetEulerZyx(0), -M_PI);
    EXPECT_LE(targetEulerZyx(0), M_PI);
    EXPECT_EQ(targetEulerZyx(1), 0.0);
    EXPECT_EQ(targetEulerZyx(2), 0.0);
    // Its attitude is the heading quaternion of the observed one, up to the
    // sign of the quaternion.
    const vector4_t expected = verticalTwistFromQuaternion(
        model.getQuatBaseRtWorldXyzw(observation.state));
    const vector4_t actual = model.getQuatBaseRtWorldXyzw(target);
    EXPECT_NEAR(std::abs(actual.dot(expected)), 1.0, 1.0e-15);
  }
}

/** The controller's link, made by a factory that keeps the reset target the
 * controller hands it. */
class CapturingFactory {
 public:
  explicit CapturingFactory(MPC_BASE& mpc) : mpc_(mpc) {}

  MpcLinkFactory factory() {
    return [this](MpcLink::ResetTargetFunction resetTarget)
               -> std::unique_ptr<MpcLink> {
      resetTarget_ = resetTarget;
      return std::make_unique<InProcessMpcLink>(mpc_, std::move(resetTarget),
                                                InProcessMpcLink::Config());
    };
  }
  const MpcLink::ResetTargetFunction& resetTarget() const {
    return resetTarget_;
  }

 private:
  MPC_BASE& mpc_;
  MpcLink::ResetTargetFunction resetTarget_;
};

TEST(CentroidalMpcResetTarget, IsWhatTheControllerHandsItsMpcLink) {
  AtlasReferenceStack stack;
  absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus =
      robot::model::RobotDescription::Create(stack.urdfFile());
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  CapturingFactory capturing(stack.mpc());
  absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>>
      controllerOrStatus = CentroidalMpcMrtJointController::Create(
          description, stack.modelSettings(), stack.model(),
          capturing.factory(), stack.pinocchioInterface(), stack.pdGainsFile());
  // The controller, which owns the link the factory captured, lives in
  // controllerOrStatus for the whole test.
  ASSERT_TRUE(controllerOrStatus.ok()) << controllerOrStatus.status();
  ASSERT_TRUE(capturing.resetTarget());
  for (const scalar_t time : {0.0, 1.25, 7.5}) {
    const SystemObservation observation = movingObservation(stack, time);
    expectSameTarget(centroidalMpcResetTargetTrajectories(
                         observation, stack.centroidalModelInfo(),
                         stack.model(), stack.pinocchioInterface()),
                     capturing.resetTarget()(observation));
  }
}

TEST(CentroidalMpcResetTarget, IsWhatTheInProcessLinkResetsTheMpcTo) {
  const AtlasReferenceStack stack;
  absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus =
      robot::model::RobotDescription::Create(stack.urdfFile());
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  // A bare scripted MPC: no synchronized module rewrites the target after the
  // reset, so the reference manager keeps it.
  mpc_test::ScriptedMpc mpc(mpc::Settings(), stack.model().getInputDim());

  robot::model::RobotState robotState(description);
  const vector_t& initialState = stack.initialState();
  robotState.setTime(2.0);
  robotState.setPosBaseRtWorldEwrtWorld(
      stack.model().getPosBaseRtWorldEwrtWorld(initialState));
  robotState.setQuatBaseRtWorld(
      quaternion_t(stack.model().getQuatBaseRtWorldXyzw(initialState)));
  const std::vector<size_t> jointIndices =
      description.getJointIndices(stack.modelSettings().mpcModelJointNames);
  const vector_t joints = stack.model().getJointAngles(initialState);
  for (size_t i = 0; i < jointIndices.size(); ++i)
    robotState.setJointPosition(jointIndices[i], joints[i]);

  SystemObservation startObservation;
  {
    // The default constructor's link: InProcessMpcLink, which resets the MPC
    // fully from the start observation.
    absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>>
        controllerOrStatus = CentroidalMpcMrtJointController::Create(
            description, stack.modelSettings(), stack.model(), mpc,
            /*solverObservationObserver=*/nullptr, stack.pinocchioInterface(),
            /*mpcDesiredFrequency=*/100.0, stack.pdGainsFile());
    ASSERT_TRUE(controllerOrStatus.ok()) << controllerOrStatus.status();
    CentroidalMpcMrtJointController& controller = **controllerOrStatus;
    controller.startMpcThread(robotState);
    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!controller.ready() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(controller.ready());
    startObservation = controller.getCurrentObservation();
  }  // Joins the solver thread, so the reference manager is read alone.

  expectSameTarget(
      centroidalMpcResetTargetTrajectories(
          startObservation, stack.centroidalModelInfo(), stack.model(),
          stack.pinocchioInterface()),
      mpc.getSolverPtr()->getReferenceManager().getTargetTrajectories());
}

}  // namespace
}  // namespace ocs2::humanoid
