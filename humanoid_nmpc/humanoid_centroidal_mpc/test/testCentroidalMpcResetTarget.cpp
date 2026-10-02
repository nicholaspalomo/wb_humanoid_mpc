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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_mpc/MPC_Settings.h>
#include <ocs2_mpc_test/ScriptedMpc.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include <robot_model/RobotDescription.h>
#include <robot_model/RobotState.h>

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h"
#include "humanoid_centroidal_mpc/mrt/CentroidalMpcResetTarget.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/mrt/InProcessMpcLink.h"
#include "humanoid_common_mpc/mrt/MpcLink.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "support/AtlasReferenceStack.h"

/*
 * centroidalMpcResetTargetTrajectories(), the one reset target of the centroidal MPC: the observation held still and
 * upright with the weight on both feet, and exactly what the MRT joint controller hands its MPC link, so that the
 * in-process link and the MPC node (which calls the function itself) reset to the same target.
 */

namespace ocs2::humanoid {
namespace {

/** The DRC Atlas standing state, moving and tilted, at `time`. */
SystemObservation movingObservation(const AtlasReferenceStack& stack, scalar_t time) {
  SystemObservation observation;
  observation.time = time;
  observation.state = stack.initialState();
  for (Eigen::Index i = 0; i < 6; ++i) observation.state(i) = 0.1 * static_cast<scalar_t>(i + 1);  // normalized momentum
  observation.state(6) += 0.3;                                                                     // base x
  observation.state(9) += 0.4;                                                                     // yaw
  observation.state(10) = 0.05;                                                                    // pitch
  observation.state(11) = -0.03;                                                                   // roll
  observation.state.tail(3).array() += 0.2;
  observation.input = vector_t::Zero(stack.model().getInputDim());
  observation.mode = ModeNumber::STANCE;
  return observation;
}

void expectSameTarget(const TargetTrajectories& expected, const TargetTrajectories& actual) {
  ASSERT_EQ(expected.timeTrajectory, actual.timeTrajectory);
  ASSERT_EQ(expected.stateTrajectory.size(), actual.stateTrajectory.size());
  ASSERT_EQ(expected.inputTrajectory.size(), actual.inputTrajectory.size());
  for (size_t i = 0; i < expected.stateTrajectory.size(); ++i) {
    EXPECT_EQ(expected.stateTrajectory[i], actual.stateTrajectory[i]) << "node " << i;
    EXPECT_EQ(expected.inputTrajectory[i], actual.inputTrajectory[i]) << "node " << i;
  }
}

TEST(CentroidalMpcResetTarget, HoldsTheObservedConfigurationStillAndUprightWithTheWeightOnBothFeet) {
  const AtlasReferenceStack stack;
  const SystemObservation observation = movingObservation(stack, /*time=*/3.5);
  const TargetTrajectories target =
      centroidalMpcResetTargetTrajectories(observation, stack.centroidalModelInfo(), stack.model(), stack.pinocchioInterface());

  ASSERT_EQ(target.timeTrajectory.size(), 2);
  EXPECT_EQ(target.timeTrajectory.front(), observation.time);
  EXPECT_GT(target.timeTrajectory.back(), target.timeTrajectory.front());
  ASSERT_EQ(target.stateTrajectory.size(), 2);
  ASSERT_EQ(target.inputTrajectory.size(), 2);
  EXPECT_EQ(target.stateTrajectory.front(), target.stateTrajectory.back());
  EXPECT_EQ(target.inputTrajectory.front(), target.inputTrajectory.back());

  const vector_t& state = target.stateTrajectory.front();
  vector_t momentum = state;
  EXPECT_TRUE(centroidal_model::getNormalizedMomentum(momentum, stack.centroidalModelInfo()).isZero(0.0));
  EXPECT_EQ(state(10), 0.0);
  EXPECT_EQ(state(11), 0.0);
  // Everything else is observed: position, yaw and the joints.
  for (Eigen::Index i = 6; i < state.size(); ++i) {
    if (i == 10 || i == 11) continue;
    EXPECT_EQ(state(i), observation.state(i)) << "state entry " << i;
  }

  // The robot's weight, carried by both feet in the world frame.
  const vector_t& input = target.inputTrajectory.front();
  const vector3_t left = stack.model().getContactForceInWorldFrame(state, input, /*contactIndex=*/0);
  const vector3_t right = stack.model().getContactForceInWorldFrame(state, input, /*contactIndex=*/1);
  const scalar_t weight = computeRobotWeight(stack.pinocchioInterface());
  EXPECT_NEAR(left.z() + right.z(), weight, 1e-9 * weight);
  EXPECT_NEAR(left.z(), right.z(), 1e-9 * weight);
}

/** The controller's link, made by a factory that keeps the reset target the controller hands it. */
class CapturingFactory {
 public:
  explicit CapturingFactory(MPC_BASE& mpc) : mpc_(mpc) {}

  MpcLinkFactory factory() {
    return [this](MpcLink::ResetTargetFunction resetTarget) -> std::unique_ptr<MpcLink> {
      resetTarget_ = resetTarget;
      return std::make_unique<InProcessMpcLink>(mpc_, std::move(resetTarget), InProcessMpcLink::Config());
    };
  }
  const MpcLink::ResetTargetFunction& resetTarget() const { return resetTarget_; }

 private:
  MPC_BASE& mpc_;
  MpcLink::ResetTargetFunction resetTarget_;
};

TEST(CentroidalMpcResetTarget, IsWhatTheControllerHandsItsMpcLink) {
  AtlasReferenceStack stack;
  const robot::model::RobotDescription description(stack.urdfFile());
  CapturingFactory capturing(stack.mpc());
  const CentroidalMpcMrtJointController controller(description, stack.modelSettings(), stack.model(), capturing.factory(),
                                                   stack.pinocchioInterface());
  ASSERT_TRUE(capturing.resetTarget());
  for (const scalar_t time : {0.0, 1.25, 7.5}) {
    const SystemObservation observation = movingObservation(stack, time);
    expectSameTarget(
        centroidalMpcResetTargetTrajectories(observation, stack.centroidalModelInfo(), stack.model(), stack.pinocchioInterface()),
        capturing.resetTarget()(observation));
  }
}

TEST(CentroidalMpcResetTarget, IsWhatTheInProcessLinkResetsTheMpcTo) {
  const AtlasReferenceStack stack;
  const robot::model::RobotDescription description(stack.urdfFile());
  // A bare scripted MPC: no synchronized module rewrites the target after the reset, so the reference manager keeps it.
  mpc_test::ScriptedMpc mpc(mpc::Settings(), stack.model().getInputDim());

  robot::model::RobotState robotState(description);
  const vector_t& initialState = stack.initialState();
  robotState.setTime(2.0);
  robotState.setRootPositionInWorldFrame(stack.model().getBasePosition(initialState));
  robotState.setRootRotationLocalToWorldFrame(
      getQuaternionFromEulerAnglesZyx(vector3_t(stack.model().getBaseOrientationEulerZYX(initialState))));
  const std::vector<size_t> jointIndices = description.getJointIndices(stack.modelSettings().mpcModelJointNames);
  const vector_t joints = stack.model().getJointAngles(initialState);
  for (size_t i = 0; i < jointIndices.size(); ++i) robotState.setJointPosition(jointIndices[i], joints[i]);

  SystemObservation startObservation;
  {
    // The default constructor's link: InProcessMpcLink, which resets the MPC fully from the start observation.
    CentroidalMpcMrtJointController controller(description, stack.modelSettings(), stack.model(), mpc, stack.pinocchioInterface(),
                                               /*mpcDesiredFrequency=*/100.0);
    controller.startMpcThread(robotState);
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!controller.ready() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(controller.ready());
    startObservation = controller.getCurrentObservation();
  }  // Joins the solver thread, so the reference manager is read alone.

  expectSameTarget(
      centroidalMpcResetTargetTrajectories(startObservation, stack.centroidalModelInfo(), stack.model(), stack.pinocchioInterface()),
      mpc.getSolverPtr()->getReferenceManager().getTargetTrajectories());
}

}  // namespace
}  // namespace ocs2::humanoid
