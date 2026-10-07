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
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_mpc/MPC_Settings.h"
#include "ocs2_mpc_test/ScriptedMpc.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/mrt/InProcessMpcLink.h"
#include "humanoid_common_mpc/mrt/MpcLink.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "humanoid_wb_mpc/mrt/WBMpcMrtJointController.h"
#include "humanoid_wb_mpc/mrt/WBMpcResetTarget.h"
#include "robot_core/ResourcePaths.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

/*
 * wbMpcResetTargetTrajectories(), the one reset target of the whole-body MPC: the observation held still and upright
 * with the weight on both feet, and exactly what the MRT joint controller hands its MPC link, so that the in-process
 * link and the MPC node (which calls the function itself) reset to the same target.
 */

namespace ocs2::humanoid {
namespace {

/** The Unitree G1 whole-body model of the shipped task file, without the CppAD libraries of the MPC. */
class G1Model {
 public:
  G1Model()
      // A file missing from the runfiles throws here, naming the data dependency to add (robot::resolveResourcePath).
      : task_(loadTaskFile(robot::resolveResourcePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto").value()).value()),
        urdfFile_(robot::resolveResourcePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf").value()),
        modelSettings_(ModelSettings::Create(task_, urdfFile_, "wb_mpc_", /*verbose=*/false).value()),
        pinocchioInterface_(loadCustomPinocchioInterface(task_, urdfFile_, modelSettings_).value()),
        model_(modelSettings_),
        description_(robot::model::RobotDescription::Create(urdfFile_).value()),
        initialState_(
            stateValuesFromConfig(task_.initial_state, stateInputLayout(modelSettings_, StateInputLayout::Mpc::kWholeBody), "initial_state")
                .value()) {}

  const ModelSettings& modelSettings() const { return modelSettings_; }
  const PinocchioInterface& pinocchioInterface() const { return pinocchioInterface_; }
  const WBAccelMpcRobotModel<scalar_t>& model() const { return model_; }
  const robot::model::RobotDescription& description() const { return description_; }
  const vector_t& initialState() const { return initialState_; }

  /** The standing state, moving and tilted, at `time`. */
  SystemObservation movingObservation(scalar_t time) const {
    SystemObservation observation;
    observation.time = time;
    observation.state = initialState_;
    observation.state(0) += 0.3;   // base x
    observation.state(3) += 0.4;   // yaw
    observation.state(4) = 0.05;   // pitch
    observation.state(5) = -0.03;  // roll
    observation.state.tail(model_.getGenCoordinatesDim()).setConstant(0.2);
    observation.input = vector_t::Zero(model_.getInputDim());
    observation.mode = ModeNumber::kStance;
    return observation;
  }

 private:
  mpc_config::TaskFile task_;
  std::string urdfFile_;
  ModelSettings modelSettings_;
  PinocchioInterface pinocchioInterface_;
  WBAccelMpcRobotModel<scalar_t> model_;
  robot::model::RobotDescription description_;
  vector_t initialState_;
};

void expectSameTarget(const TargetTrajectories& expected, const TargetTrajectories& actual) {
  ASSERT_EQ(expected.timeTrajectory, actual.timeTrajectory);
  ASSERT_EQ(expected.stateTrajectory.size(), actual.stateTrajectory.size());
  ASSERT_EQ(expected.inputTrajectory.size(), actual.inputTrajectory.size());
  for (size_t i = 0; i < expected.stateTrajectory.size(); ++i) {
    EXPECT_EQ(expected.stateTrajectory[i], actual.stateTrajectory[i]) << "node " << i;
    EXPECT_EQ(expected.inputTrajectory[i], actual.inputTrajectory[i]) << "node " << i;
  }
}

TEST(WBMpcResetTarget, HoldsTheObservedConfigurationStillAndUprightWithTheWeightOnBothFeet) {
  const G1Model g1;
  const SystemObservation observation = g1.movingObservation(/*time=*/3.5);
  const TargetTrajectories target = wbMpcResetTargetTrajectories(observation, g1.model(), g1.pinocchioInterface());

  ASSERT_EQ(target.timeTrajectory.size(), 2);
  EXPECT_EQ(target.timeTrajectory.front(), observation.time);
  EXPECT_GT(target.timeTrajectory.back(), target.timeTrajectory.front());
  ASSERT_EQ(target.stateTrajectory.size(), 2);
  ASSERT_EQ(target.inputTrajectory.size(), 2);
  EXPECT_EQ(target.stateTrajectory.front(), target.stateTrajectory.back());
  EXPECT_EQ(target.inputTrajectory.front(), target.inputTrajectory.back());

  const vector_t& state = target.stateTrajectory.front();
  const size_t numCoordinates = g1.model().getGenCoordinatesDim();
  EXPECT_TRUE(state.tail(numCoordinates).isZero(0.0));
  EXPECT_EQ(state(4), 0.0);
  EXPECT_EQ(state(5), 0.0);
  // The configuration is observed: position, yaw and the joints.
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(numCoordinates); ++i) {
    if (i == 4 || i == 5) continue;
    EXPECT_EQ(state(i), observation.state(i)) << "state entry " << i;
  }

  // The robot's weight, carried by both feet in the world frame.
  const vector_t& input = target.inputTrajectory.front();
  const vector3_t left = g1.model().getContactForceInWorldFrame(state, input, /*contactIndex=*/0);
  const vector3_t right = g1.model().getContactForceInWorldFrame(state, input, /*contactIndex=*/1);
  const scalar_t weight = computeRobotWeight(g1.pinocchioInterface());
  EXPECT_NEAR(left.z() + right.z(), weight, 1.0e-9 * weight);
  EXPECT_NEAR(left.z(), right.z(), 1.0e-9 * weight);
}

TEST(WBMpcResetTarget, IsWhatTheControllerHandsItsMpcLink) {
  const G1Model g1;
  mpc_test::ScriptedMpc mpc(mpc::Settings(), g1.model().getInputDim());
  MpcLink::ResetTargetFunction captured;
  const MpcLinkFactory factory = [&mpc, &captured](MpcLink::ResetTargetFunction resetTarget) -> std::unique_ptr<MpcLink> {
    captured = resetTarget;
    return std::make_unique<InProcessMpcLink>(mpc, std::move(resetTarget), InProcessMpcLink::Config());
  };
  const absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>> controller =
      WBMpcMrtJointController::Create(g1.description(), g1.modelSettings(), factory, g1.pinocchioInterface());
  ASSERT_TRUE(controller.ok()) << controller.status();
  ASSERT_TRUE(captured);
  for (const scalar_t time : {0.0, 1.25, 7.5}) {
    const SystemObservation observation = g1.movingObservation(time);
    expectSameTarget(wbMpcResetTargetTrajectories(observation, g1.model(), g1.pinocchioInterface()), captured(observation));
  }
}

TEST(WBMpcResetTarget, IsWhatTheInProcessLinkResetsTheMpcTo) {
  const G1Model g1;
  // A bare scripted MPC: no synchronized module rewrites the target after the reset, so the reference manager keeps it.
  mpc_test::ScriptedMpc mpc(mpc::Settings(), g1.model().getInputDim());

  robot::model::RobotState robotState(g1.description());
  const vector_t& initialState = g1.initialState();
  robotState.setTime(2.0);
  robotState.setRootPositionInWorldFrame(g1.model().getBasePosition(initialState));
  robotState.setRootRotationLocalToWorldFrame(
      getQuaternionFromEulerAnglesZyx(vector3_t(g1.model().getBaseOrientationEulerZYX(initialState))));
  const std::vector<size_t> jointIndices = g1.description().getJointIndices(g1.modelSettings().mpcModelJointNames);
  const vector_t joints = g1.model().getJointAngles(initialState);
  for (size_t i = 0; i < jointIndices.size(); ++i) robotState.setJointPosition(jointIndices[i], joints[i]);

  SystemObservation startObservation;
  {
    // The default constructor's link: InProcessMpcLink, which resets the MPC fully from the start observation.
    absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>> created =
        WBMpcMrtJointController::Create(g1.description(), g1.modelSettings(), mpc, g1.pinocchioInterface(), /*mpcDesiredFrequency=*/100.0);
    ASSERT_TRUE(created.ok()) << created.status();
    WBMpcMrtJointController& controller = **created;
    controller.startMpcThread(robotState);
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!controller.ready() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(controller.ready());
    startObservation = controller.getCurrentObservation();
  }  // Joins the solver thread, so the reference manager is read alone.

  expectSameTarget(wbMpcResetTargetTrajectories(startObservation, g1.model(), g1.pinocchioInterface()),
                   mpc.getSolverPtr()->getReferenceManager().getTargetTrajectories());
}

}  // namespace
}  // namespace ocs2::humanoid
