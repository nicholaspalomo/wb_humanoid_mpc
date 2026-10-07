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

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h"
#include "humanoid_common_mpc/mrt/InProcessMpcLink.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"
#include "support/AtlasReferenceStack.h"

/*
 * The joint actions CentroidalMpcMrtJointController commands, bit for bit, on every path of computeJointControlAction():
 * WB_MPC before the first policy (weight compensation), WB_MPC executing a policy, ZERO_TORQUE, JOINT_PD, GRAVITY_COMP,
 * SAFETY, the holds of an entry into WB_MPC from JOINT_PD and from GRAVITY_COMP, the ramp into the policy, and a
 * diverged policy. The link is the lockstep one (InProcessMpcLink::Execution::kCaller), so every run solves at the same
 * cycles and the run is the same run every time.
 *
 * The golden file was recorded by the control loop that read every joint action with `.at(index).value()`, before the
 * constructor validated the joint indices once and the loop stopped checking them per cycle; it is what shows that the
 * rewrite changed no action. A change that alters the actions on purpose re-records it: on a mismatch the test writes
 * what it computed to $TEST_UNDECLARED_OUTPUTS_DIR/mrt_joint_controller_actions.txt (in the test's outputs.zip), which
 * replaces test/golden/mrt_joint_controller_actions.txt.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kControlPeriod = 0.01;  // [s] of observation time per control cycle
constexpr char kGoldenFile[] = "humanoid_nmpc/humanoid_centroidal_mpc/test/golden/mrt_joint_controller_actions.txt";
constexpr char kOutputName[] = "mrt_joint_controller_actions.txt";

/** A plan whose joints are `offset` [rad] away from the observed ones over the whole horizon. */
mpc_test::ScriptedSolver::PlanFunction jointOffsetPlan(const AtlasReferenceStack& stack, scalar_t offset) {
  const size_t jointStart = stack.model().getJointStartindex();
  const size_t jointDim = stack.model().getJointDim();
  return [jointStart, jointDim, offset](scalar_t /*time*/, scalar_t /*initTime*/, const vector_t& initState) {
    vector_t planned = initState;
    planned.segment(jointStart, jointDim).array() += offset;
    return planned;
  };
}

/** The controller on the lockstep link, its robot, and the record of every action it commanded. */
class ActionRecorder {
 public:
  ActionRecorder()
      : description_(robot::model::RobotDescription::Create(stack_.urdfFile()).value()), robotState_(description_), action_(description_) {
    InProcessMpcLink::Config config;
    config.execution = InProcessMpcLink::Execution::kCaller;
    controller_ = CentroidalMpcMrtJointController::Create(description_, stack_.modelSettings(), stack_.model(),
                                                          InProcessMpcLink::factory(stack_.mpc(), std::move(config), &link_),
                                                          stack_.pinocchioInterface())
                      .value();
    mpcJointIndices_ = description_.getJointIndices(stack_.modelSettings().mpcModelJointNames);
    setState(/*cycle=*/0);
    std::vector<scalar_t> nominal(description_.getNumJoints(), 0.0);
    for (size_t joint = 0; joint < description_.getNumJoints(); ++joint) nominal[joint] = robotState_.getJointPosition(joint);
    controller_->setNominalJointPositions(nominal);
  }

  AtlasReferenceStack& stack() { return stack_; }
  CentroidalMpcMrtJointController& controller() { return *controller_; }
  InProcessMpcLink& link() { return *link_; }
  const std::string& record() const { return record_; }

  /** startMpcThread() at the current time: on the lockstep link it serves the start-up reset and starts no thread. */
  void start() {
    robotState_.setTime(time_);
    controller_->startMpcThread(robotState_);
  }

  /** `numCycles` control cycles of the robot moving on a fixed path, each one's action appended to the record. */
  void cycles(absl::string_view label, int numCycles) {
    for (int k = 0; k < numCycles; ++k) {
      setState(cycle_);
      robotState_.setTime(time_);
      controller_->computeJointControlAction(time_, robotState_, action_);
      absl::StrAppendFormat(&record_, "%s %d\n", label, cycle_);
      for (size_t joint = 0; joint < description_.getNumJoints(); ++joint) {
        const std::optional<robot::model::JointAction>& slot = action_.at(joint);
        if (!slot.has_value()) {
          absl::StrAppendFormat(&record_, "%d none\n", joint);
          continue;
        }
        absl::StrAppendFormat(&record_, "%d %a %a %a %a %a\n", joint, slot->q_des, slot->qd_des, slot->kp, slot->kd,
                              slot->feed_forward_effort);
      }
      time_ += kControlPeriod;
      ++cycle_;
    }
  }

 private:
  /** The robot at the task file's initial state, swaying on a fixed path: every cycle has its own state. */
  void setState(int cycle) {
    const CentroidalMpcRobotModel<scalar_t>& model = stack_.model();
    const vector_t& state = stack_.initialState();
    const scalar_t phase = 0.3 * static_cast<scalar_t>(cycle);
    robotState_.setConfigurationToZero();
    robotState_.setRootPositionInWorldFrame(model.getBasePosition(state) + vector3_t(0.01 * std::sin(phase), 0.005 * std::cos(phase), 0.0));
    const vector3_t eulerZyx(model.getBaseOrientationEulerZYX(state) + vector3_t(0.02 * std::sin(phase), 0.01, -0.01 * std::cos(phase)));
    const quaternion_t orientation = getQuaternionFromEulerAnglesZyx(eulerZyx);
    robotState_.setRootRotationLocalToWorldFrame(orientation);
    robotState_.setRootLinearVelocityInLocalFrame(vector3_t(0.1 * std::cos(phase), -0.05, 0.02));
    robotState_.setRootAngularVelocityInLocalFrame(vector3_t(0.03, -0.02 * std::sin(phase), 0.01));
    const vector_t joints = model.getJointAngles(state);
    for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
      const scalar_t jointPhase = phase + static_cast<scalar_t>(i);
      robotState_.setJointPosition(mpcJointIndices_[i], joints[i] + 0.02 * std::sin(jointPhase));
      robotState_.setJointVelocity(mpcJointIndices_[i], 0.1 * std::cos(jointPhase));
    }
    robotState_.setContactFlag(/*index=*/0, /*contactFlag=*/true);
    robotState_.setContactFlag(/*index=*/1, /*contactFlag=*/cycle % 7 != 3);
  }

  AtlasReferenceStack stack_;
  robot::model::RobotDescription description_;
  robot::model::RobotState robotState_;
  robot::model::RobotJointAction action_;
  std::vector<size_t> mpcJointIndices_;
  scalar_t time_ = 1.0;
  int cycle_ = 0;
  std::string record_;
  InProcessMpcLink* absl_nullable link_ = nullptr;
  std::unique_ptr<CentroidalMpcMrtJointController> controller_;
};

/** The whole file at `path`, or nullopt when it cannot be read. */
std::optional<std::string> readFile(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) return std::nullopt;
  std::stringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

/** Writes `record` where the test's undeclared outputs go, and returns where that is (empty when it cannot). */
std::string writeOutput(const std::string& record) {
  const char* absl_nullable outputDir = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  if (outputDir == nullptr) return std::string();
  const std::string path = (std::filesystem::path(outputDir) / kOutputName).string();
  std::ofstream file(path, std::ios::trunc);
  file << record;
  return file.good() ? path : std::string();
}

TEST(MrtJointControllerActionParity, EveryPathCommandsTheRecordedActions) {
  ActionRecorder recorder;
  CentroidalMpcMrtJointController& controller = recorder.controller();
  controller.setMpcEntryBlendTime(0.05);
  recorder.stack().mpc().solver().setPlan(jointOffsetPlan(recorder.stack(), /*offset=*/0.1));
  recorder.start();

  recorder.cycles("WB_MPC without a policy", /*numCycles=*/2);
  ASSERT_TRUE(recorder.link().runSolverIteration().status.ok());
  recorder.cycles("WB_MPC", /*numCycles=*/3);
  controller.setControlMode("ZERO_TORQUE");
  recorder.cycles("ZERO_TORQUE", /*numCycles=*/2);
  controller.setControlMode("JOINT_PD");
  recorder.cycles("JOINT_PD", /*numCycles=*/2);
  controller.setControlMode("GRAVITY_COMP");
  recorder.cycles("GRAVITY_COMP", /*numCycles=*/2);
  controller.setControlMode("WB_MPC");
  recorder.cycles("WB_MPC held in GRAVITY_COMP", /*numCycles=*/2);
  controller.setControlMode("SAFETY");
  recorder.cycles("SAFETY", /*numCycles=*/4);
  controller.setControlMode("JOINT_PD");
  recorder.cycles("JOINT_PD after SAFETY", /*numCycles=*/1);
  controller.setControlMode("WB_MPC");
  recorder.cycles("WB_MPC held in JOINT_PD", /*numCycles=*/2);
  ASSERT_TRUE(recorder.link().runSolverIteration().status.ok());
  recorder.cycles("WB_MPC ramp", /*numCycles=*/7);
  recorder.stack().mpc().solver().setPlan(jointOffsetPlan(recorder.stack(), /*offset=*/1.0));
  ASSERT_TRUE(recorder.link().runSolverIteration().status.ok());
  recorder.cycles("WB_MPC diverged", /*numCycles=*/2);

  const std::string goldenPath = atlasRunfilePath(kGoldenFile);
  ASSERT_FALSE(goldenPath.empty()) << kGoldenFile << " is not in the runfiles";
  const std::optional<std::string> golden = readFile(goldenPath);
  if (!golden.has_value()) GTEST_FAIL() << "cannot read " << goldenPath;
  if (*golden == recorder.record()) return;

  const std::vector<absl::string_view> expected = absl::StrSplit(*golden, '\n');
  const std::vector<absl::string_view> actual = absl::StrSplit(recorder.record(), '\n');
  size_t line = 0;
  while (line < expected.size() && line < actual.size() && expected[line] == actual[line]) ++line;
  const absl::string_view expectedLine = line < expected.size() ? expected[line] : absl::string_view("(end of file)");
  const absl::string_view actualLine = line < actual.size() ? actual[line] : absl::string_view("(end of record)");
  ADD_FAILURE() << "the actions differ from " << kGoldenFile << " from line " << line + 1 << " on: expected '" << expectedLine
                << "', commanded '" << actualLine << "'. What was commanded is in " << writeOutput(recorder.record());
}

}  // namespace
}  // namespace ocs2::humanoid
