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

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <string>
#include <vector>

#include <pinocchio/algorithm/rnea.hpp>

#include <robot_model/RobotDescription.h>
#include <robot_model/RobotJointAction.h>
#include <robot_model/RobotState.h>

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/mrt/ControllerEventSink.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/NullMpcLink.h"
#include "humanoid_wb_mpc/mrt/WBMpcMrtJointController.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

/*
 * What the whole-body MRT joint controller does on the realtime thread of the robot process adds no allocation of its
 * own once it is built: the observation of every cycle, the passive modes (each entry included) and the hold of WB_MPC
 * until a policy arrives. What remains is Pinocchio's: the nonlinear effects of the gravity compensation make a
 * temporary of the composite base joint's dynamic-size motion subspace, which the test measures on the same model and
 * expects. The policy's execution in WB_MPC is not covered: its inverse dynamics still allocates (the package README).
 */

namespace ocs2::humanoid {
namespace {

std::string runfilePath(const std::string& relativePath) {
  const char* srcDir = std::getenv("TEST_SRCDIR");
  const std::filesystem::path candidate =
      srcDir != nullptr ? std::filesystem::path(srcDir) / "_main" / relativePath : std::filesystem::path(relativePath);
  return candidate.string();
}

class CountingSink final : public ControllerEventSink {
 public:
  bool post(const ControllerEvent& /*event*/) override {
    posted.fetch_add(1);
    return true;
  }
  std::atomic<int> posted{0};
};

constexpr const char* kModes[] = {"ZERO_TORQUE", "JOINT_PD", "GRAVITY_COMP", "SAFETY", "JOINT_PD", "WB_MPC", "ZERO_TORQUE"};
/** Whether a mode's action needs the gravity compensation (the WB_MPC hold is JOINT_PD's). */
constexpr bool kComputesGravity[] = {false, true, true, false, true, true, false};
constexpr int kCyclesPerMode = 20;

void runModes(WBMpcMrtJointController& controller,
              const std::vector<scalar_t>& nominal,
              robot::model::RobotState& state,
              robot::model::RobotJointAction& action,
              scalar_t& time,
              std::vector<std::size_t>* allocationsPerMode = nullptr) {
  for (const char* mode : kModes) {
    const std::size_t before = robot::realtime::heapAllocationCountOnThisThread();
    for (int cycle = 0; cycle < kCyclesPerMode; ++cycle) {
      // The whole-body controller takes the posture before the mode, as its sim did (CycleInputOrder).
      controller.setNominalJointPositions(nominal);
      controller.setControlMode(mode);
      time += 0.002;
      state.setTime(time);
      controller.computeJointControlAction(time, state, action);
    }
    if (allocationsPerMode != nullptr) allocationsPerMode->push_back(robot::realtime::heapAllocationCountOnThisThread() - before);
  }
}

TEST(WBMpcMrtJointControllerAllocations, TheObservationThePassiveModesAndTheHoldAddNoAllocationOfTheirOwn) {
  const std::string taskFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml");
  const std::string urdfFile = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
  const ModelSettings modelSettings(taskFile, urdfFile, "wb_mpc_", /*verbose=*/false);
  const PinocchioInterface pinocchioInterface = createCustomPinocchioInterface(taskFile, urdfFile, modelSettings);
  const robot::model::RobotDescription description(urdfFile);
  robot::model::RobotState state(description);
  state.setRootPositionInWorldFrame(vector3_t(0.0, 0.0, 0.75));
  robot::model::RobotJointAction action(description);
  WBMpcMrtJointController controller(description, modelSettings, test_support::NullMpcLink::factory(), pinocchioInterface);
  CountingSink sink;
  controller.setEventSink(&sink);
  const std::vector<scalar_t> nominal(description.getNumJoints(), 0.1);
  scalar_t time = 0.0;

  // Pinocchio's own allocations in one pass of the nonlinear effects, on a copy of the controller's model.
  PinocchioInterface probe = pinocchioInterface;
  const vector_t q = vector_t::Zero(probe.getModel().nq);
  const vector_t v = vector_t::Zero(probe.getModel().nv);
  pinocchio::nonLinearEffects(probe.getModel(), probe.getData(), q, v);
  const std::size_t before = robot::realtime::heapAllocationCountOnThisThread();
  pinocchio::nonLinearEffects(probe.getModel(), probe.getData(), q, v);
  const std::size_t perNonlinearEffects = robot::realtime::heapAllocationCountOnThisThread() - before;

  runModes(controller, nominal, state, action, time);  // the first time through, sizing what sizes itself on first use
  const int eventsBefore = sink.posted.load();

  std::vector<std::size_t> allocationsPerMode;
  allocationsPerMode.reserve(std::size(kModes));
  runModes(controller, nominal, state, action, time, &allocationsPerMode);
  ASSERT_EQ(allocationsPerMode.size(), std::size(kModes));
  for (size_t index = 0; index < allocationsPerMode.size(); ++index) {
    const std::size_t pinocchio = kComputesGravity[index] ? kCyclesPerMode * perNonlinearEffects : 0;
    EXPECT_EQ(allocationsPerMode[index], pinocchio)
        << kCyclesPerMode << " cycles of " << kModes[index] << " (entry included) allocated beyond Pinocchio's " << perNonlinearEffects
        << " per nonlinear effects";
  }
  EXPECT_GT(sink.posted.load(), eventsBefore) << "SAFETY's entry was reported through the sink";
  EXPECT_EQ(controller.getControlMode(), "ZERO_TORQUE");
}

}  // namespace
}  // namespace ocs2::humanoid
