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

#include <atomic>
#include <cstddef>
#include <iterator>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/rnea.hpp"

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h"
#include "humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h"
#include "humanoid_common_mpc/mrt/ControllerEventSink.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/NullMpcLink.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

/*
 * What the centroidal MRT joint controller does on the realtime thread of the robot process adds no allocation of its
 * own once it is built: the observation of every cycle, the passive modes (ZERO_TORQUE, JOINT_PD, GRAVITY_COMP, SAFETY,
 * each entry included) and the hold of WB_MPC until a policy arrives. A cycle used to make a dozen (the joint vectors
 * of the robot state, the contact flags, the gravity compensation's workspace, a std::function), and SAFETY's first
 * cycle resized its posture.
 *
 * What remains is Pinocchio's: a pass over the model's composite base joint (the Euler-angle floating base) makes a
 * temporary of the joint's dynamic-size motion subspace - the centroidal map of the observation, the nonlinear effects
 * of the gravity compensation. The test measures those passes on the same model and expects exactly them, so that the
 * controller's own code is pinned at zero whatever Pinocchio does. The policy's execution in WB_MPC is not covered: it
 * still allocates in the model's accessors and the inverse dynamics (the package README).
 */

namespace ocs2::humanoid {
namespace {

/** A sink that counts, as the realtime event log takes events: a copy, no allocation. */
class CountingSink final : public ControllerEventSink {
 public:
  bool post(const ControllerEvent& /*event*/) override {
    posted.fetch_add(1);
    return true;
  }
  std::atomic<int> posted{0};
};

/** The modes the robot process hands the controller, in turn: every entry, and the hold of WB_MPC. */
constexpr const char* absl_nonnull kModes[] = {"ZERO_TORQUE", "JOINT_PD", "GRAVITY_COMP", "SAFETY", "JOINT_PD", "WB_MPC", "ZERO_TORQUE"};
/** Whether a mode's action needs the gravity compensation (the WB_MPC hold is JOINT_PD's). */
constexpr bool kComputesGravity[] = {false, true, true, false, true, true, false};
constexpr int kCyclesPerMode = 20;

/** What the robot process calls every cycle - the mode and the posture, then the action - through kModes. */
void runModes(CentroidalMpcMrtJointController& controller,
              const std::vector<scalar_t>& nominal,
              robot::model::RobotState& state,
              robot::model::RobotJointAction& action,
              scalar_t& time,
              std::vector<size_t>* absl_nullable allocationsPerMode = nullptr) {
  for (const char* absl_nonnull mode : kModes) {
    const size_t before = robot::realtime::heapAllocationCountOnThisThread();
    for (int cycle = 0; cycle < kCyclesPerMode; ++cycle) {
      controller.setControlMode(mode);
      controller.setNominalJointPositions(nominal);
      time += 0.01;
      state.setTime(time);
      controller.computeJointControlAction(time, state, action);
    }
    if (allocationsPerMode != nullptr) allocationsPerMode->push_back(robot::realtime::heapAllocationCountOnThisThread() - before);
  }
}

TEST(CentroidalMpcMrtJointControllerAllocations, TheObservationThePassiveModesAndTheHoldAddNoAllocationOfTheirOwn) {
  CentroidalTestingModelInterface model;
  absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus = robot::model::RobotDescription::Create(model.urdfFile);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  robot::model::RobotState state(description);
  state.setRootPositionInWorldFrame(vector3_t(0.0, 0.0, 0.75));
  robot::model::RobotJointAction action(description);
  absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>> controllerOrStatus = CentroidalMpcMrtJointController::Create(
      description, model.getModelSettings(), model.getMpcRobotModel(), test_support::NullMpcLink::factory(), model.getPinocchioInterface());
  ASSERT_TRUE(controllerOrStatus.ok()) << controllerOrStatus.status();
  CentroidalMpcMrtJointController& controller = **controllerOrStatus;
  CountingSink sink;
  controller.setEventSink(&sink);
  const std::vector<scalar_t> nominal(description.getNumJoints(), 0.1);
  scalar_t time = 0.0;

  // Pinocchio's own allocations, one pass each on a copy of the controller's model.
  PinocchioInterface probe = model.getPinocchioInterface();
  const vector_t q = vector_t::Zero(probe.getModel().nq);
  const vector_t v = vector_t::Zero(probe.getModel().nv);
  pinocchio::computeCentroidalMap(probe.getModel(), probe.getData(), q);
  pinocchio::nonLinearEffects(probe.getModel(), probe.getData(), q, v);
  size_t before = robot::realtime::heapAllocationCountOnThisThread();
  pinocchio::computeCentroidalMap(probe.getModel(), probe.getData(), q);
  const size_t perCentroidalMap = robot::realtime::heapAllocationCountOnThisThread() - before;
  before = robot::realtime::heapAllocationCountOnThisThread();
  pinocchio::nonLinearEffects(probe.getModel(), probe.getData(), q, v);
  const size_t perNonlinearEffects = robot::realtime::heapAllocationCountOnThisThread() - before;

  runModes(controller, nominal, state, action, time);  // the first time through, sizing what sizes itself on first use
  const int eventsBefore = sink.posted.load();

  std::vector<size_t> allocationsPerMode;
  allocationsPerMode.reserve(std::size(kModes));
  runModes(controller, nominal, state, action, time, &allocationsPerMode);
  ASSERT_EQ(allocationsPerMode.size(), std::size(kModes));
  for (size_t index = 0; index < allocationsPerMode.size(); ++index) {
    const size_t pinocchio = kCyclesPerMode * (perCentroidalMap + (kComputesGravity[index] ? perNonlinearEffects : 0));
    EXPECT_EQ(allocationsPerMode[index], pinocchio)
        << kCyclesPerMode << " cycles of " << kModes[index] << " (entry included) allocated beyond Pinocchio's " << perCentroidalMap
        << " per centroidal map and " << perNonlinearEffects << " per nonlinear effects";
  }
  EXPECT_GT(sink.posted.load(), eventsBefore) << "SAFETY's entry was reported through the sink";
  EXPECT_EQ(controller.getControlMode(), "ZERO_TORQUE");
}

}  // namespace
}  // namespace ocs2::humanoid
