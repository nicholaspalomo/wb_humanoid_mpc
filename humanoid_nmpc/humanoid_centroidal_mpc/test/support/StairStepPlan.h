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

#pragma once

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "absl/status/statusor.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"

namespace ocs2::humanoid {

/** The plan times of the hand-written stair step [s]: the lift-offs and
 * touch-downs of the two feet, and the end. */
struct StairStepTimes {
  static constexpr scalar_t kLeftLiftOff = 1.5;
  static constexpr scalar_t kLeftTouchDown = 2.5;
  static constexpr scalar_t kRightLiftOff = 4.5;
  static constexpr scalar_t kRightTouchDown = 5.5;
  static constexpr scalar_t kEnd = 7.0;
  // [s] The spacing of the knots.
  static constexpr scalar_t kKnotSpacing = 0.05;
};

/**
 * A hand-written, feet-only plan of the DRC Atlas stepping onto the step
 * "step_1" of `scenario` (test/testdata/multi_contact/stair_step_scenario
 * .textproto), for the tests of the centroidal MPC tracking a plan until the
 * stance-graph planner's stairs plan exists: from the scenario's initial pose
 * it lowers its base and moves its center of mass over the right foot, swings
 * the left foot onto the step (on the generalized swing planner's arc, clear
 * of the riser), moves its center of mass over the left foot, swings the right
 * foot up beside it, and stands up on the step. The configuration of every
 * knot is solved by inverse kinematics of the legs on the MPC's Pinocchio
 * model (`pinocchioInterface`, copied, with its contact frames
 * foot_l_contact and foot_r_contact) at a base placed so that the center of
 * mass is over the stance foot; the momenta are the configurations'
 * differences. The hashes are `scenario`'s.
 *
 * @return Internal when the inverse kinematics does not converge;
 * InvalidArgument when the scenario has no step_1 or the model lacks a foot.
 */
absl::StatusOr<multi_contact::config::MultiContactPlan> makeStairStepPlan(
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& model,
    const multi_contact::LoadedScenario& scenario);

}  // namespace ocs2::humanoid
