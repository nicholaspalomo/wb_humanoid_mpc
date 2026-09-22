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

#include <string>

#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "absl/status/statusor.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The properties of the ROBOT, as opposed to the tuning, that the heuristics need.
 *
 * Derived once from the model at start-up and never task-file keys, for the reason
 * ContactPlanningModelParameters states for the contact planner and this repository applies everywhere: a number that
 * has to agree with the URDF must not be maintained by hand in a YAML file next to it. It is also what lets every
 * heuristic be a pure function with no access to Pinocchio - see LocomotionHeuristic.
 */
struct LocomotionHeuristicModelParameters {
  scalar_t totalMass = 0.0;    // [kg]
  scalar_t gravity = 9.81;     // [m/s^2]
  scalar_t totalWeight = 0.0;  // [N] totalMass * gravity, precomputed because every wrench heuristic divides by it
  /** [m] centre of mass above the mean foot height in the nominal standing posture: the pendulum length. */
  scalar_t nominalComHeight = 0.0;
  /**
   * [m] position of each leg's hip in the BASE frame, at the nominal posture, with the vertical component dropped.
   *
   * The landmark `hip_centered_stepping` places the foot under. Found by walking up the kinematic tree from the joint
   * that carries the contact frame to the last joint before the floating base, which is the hip however the URDF
   * happens to name it, and falling back to the contact frame's own horizontal position in the base frame when that
   * walk finds nothing - a fallback that is exact for a leg standing vertically, which is the posture this is
   * evaluated in.
   */
  feet_array_t<vector2_t> hipPositionInBaseFrame = makeFeetArray(vector2_t(vector2_t::Zero()));

  /** One line per derived quantity, for the start-up banner. */
  std::string summary() const;
};

/**
 * Derives the model parameters from the robot model at the nominal standing state.
 *
 * `nominalState` is the same default posture the rest of the interface uses (the `defaultJointState` of reference.yaml
 * on a level base), because these are the constants of the gait's neighbourhood rather than of any one instant.
 */
absl::StatusOr<LocomotionHeuristicModelParameters> deriveLocomotionHeuristicModelParameters(
    PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const vector_t& nominalState,
    scalar_t gravity = 9.81);

}  // namespace ocs2::humanoid
