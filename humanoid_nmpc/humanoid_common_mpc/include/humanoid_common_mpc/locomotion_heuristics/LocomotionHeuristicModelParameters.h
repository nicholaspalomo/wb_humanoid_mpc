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

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The properties of the ROBOT, as opposed to the tuning, that the heuristics
 * need.
 *
 * Derived once from the model at start-up and never task-file keys, for the
 * reason ContactPlanningModelParameters states for the contact planner and this
 * repository applies everywhere: a number that has to agree with the URDF must
 * not be maintained by hand in a configuration file next to it. It is also what
 * lets every heuristic be a pure function with no access to Pinocchio - see
 * LocomotionHeuristic.
 */
struct LocomotionHeuristicModelParameters {
  scalar_t totalMass = 0.0;  // [kg]
  // [m/s^2] the model's gravity (modelGravity()), the g of the capture point's
  // sqrt(z / g).
  scalar_t gravity = kGravityMagnitude;
  scalar_t totalWeight = 0.0;  // [N] totalMass * gravity, precomputed because
                               // every wrench heuristic divides by it
  /** [m] center of mass above the mean foot height in the nominal standing
   * posture: the pendulum length. */
  scalar_t nomPosComRtFeetmeanEwrtWorldZ = 0.0;
  /**
   * [m] position of each leg's hip relative to the base, in the heading frame
   * (rotated by the nominal base yaw only), at the nominal posture, with the
   * vertical component dropped: one entry per leg of ModelSettings::legs, in
   * leg order, so entry kContactLeftIndex / kContactRightIndex is the biped's
   * left / right hip.
   *
   * The landmark `hip_centered_stepping` places the foot under. Found by
   * walking up the kinematic tree from the joint that carries the leg's first
   * contact to the last joint before the floating base (hipJointOf()), which is
   * the hip however the URDF happens to name it, and falling back to that
   * contact's own horizontal position relative to the base, in the heading
   * frame, when that walk finds nothing - a fallback that is exact for a leg
   * standing vertically, which is the posture this is evaluated in. A contact
   * set without legs gets the biped's two entries from contacts 0 and 1. The
   * default is the biped's two hips at the base origin, which a hand-built
   * value overwrites by index.
   */
  std::vector<vector2_t> posHipsRtBaseEwrtHeadingXy = std::vector<vector2_t>(
      static_cast<size_t>(kBipedFootCount), vector2_t::Zero());
  /**
   * [m] position of each contact (ModelSettings::contactNames, in contact
   * order) relative to the base, in the heading frame, at the nominal posture,
   * with the vertical component dropped: where the contact stands when the
   * robot stands nominally. The `nominal_stance` anchor of
   * nominal_foothold.anchor lands a swing contact here relative to the
   * predicted base. Empty until derived.
   */
  std::vector<vector2_t> nomPosContactsRtBaseEwrtHeadingXy;

  /** One line per derived quantity, for the start-up banner. */
  std::string summary() const;
};

/**
 * Derives the model parameters from the robot model at the nominal standing
 * state.
 *
 * `nominalState` is the same default posture the rest of the interface uses
 * (the `default_joint_state` of reference.textproto on a level base), because
 * these are the constants of the gait's neighborhood rather than of any one
 * instant. The gravity is the model's (modelGravity()), the one its dynamics
 * use; InvalidArgument when it does not point down.
 */
absl::StatusOr<LocomotionHeuristicModelParameters>
deriveLocomotionHeuristicModelParameters(
    PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const vector_t& nominalState);

/**
 * Computes each contact's position [m] relative to the base at `nominalState`,
 * in the
 * heading frame (the base yaw of `nominalState` only), with the vertical
 * dropped, in contact order (LocomotionHeuristicModelParameters::
 * nomPosContactsRtBaseEwrtHeadingXy). Runs the forward kinematics of
 * `nominalState` on `pinocchioInterface`'s data.
 */
std::vector<vector2_t> computeNomPosContactsRtBaseEwrtHeadingXy(
    PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const vector_t& nominalState);

}  // namespace ocs2::humanoid
