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

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristic.h"

namespace ocs2::humanoid {

/**
 * The offset one base-pose heuristic contributes to the base-pose reference: Bledt's H_Theta on the first two members
 * and H_z on the third (Appendix C).
 *
 * Only these three channels. The remaining three of the base pose - x, y and yaw - are the operator's command
 * integrated forward, and a heuristic that moved them would be steering the robot rather than shaping how it carries
 * itself along the path it was told to take.
 *
 * The members are named rather than packed into a vector3_t because the base pose this is added to is Euler ZYX with
 * YAW FIRST, so roll is index 5, pitch is index 4 and height is index 2 - an ordering that is the reverse of the
 * (roll, pitch, yaw) every textbook writes, and one that a bare vector would invite getting wrong at each of the ten
 * call sites. The mapping is performed in exactly one place; see the base_pose_heuristic_seam directive in
 * SwitchedModelReferenceManager.cpp.
 */
// LINT.IfChange(base_pose_offset_convention)
struct BasePoseOffset {
  scalar_t roll = 0.0;    // [rad] added to base pose index 5
  scalar_t pitch = 0.0;   // [rad] added to base pose index 4
  scalar_t height = 0.0;  // [m] added to base pose index 2

  BasePoseOffset& operator+=(const BasePoseOffset& other) {
    roll += other.roll;
    pitch += other.pitch;
    height += other.height;
    return *this;
  }
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/reference_manager/SwitchedModelReferenceManager.cpp:base_pose_heuristic_seam)

/**
 * Everything a base-pose heuristic is allowed to see, evaluated at one shooting node.
 *
 * It is a snapshot deliberately: the evaluation runs per node per SQP iteration on every worker thread, so a heuristic
 * that reached back into the reference manager would be reading state another thread is free to be writing. Every
 * member here is either a constant of the solve or a value read out of the node's own reference.
 */
struct BasePoseHeuristicContext {
  /** [m/s] commanded CoM velocity at this node, rotated into the reference base's yaw frame: x forward, y left. */
  vector2_t commandedVelocityInBaseFrame = vector2_t::Zero();
  /** [rad/s] commanded yaw rate at this node. */
  scalar_t commandedYawRate = 0.0;
  /**
   * Gait phase at this node, in [0, 1), from SwitchedModelReferenceManager::getPhaseVariable().
   *
   * [0, 0.5) is the LF mode and [0.5, 1) the RF mode, and those name the foot IN CONTACT, not the one swinging:
   * modeNumber2StanceLeg maps LF to contact flags {true, false}. So the first half of the cycle is left stance with
   * the RIGHT foot in the air, and the second half is the mirror of it.
   *
   * It FREEZES at 0.0 or 0.5 through double support rather than advancing, so a periodic heuristic holds a constant
   * offset there instead of continuing its cycle.
   */
  scalar_t gaitPhase = 0.0;
};

/**
 * A heuristic that shapes the ROLL, PITCH and HEIGHT of the base-pose reference.
 *
 * This is the channel with the most to gain, because the reference it shapes is currently a hard zero:
 * TargetTrajectoriesCalculatorBase::integrateTargetBasePose writes `targetPose[4] = 0.0; targetPose[5] = 0.0;` on
 * every call, while Q's base-pose block is live and non-zero on every robot in this repository. The MPC is therefore
 * being asked to walk with the base held exactly level at every speed, which is not what a legged system wants and is
 * precisely the observation that produced Bledt's orientation-compensation heuristic (dissertation section 4.2,
 * figure 4-5).
 *
 * The offsets are added at the point of use, per node, in SwitchedModelReferenceManager::getDesiredState(), and NOT in
 * the target-trajectory calculator where the zeros are written. Three reasons, all decisive: the target trajectory has
 * three knots over the whole horizon and could not represent a limit cycle at gait frequency; it is published one
 * solve ahead of the solver that consumes it; and it is also what the operator's own delta-pose commands are measured
 * against, so shaping it would make the commanded pose drift away from what was asked for.
 */
class BasePoseHeuristic : public LocomotionHeuristic {
 public:
  ~BasePoseHeuristic() override = default;

  /** The offset this heuristic contributes at one shooting node. Must be a pure function of its coefficients and `context`. */
  virtual BasePoseOffset offset(const BasePoseHeuristicContext& context) const = 0;
};

}  // namespace ocs2::humanoid
