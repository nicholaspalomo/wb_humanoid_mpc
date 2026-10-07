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

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristic.h"

namespace ocs2::humanoid {

/**
 * Everything a foothold heuristic is allowed to see about one swing foot's landing.
 *
 * Built per touch-down by SwitchedModelReferenceManager::footholdContext() from measurements latched once per solve
 * (captureMeasuredState()), the commands at the touch-down time, and the mode schedule - see the three kinds of field
 * below. Latching the measurements also keeps the promise made in LocomotionHeuristic: no heuristic ever touches the
 * Pinocchio model, so none of them needs a per-thread copy of it.
 */
struct FootholdHeuristicContext {
  // Three kinds of field, and it matters which is which. PREDICTED: where the base will be when this foot lands,
  // because Bledt's H_r is relative to the hip at the step's own time. MEASURED at the last solve and latched: the
  // feedback quantities the capture point closes its loop on. COMMANDED at the touch-down time: what the operator asked
  // for, which the Raibert-style leads are proportional to.

  /** Which foot is landing; kContactLeftIndex or kContactRightIndex. */
  size_t contactIndex = 0;
  /**
   * +1 for the left foot and -1 for the right: the side of the robot this foot is on.
   *
   * Every lateral coefficient in this family is multiplied by it, so that one number in the task file describes both
   * feet and the two feet step to opposite sides of the body rather than both to the left.
   */
  scalar_t side = 1.0;
  /** [m] PREDICTED base position in the world at the touch-down: measured at the last solve, carried forward by the command. */
  vector2_t basePosition = vector2_t::Zero();
  /** [rad] PREDICTED base yaw at the touch-down; the heading every base-frame offset below is rotated by. */
  scalar_t baseYaw = 0.0;
  /** [m/s] MEASURED CoM linear velocity in the world at the last solve. */
  vector2_t measuredVelocity = vector2_t::Zero();
  /** [m/s] COMMANDED CoM linear velocity in the world at the touch-down time. */
  vector2_t commandedVelocity = vector2_t::Zero();
  /** [rad/s] COMMANDED yaw rate at the touch-down time. */
  scalar_t commandedYawRate = 0.0;
  /** [m] MEASURED center-of-mass height above the mean foot height at the last solve; the pendulum length. */
  scalar_t comHeight = 0.0;
  /**
   * [s] Duration of the stance this foot begins by landing, from the mode schedule; 0 when the schedule does not say.
   * Raibert's placement is half of it times the velocity, and the gait scheduler changes it with the commanded speed,
   * which is why the stepping heuristics have terms proportional to it as well as Bledt's constant ones.
   */
  scalar_t stanceDuration = 0.0;
};

/**
 * A heuristic that shapes WHERE THE SWING FOOT LANDS: Bledt's H_r (Appendix C, the r rows of both tables).
 *
 * The contribution is an additive offset, in the WORLD frame and in the ground plane, to the landing target
 * SwitchedModelReferenceManager::nominalFoothold() would otherwise return. Ground-projected because every layer of
 * this stack assumes flat ground and the landing height is owned by the swing trajectory planner, which interpolates
 * it from `terrainHeight`; a heuristic that also had an opinion about height would be a second, conflicting one.
 *
 * THREE PRECONDITIONS, all checked by LocomotionHeuristicLayer::Create() at start-up rather than discovered in
 * simulation - two refused, one warned about:
 *
 *  - NOT WITH AN ONLINE CONTACT PLANNER. ContactPlanningReferenceManager overrides getSwingFootReference() and never
 *    calls nominalFoothold(), so a foothold heuristic listed alongside `contactScheduleSource: contact_planner` would do
 *    nothing at all, silently. That is also the right answer on the merits: these heuristics ARE the analytic
 *    alternative to an optimization-based foothold planner, and running both is two opinions fighting over one
 *    variable. The combination is rejected.
 *  - SOMETHING MUST KEEP THE FEET APART. The anchor takes its lateral separation from the stance foot plus
 *    `nominal_foothold.step_width`, or - with `hip_centered_stepping` listed - from the hips. With the
 *    step width at 0 and no hip_centered_stepping, every other heuristic would correct a target on the stance foot's
 *    own lateral line and the swing foot would be aimed at the stance foot. That combination is rejected too.
 *  - THE FOOT COST'S XY WEIGHTS MUST BE NON-ZERO. `task_space_foot_cost.weights.pos_x` and `pos_y` gate this whole
 *    channel and are 0 on every robot shipped here, because with `zero_velocity` in `hard_constraints` the stance
 *    foot is pinned by the schedule and placement follows from it. A landing target multiplied by a zero weight is
 *    dead weight, so Create() warns (LocomotionHeuristicEnvironment::footPositionIsUntracked) rather than letting the
 *    operator tune coefficients that cannot move anything.
 */
class FootholdHeuristic : public LocomotionHeuristic {
 public:
  ~FootholdHeuristic() override = default;

  /**
   * The world-frame, ground-plane offset this heuristic contributes to the landing target.
   *
   * Must be a pure function of its coefficients and `context`; see LocomotionHeuristic for why.
   */
  virtual vector2_t offset(const FootholdHeuristicContext& context) const = 0;

  /**
   * True when this heuristic MOVES THE ANCHOR of the foothold reference rather than nudging it.
   *
   * Only `hip_centered_stepping` does: it replaces "a step width to the side of the stance foot" with "under this
   * foot's own hip", which is a different landmark, not a correction to the existing one. It is the term Bledt sums
   * the velocity-dependent ones on top of (dissertation figure 4-8), so the layer warns when it is listed alone and
   * when it is not listed at all while others are - the two configurations in which the sum is not the one the
   * dissertation validates.
   */
  virtual bool movesAnchor() const { return false; }
};

}  // namespace ocs2::humanoid
