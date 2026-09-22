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
 * Unlike the base-pose context this is latched once per solve rather than per node, because every quantity in it is a
 * MEASUREMENT: Bledt's stepping heuristics are feedback laws on where the robot actually is and how fast it is
 * actually going (dissertation section 4.3), not functions of the plan. Latching them in
 * SwitchedModelReferenceManager::captureMeasuredState() also keeps the promise made in LocomotionHeuristic: no
 * heuristic ever touches the Pinocchio model, so none of them needs a per-thread copy of it.
 */
struct FootholdHeuristicContext {
  /** Which foot is landing; CONTACT_LEFT_INDEX or CONTACT_RIGHT_INDEX. */
  size_t contactIndex = 0;
  /**
   * +1 for the left foot and -1 for the right: the side of the robot this foot is on.
   *
   * Every lateral coefficient in this family is multiplied by it, so that one number in the task file describes both
   * feet and the two feet step to opposite sides of the body rather than both to the left.
   */
  scalar_t side = 1.0;
  /** [m] measured base position in the world at the last solve. */
  vector2_t measuredBasePosition = vector2_t::Zero();
  /** [rad] measured base yaw at the last solve; the heading every base-frame offset below is rotated by. */
  scalar_t measuredBaseYaw = 0.0;
  /** [m/s] measured CoM linear velocity in the world at the last solve. */
  vector2_t measuredVelocity = vector2_t::Zero();
  /** [m/s] commanded CoM linear velocity in the world. */
  vector2_t commandedVelocity = vector2_t::Zero();
  /** [rad/s] commanded yaw rate. */
  scalar_t commandedYawRate = 0.0;
  /** [m] measured centre-of-mass height above the mean foot height at the last solve; the pendulum length. */
  scalar_t comHeight = 0.0;
};

/**
 * A heuristic that shapes WHERE THE SWING FOOT LANDS: Bledt's H_r (Appendix C, the r rows of both tables).
 *
 * The contribution is an additive offset, in the WORLD frame and in the ground plane, to the landing target
 * SwitchedModelReferenceManager::nominalFoothold() would otherwise return. Ground-projected because every layer of
 * this stack assumes flat ground and the landing height is owned by the swing trajectory planner, which interpolates
 * it from `terrainHeight`; a heuristic that also had an opinion about height would be a second, conflicting one.
 *
 * TWO PRECONDITIONS, both checked at construction rather than discovered in simulation:
 *
 *  - NOT WITH AN ONLINE CONTACT PLANNER. ContactPlanningReferenceManager overrides getSwingFootReference() and never
 *    calls nominalFoothold(), so a foothold heuristic listed alongside `useContactPlanning: true` would do nothing at
 *    all, silently. That is also the right answer on the merits: these heuristics ARE the analytic alternative to an
 *    optimization-based foothold planner, and running both is two opinions fighting over one variable. The
 *    combination is rejected.
 *  - THE FOOT COST'S XY WEIGHTS MUST BE NON-ZERO. `task_space_foot_cost_weights.pos_x` and `pos_y` gate this whole
 *    channel and are 0 on every robot shipped here, because with `zero_velocity` in `hard_constraints` the stance
 *    foot is pinned by the schedule and placement follows from it. A landing target multiplied by a zero weight is
 *    dead weight, so the layer warns rather than letting the operator tune coefficients that cannot move anything.
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
