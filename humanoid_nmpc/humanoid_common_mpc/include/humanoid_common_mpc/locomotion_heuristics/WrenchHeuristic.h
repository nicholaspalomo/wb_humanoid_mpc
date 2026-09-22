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

/** Everything a contact-wrench heuristic is allowed to see, evaluated at one shooting node. */
struct WrenchHeuristicContext {
  /** Which feet the mode schedule calls stance feet at this node. */
  contact_flag_t contactFlags = makeFeetArray(false);
  /** How many of them there are; never zero when a heuristic is called, because a flight phase has no stance foot to load. */
  size_t numStanceFeet = 0;
  /**
   * Per-foot stance duty factor over the horizon, in (0, 1]: the fraction of the scheduled gait cycle this foot is on
   * the ground. 1 while the foot never leaves the ground, which is what standing gives.
   */
  feet_array_t<scalar_t> stanceDutyFactor = makeFeetArray(1.0);
  /** [m/s] commanded CoM linear velocity in the world at this node. */
  vector2_t commandedVelocity = vector2_t::Zero();
  /** [rad/s] commanded yaw rate at this node. */
  scalar_t commandedYawRate = 0.0;
  /** [N] the robot's weight, mass * gravity: the scale every force offset here is naturally measured in. */
  scalar_t totalWeight = 0.0;
};

/**
 * A heuristic that shapes the CONTACT FORCE REFERENCE: Bledt's H_f (Appendix C, the f rows of Table C.1).
 *
 * The reference being shaped is weight compensation and nothing else today - the robot's weight divided evenly over
 * the feet the mode schedule calls stance feet, purely vertical - and it reaches the solver through
 * InputQuadraticCost and StateInputQuadraticCost, which both build it with weightCompensatingInput() and therefore
 * both ignore the input channel of the target trajectory entirely. The layer is inserted by routing those two costs
 * through SwitchedModelReferenceManager::getDesiredInput(), which returns exactly that same vector when the list is
 * empty.
 *
 * The contribution is an additive force in the WORLD frame, per stance foot. World frame rather than the input's own
 * frame because that is the frame Bledt's formulae are written in and the one m*omega x v means anything in; the
 * conversion is the model's job, through MpcRobotModelBase::setContactForceInWorldFrame(), which is the identity for
 * the direct-wrench parameterization and a thread-safe forward-kinematics rotation for the basis-vector one.
 *
 * WHY producesHorizontalForce() EXISTS. Both cost terms carry a comment explaining that they deliberately use the
 * cheap input-only weightCompensatingInput() overload "because the nominal contact force is purely vertical and a
 * stance foot is flat, so it is identical in the world and local contact frames (yaw-invariant). The state-aware
 * overload would add a forward-kinematics pass per node and iteration to this hot path for no practical gain." That
 * argument is exactly right for a vertical force and exactly wrong for a horizontal one, whose expression in the
 * local contact frame depends on the foot's yaw. So the layer asks each heuristic which kind of force it makes, and
 * pays for the state-aware path only when one of them actually needs it - which, of the ten in the dissertation, is
 * `centripetal_acceleration` alone.
 */
class WrenchHeuristic : public LocomotionHeuristic {
 public:
  ~WrenchHeuristic() override = default;

  /**
   * The world-frame force this heuristic adds to the reference of one stance foot, in newtons.
   *
   * Called only for feet `context.contactFlags` reports in contact, so an implementation never has to guard against
   * dividing by a zero stance count.
   */
  virtual vector3_t forceOffset(const WrenchHeuristicContext& context, size_t contactIndex) const = 0;

  /**
   * True when forceOffset() can return a force with a non-zero horizontal component.
   *
   * It decides which of the two weightCompensatingInput() overloads the whole layer uses, so it is a property of the
   * FORMULA and not of the current coefficients: a heuristic whose gain happens to be zero today must still answer
   * truthfully, or hot-reloading a non-zero gain would silently start writing a horizontal force down the vertical-only
   * path and the basis-vector parameterization would receive it in the wrong frame.
   */
  virtual bool producesHorizontalForce() const { return false; }
};

}  // namespace ocs2::humanoid
