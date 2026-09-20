/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <memory>

#include <ocs2_core/constraint/StateInputConstraint.h>

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/FootprintCornerHeights.h"

namespace ocs2::humanoid {

/**
 * The relaxed contact complementarity of one foot: g(x, u) = f_n(u) * h(x), the normal contact force times the height
 * of the foot above the terrain.
 *
 * This is what makes the whole-body MPC contact-implicit in the sense of "Reduced-Order Model Guided Contact-Implicit
 * Model Predictive Control for Humanoid Locomotion" (arXiv:2502.15630). The paper's CI-MPC decides contact implicitly
 * because its inverse-dynamics formulation makes the contact force a smooth function of the configuration under a
 * compliant contact model. Here the contact wrenches are decision variables instead, so the same freedom is obtained
 * by dropping the mode-scheduled hard constraints (ZeroWrenchConstraint on a swinging foot, the stance zero-velocity
 * constraint) and asking the optimizer for the complementarity condition of rigid contact directly:
 *
 *   f_n >= 0,   h >= 0,   f_n h = 0,
 *
 * of which the first is already enforced (the friction cone, or the non-negativity of the basis scalings), the second
 * is GroundPenetrationConstraint and the third is this term, penalised rather than imposed. A foot may then carry load
 * only where it touches the ground, and where it touches the ground it may carry load whatever the nominal gait says.
 * The contact schedule from the reduced-order planner survives only as a reference for the swing-foot cost, which is
 * exactly the role the paper gives it.
 *
 * The term is bilinear - the force enters linearly through the model's contact parameterization and the height enters
 * through the foot kinematics - so its linear approximation is assembled in closed form from the end-effector
 * kinematics and the constant row that maps the input to the normal force. No automatic differentiation is needed.
 *
 * The normal direction is the sole's, i.e. the third component of the model's contact force. For the flat terrain the
 * reduced-order planner assumes, that is the world vertical; on a tilted foot it is the physically correct normal.
 *
 * WHICH HEIGHT. h is the GAP - the height of the LOWEST point of the footprint above the terrain - and not the height
 * of the sole's centre. The two differ whenever the foot is pitched or rolled, which this formulation makes the normal
 * case rather than an exceptional one: ForceWeightedSlipConstraint deliberately leaves the rocking rates free so that
 * the foot can roll from heel to toe under load. Measured at the centre, a foot up on its heel reads a positive height
 * while it is carrying the whole robot, and this term then penalises the force it is physically holding - it pays for
 * a contact that exists. It also disagreed with GroundPenetrationConstraint, which had already been moved to the
 * corners; the two now share one FootprintCornerHeights, so they cannot disagree again.
 *
 * The gap is the smoothMinimumHeight() of the corner heights rather than their exact minimum, because the exact
 * minimum is non-differentiable precisely at the flat-footed stance where the robot spends most of its time; see that
 * function for why the normalisation inside it is not cosmetic.
 *
 * The residual is normalised - it is (f_n / f_ref) (h / h_ref), not f_n h - and that matters more than it looks. The
 * penalty wrapped around this term is quadratic, so what the solver actually sees is a curvature of
 *
 *   d2/dh2 [ w g^2 / 2 ] = w f_n^2 / f_ref^2 / h_ref^2,
 *
 * proportional to the square of the normal force. Unnormalised, on a 160 kg robot, f_n^2 spans nine orders of
 * magnitude between a foot in flight and a foot carrying the whole body, so one weight cannot be right at both ends:
 * chosen for the loaded foot it is negligible in flight, and chosen for the flight foot it dwarfs every other term in
 * the problem. It dwarfs, in particular, the swing height reference - and since f_n h = 0 is satisfied just as well by
 * pressing the foot down as by unloading it, an oversized weight buys its reduction by landing the foot early rather
 * than by taking the force off it.
 *
 * Dividing by a reference force and a reference height makes the residual dimensionless and O(1) at the worst
 * configuration the robot can reach - a foot at full swing height carrying full body weight - so the weight means the
 * same thing at every point of the horizon and is directly comparable with the task-space weights it competes against.
 */
class ContactComplementarityConstraint final : public StateInputConstraint {
 public:
  /**
   * @param [in] cornerHeights : the points of this foot whose lowest height is the gap. Pass the same object given to
   *        this foot's GroundPenetrationConstraint.
   * @param [in] mpcRobotModel : the robot model, for the input block that carries this contact's force.
   * @param [in] contactPointIndex : the contact this term belongs to.
   * @param [in] terrainHeight : [m] height of the ground under the foot.
   * @param [in] forceReference : [N] the normal force the residual is measured in, normally the robot's weight.
   * @param [in] heightReference : [m] the height the residual is measured in, normally the swing apex.
   * @param [in] gapSmoothing : [m] the length scale over which the corners' minimum is smoothed.
   */
  ContactComplementarityConstraint(const FootprintCornerHeights& cornerHeights,
                                   const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                   size_t contactPointIndex,
                                   scalar_t terrainHeight = 0.0,
                                   scalar_t forceReference = 1.0,
                                   scalar_t heightReference = 1.0,
                                   scalar_t gapSmoothing = 1.0e-3);

  ~ContactComplementarityConstraint() override = default;
  ContactComplementarityConstraint* clone() const override { return new ContactComplementarityConstraint(*this); }

  size_t getNumConstraints(scalar_t time) const override { return 1; }
  vector_t getValue(scalar_t time, const vector_t& state, const vector_t& input, const PreComputation& preComp) const override;
  VectorFunctionLinearApproximation getLinearApproximation(scalar_t time,
                                                           const vector_t& state,
                                                           const vector_t& input,
                                                           const PreComputation& preComp) const override;

  void setTerrainHeight(scalar_t terrainHeight) { terrainHeight_ = terrainHeight; }
  scalar_t getTerrainHeight() const { return terrainHeight_; }
  scalar_t getForceReference() const { return 1.0 / inverseForceReference_; }
  scalar_t getHeightReference() const { return 1.0 / inverseHeightReference_; }
  /** Retunes the height the residual is measured in, so that the key is live in the tuning dashboard like the weight
   * beside it. The force reference has no setter on purpose: it is the robot's weight, not a tuning parameter. */
  void setHeightReference(scalar_t heightReference) { inverseHeightReference_ = 1.0 / heightReference; }
  scalar_t getGapSmoothing() const { return gapSmoothing_; }
  void setGapSmoothing(scalar_t gapSmoothing);
  /** [m] the gap between the lowest point of the footprint and the terrain, as this term sees it. */
  scalar_t getGap(const vector_t& state) const;
  /** The constant row with f_n = normalForceRow . u; exposed for the tests. */
  const vector_t& getNormalForceRow() const { return normalForceRow_; }

 private:
  ContactComplementarityConstraint(const ContactComplementarityConstraint& rhs);

  std::unique_ptr<FootprintCornerHeights> cornerHeightsPtr_;
  size_t contactPointIndex_;
  scalar_t terrainHeight_;
  scalar_t gapSmoothing_;
  // Stored as reciprocals: the residual and its two derivative blocks each need the scale, and a multiplication in the
  // solver's inner loop is cheaper than a division.
  scalar_t inverseForceReference_;
  scalar_t inverseHeightReference_;
  vector_t normalForceRow_;
};

/** The row that maps the input to the normal force of `contactPointIndex`, probed from the model's linear accessor. */
vector_t normalContactForceRow(const MpcRobotModelBase<scalar_t>& mpcRobotModel, size_t contactPointIndex);

}  // namespace ocs2::humanoid
