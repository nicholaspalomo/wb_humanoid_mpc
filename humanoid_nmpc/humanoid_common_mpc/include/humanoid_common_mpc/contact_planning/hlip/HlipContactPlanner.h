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

#include <string>
#include <vector>

#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerInterface.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/hlip/HlipModel.h"
#include "humanoid_common_mpc/contact_planning/hlip/HlipStandingBlend.h"

namespace ocs2::humanoid {

/**
 * The reduced-order layer of "Reduced-Order Model Guided Contact-Implicit Model Predictive Control for Humanoid
 * Locomotion" (Esteban, Kurtz, Ghansah, Ames, arXiv:2502.15630): a closed-form contact planner on the Hybrid Linear
 * Inverted Pendulum. See humanoid_nmpc/docs/hlip_contact_planner/README.md.
 *
 * There is no optimization in this planner and nothing is searched. A plan is three steps:
 *
 *  1. The cadence is fixed. Single support phases of `hlip.sspDuration` alternate between the feet, separated by
 *     double supports of `hlip.dspDuration`, continuing the phase the robot is executing at the planning instant. The
 *     contact sequence is therefore decided before the first number is computed, which is exactly what the paper asks
 *     of the reduced-order model: propose a nominal gait, and leave the whole-body controller free to depart from it.
 *  2. Each foothold is the H-LIP deadbeat step. The reduced model is rolled forward to the pre-impact state of the
 *     step, and the step length follows from u = u* + K (x - x*), with the nominal orbit of the commanded velocity
 *     (a period-one orbit along the heading, a period-two orbit laterally) and the closed-form gain K of HlipModel.
 *     No gain and no weight is tuned; the clip to the reachable step is the only bound applied to the result.
 *  3. Standing and walking are blended by the paper's alpha(phi) (HlipStandingBlend): below a half the planner emits
 *     a standing plan and the feet stay where they are, above it the commanded velocity is scaled by alpha, so the
 *     stride grows continuously out of standing.
 *
 * What this replaces: the mixed-integer program of LipContactPlanner and everything that surrounds it - the branch
 * and bound, the diving and local search stages, the fifteen costs, the reachability, support region and separation
 * constraints, the combinatorial phase-duration and alternation rules, and the execution heuristics that corrected
 * its plans after the fact. None of them are needed once the whole-body MPC, and not the planner, owns the contact
 * decision.
 *
 * The plan is a ContactPlan on the node grid of `planner.dt`, like the mixed-integer planner's, so everything
 * downstream is unchanged. The cadence is continuous but the mode schedule is not: a phase boundary is rounded to
 * the nearest node, so `planner.dt` should be small (0.025 s is a millisecond-scale plan, since nothing is solved).
 */
class HlipContactPlanner final : public ContactPlannerInterface {
 public:
  /** One phase of the nominal gait. `swingFoot` is -1 in double support and while standing. */
  struct GaitPhase {
    scalar_t startTime = 0.0;
    scalar_t endTime = 0.0;
    contact_flag_t contacts = makeFeetArray(true);
    int swingFoot = -1;

    scalar_t duration() const { return endTime - startTime; }
    bool isSingleSupport() const { return swingFoot >= 0; }
  };

  /** The configuration must already be valid (ContactPlanningConfig::validate()). */
  explicit HlipContactPlanner(ContactPlanningConfig config);

  ContactPlan plan(const ContactPlannerInput& input) override;
  void setConfig(const ContactPlanningConfig& config) override;
  /** The planner keeps nothing between plans, so this does nothing; it exists for the interface. */
  void reset() override {}
  std::string getFormulationSummary() const override { return formulationSummary(config_); }

  /** The same summary for a configuration, without building a planner. */
  static std::string formulationSummary(const ContactPlanningConfig& config);

  /**
   * The lateral step the deadbeat law demands of the first step out of a standstill, in metres.
   *
   * This is the quantity that decides whether the gait can start at all, and it is not obvious from the cadence. A
   * robot standing still has its centre of mass half a step width from either foot with no lateral velocity, so once
   * the first foot leaves the ground it falls sideways fast: at the shipped height it reaches the pre-impact instant
   * far outside the orbit, and catching it takes a step much wider than the nominal one. If that step does not fit
   * inside `hlip.maxStepWidth` it is clipped, a clipped step is no longer the deadbeat step, the lateral pendulum
   * amplifies what is left by cosh(omega * sspDuration) every step, and the gait locks into an alternating wide and
   * narrow limit cycle it can never leave - the feet come together, the robot walks itself sideways and falls.
   *
   * Shortening the single support is what shrinks this: less time falling before the foot lands. It drops from 0.55 m
   * at a 0.35 s single support to 0.42 m at 0.25 s, which is the difference between not fitting and fitting.
   */
  static scalar_t startUpLateralStep(const ContactPlanningConfig& config);

  const ContactPlanningConfig& getConfig() const { return config_; }
  const HlipModel& getModel() const { return model_; }
  const HlipStandingBlend& getBlend() const { return blend_; }

  // The following are public for the tests.

  /** Whether the blend asks for a stepping gait at this input. */
  bool isWalking(const ContactPlannerInput& input) const;

  /**
   * The nominal gait over the planning horizon, continuing the phase the robot is executing at `input.time`. A
   * standing input gives a single double support phase covering the horizon.
   */
  std::vector<GaitPhase> buildGait(const ContactPlannerInput& input, bool walking) const;

  /** The foot that swings next, from the last swung foot when it is known and from the phase timing otherwise. */
  static size_t nextSwingFoot(const ContactPlannerInput& input);

 private:
  /** The H-LIP of a configuration. */
  static HlipModel makeModel(const ContactPlanningConfig& config);

  /** The heading at `time`, the commanded yaw rate integrated from the planning instant. */
  scalar_t headingAt(const ContactPlannerInput& input, scalar_t time, scalar_t blendWeight) const;

  /**
   * The deadbeat step that places `swingFoot`, in the heading frame: the pre-impact state of the step per axis, the
   * commanded velocity in that frame, and the foot being placed. `clipped` reports whether the reachable region cut
   * the step, which costs the deadbeat property and is counted onto the plan.
   */
  vector2_t deadbeatStep(const HlipModel::State& preImpactX,
                         const HlipModel::State& preImpactY,
                         const vector2_t& commandedVelocity,
                         size_t swingFoot,
                         bool& clipped) const;

  ContactPlanningConfig config_;
  HlipModel model_;
  HlipStandingBlend blend_;
};

}  // namespace ocs2::humanoid
