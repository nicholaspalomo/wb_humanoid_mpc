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

#include "humanoid_common_mpc/contact_planning/search/SearchStage.h"

namespace ocs2::humanoid {

/**
 * `cadence_stretch`: re-time the whole incumbent by scaling the node grid, recovering the cadences the grid's
 * quantisation cannot express.
 *
 * A phase of the plan lasts a whole number of nodes, so the stride the planner can emit is quantised by `planner.dt`:
 * at dt 0.1 with swing limits [0.4, 0.5] a swing is four or five nodes and nothing in between, a 25% jump in the step
 * the commanded speed needs. Under a *fixed* contact pattern, though, the cadence simply is the scale of the grid, so
 * re-solving the same pattern on a grid of node duration `s * dt` re-times every phase of the plan together. Every
 * term is an exact function of the node duration (the LIP block is cosh / sinh of omega dt, the heading block is
 * linear in it, the nominal step displacement is a ratio of it), and ContactPlan carries its own dt, so a re-timed
 * plan needs no new representation - which is what `SearchRun::assembleWithGrid` and `SearchRun::chosenDt` exist for.
 *
 * Two qualifications on that exactness, both of which used to be missing and both of which are properties of the
 * caller rather than of this stage. The LINEARISATION POINT is not a function of the node duration: the nominal
 * heading trajectory the frame terms are linearised around is indexed by node, so it has to be rebuilt for the
 * candidate grid, or the terms that read `ctx.dt` end up on one clock and the terms that read `ctx.nominal` on
 * another (LipContactPlanner::plan binds `assembleWithGrid` so that it is). And "no new representation" holds only as
 * long as the rest of the planner accepts a plan whose dt is not planner.dt: LipContactPlanner::previousPlanShift
 * measures the shift on the stored plan's own grid for exactly that reason, since rejecting it there would leave the
 * warm start, both consistency terms and the previous-plan nominal inert on every cycle after a stretch.
 *
 * The stage evaluates `samples` stretches spread over the admissible range and keeps the best. The range starts at 1:
 * a stretch below 1 shrinks the committed window below `planner.commitTime` and the horizon below `mpc.timeHorizon`,
 * which makes the merge pad the tail with STANCE and throws away the last steps' anticipation. It ends at
 * `maxStretch`, clipped so that no phase of the incumbent is stretched past the gait limits.
 */
class CadenceStretchStage final : public SearchStage {
 public:
  std::string describe() const override;
  std::vector<std::string> requiredBlocks() const override { return {}; }
  void configure(const ContactPlanningConfig& config) override;
  void afterSearch(SearchRun& run) const override;

  /**
   * The largest stretch that keeps every phase of `assignment` within the gait limits, never below 1. Public for the
   * test: this is the bound that makes the stage safe to enable, and it is the only part of it that is not arithmetic.
   *
   * `input` carries the part of the phase active at planning time that has ALREADY elapsed, and the bound is wrong
   * without it. The gait limits are limits on a whole phase, and PhaseDurationsRule enforces them as
   * initialPhaseNodes(foot, true) plus the nodes of the phase inside the horizon; the run of the assignment that
   * contains node 0 is only that second part. Budgeting the entire maximum for the remainder - which is what this
   * function did, because it never saw the elapsed time - returns a stretch the total duration cannot absorb: at dt
   * 0.1 with maxSwingDuration 0.6, a foot 0.4 s into its swing and two swing nodes left in the horizon, the bound was
   * 0.6 / 0.2 = 3.0 and the stage was free to stretch to its configured maximum, executing a 0.4 + 0.25 = 0.65 s swing
   * against a 0.6 s limit that the swing trajectory planner and the whole-body MPC are tuned against. Nothing
   * downstream re-checks it: solveFixed re-runs the propagation over a ContactLogicState built on the unstretched
   * planner.dt, and mergeModeSchedules keeps the past lift-off from the applied schedule while taking the stretched
   * touch-down from the plan. A null `input` keeps the old, optimistic reading for callers that have no input to give
   * (the bound then assumes nothing has elapsed); afterSearch always passes SearchRun::input.
   *
   * The run that ends at the horizon's end needs no such correction: ContactPlan::toModeSchedule() closes an open
   * swing at endTime(), so the duration it emits is exactly its in-horizon nodes times the stretched node duration.
   */
  static scalar_t admissibleStretch(const ContactPlanningConfig& config,
                                    const MiqpAssignment& assignment,
                                    int numNodes,
                                    scalar_t maxStretch,
                                    const ContactPlannerInput* input = nullptr);

 private:
  int samples_ = 0;
  scalar_t maxStretch_ = 1.25;
  scalar_t timeBudget_ = 0.0;
};

}  // namespace ocs2::humanoid
