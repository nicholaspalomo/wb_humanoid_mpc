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

#include "humanoid_common_mpc/contact_planning/search/CadenceStretchStage.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "humanoid_common_mpc/contact_planning/logic/ContactLogicState.h"

#include "absl/log/log.h"

namespace ocs2::humanoid {
namespace {

/** Stretches closer to 1 than this change nothing worth a QP. */
constexpr scalar_t kStretchTolerance = 1e-3;

/**
 * The runs of equal contact state of one foot in an assignment, in nodes.
 *
 * The run that contains node 0 is reported separately and is NOT counted in the two longest runs, because it is the
 * only one whose real duration is not the number of nodes it occupies: the phase was already in flight when the plan
 * was made, and the gait limits apply to the whole of it. Every other run starts inside the horizon, where the nodes
 * are the phase.
 */
struct FootPhaseRuns {
  int longestContact = 0;                // longest run in contact that starts inside the horizon, in nodes
  int longestSwing = 0;                  // longest run in the air that starts inside the horizon, in nodes
  int leadingNodes = 0;                  // nodes of the run that contains node 0
  std::int8_t leadingValue = kMiqpFree;  // its contact state (kMiqpFree when node 0 is not decided)
};

FootPhaseRuns footPhaseRuns(const MiqpAssignment& assignment, int numNodes, size_t foot) {
  FootPhaseRuns runs;
  int run = 0;
  std::int8_t runValue = kMiqpFree;
  bool leading = true;
  for (int node = 0; node < numNodes; ++node) {
    const std::int8_t value = assignment[static_cast<size_t>(ContactLogicState::contactBinaryIndex(node, foot))];
    if (value != runValue || node == 0) {
      if (node > 0) leading = false;
      runValue = value;
      run = 0;
    }
    ++run;
    if (leading) {
      runs.leadingNodes = run;
      runs.leadingValue = runValue;
    } else if (runValue > 0) {
      runs.longestContact = std::max(runs.longestContact, run);
    } else if (runValue == 0) {
      runs.longestSwing = std::max(runs.longestSwing, run);
    }
  }
  return runs;
}

}  // namespace

std::string CadenceStretchStage::describe() const {
  std::ostringstream out;
  out << "re-time the incumbent by scaling the node grid, " << samples_ << " stretch(es) up to " << maxStretch_
      << " (0 samples disables the stage)";
  return out.str();
}

void CadenceStretchStage::configure(const ContactPlanningConfig& config) {
  if (config.cadenceStretch.samples < 0) {
    throw std::invalid_argument("[cadence_stretch] samples must be non-negative");
  }
  if (config.cadenceStretch.samples > 0 && config.cadenceStretch.maxStretch < 1.0) {
    throw std::invalid_argument("[cadence_stretch] maxStretch must be at least 1: a stretch below 1 shrinks the committed window");
  }
  samples_ = config.cadenceStretch.samples;
  maxStretch_ = config.cadenceStretch.maxStretch;
  // The plan's own time budget, as the re-linearisation stage uses: the branch-and-bound's plus the local search's.
  timeBudget_ = config.planner.maxSolveTime + config.eventShiftLocalSearch.maxTime;
}

scalar_t CadenceStretchStage::admissibleStretch(const ContactPlanningConfig& config,
                                                const MiqpAssignment& assignment,
                                                int numNodes,
                                                scalar_t maxStretch,
                                                const ContactPlannerInput* input) {
  const GaitLimits& limits = config.shared.gaitLimits;
  const scalar_t dt = config.planner.dt;
  scalar_t stretch = std::max(1.0, maxStretch);
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const FootPhaseRuns runs = footPhaseRuns(assignment, numNodes, foot);
    // A maximum that is already violated at s = 1 cannot be repaired by stretching further, so the bound is only ever
    // applied where the phase still fits: max(1, ...) leaves such a plan alone rather than refusing to stretch at all.
    if (limits.maxSwingDuration > 0.0 && runs.longestSwing > 0) {
      stretch = std::min(stretch, std::max(1.0, limits.maxSwingDuration / (dt * static_cast<scalar_t>(runs.longestSwing))));
    }
    if (limits.maxContactDuration > 0.0 && runs.longestContact > 0) {
      stretch = std::min(stretch, std::max(1.0, limits.maxContactDuration / (dt * static_cast<scalar_t>(runs.longestContact))));
    }
    // The phase that is already in flight gets the same limit, minus what it has already spent: only the in-horizon
    // remainder is stretched, and the seconds behind the plan's start are not re-timed by anything. Without this the
    // leading run was budgeted the whole maximum for its remainder alone, which is the reading PhaseDurationsRule
    // contradicts (it counts initialPhaseNodes(foot, true) plus the in-horizon nodes against the same limit), and the
    // executed phase came out longer than the gait limit the rest of the stack is tuned for.
    if (runs.leadingNodes > 0 && runs.leadingValue != kMiqpFree) {
      const bool leadingInContact = runs.leadingValue > 0;
      const scalar_t limit = leadingInContact ? limits.maxContactDuration : limits.maxSwingDuration;
      // The elapsed time belongs to the phase the foot is in AT PLANNING TIME. When the assignment already switches at
      // node 0 the leading run is a different phase that starts there with nothing behind it, so charging it would
      // shorten the bound for a phase that has not begun.
      const bool elapsedBelongsToLeadingRun = input != nullptr && input->contacts[foot] == leadingInContact;
      const scalar_t elapsed = elapsedBelongsToLeadingRun ? std::max(0.0, input->phaseElapsedTime[foot]) : 0.0;
      if (limit > 0.0) {
        stretch = std::min(stretch, std::max(1.0, (limit - elapsed) / (dt * static_cast<scalar_t>(runs.leadingNodes))));
      }
    }
  }
  return stretch;
}

void CadenceStretchStage::afterSearch(SearchRun& run) const {
  MiqpResult& result = *run.result;
  if (samples_ <= 0 || !result.hasIncumbent) return;

  const scalar_t upperStretch = admissibleStretch(*run.config, result.assignment, run.config->planner.numNodes, maxStretch_, run.input);
  if (upperStretch <= 1.0 + kStretchTolerance) return;

  const scalar_t dt = run.config->planner.dt;
  scalar_t bestStretch = 1.0;
  scalar_t bestObjective = result.incumbentObjective;
  OcpQpProblem bestProblem;
  OcpQpSolution bestSolution;

  for (int sample = 1; sample <= samples_; ++sample) {
    if (run.elapsedSeconds() > timeBudget_) break;
    const scalar_t stretch = 1.0 + (upperStretch - 1.0) * static_cast<scalar_t>(sample) / static_cast<scalar_t>(samples_);
    try {
      OcpQpProblem stretched = run.assembleWithGrid(stretch * dt);
      OcpQpSolution solution;
      scalar_t objective = 0.0;
      if (!run.miqp->solveFixed(stretched, *run.binaries, result.assignment, *run.propagate, *run.assignmentCost, solution, objective)) {
        continue;
      }
      run.statistics->totalQpIterations += solution.iterations;
      if (objective < bestObjective) {
        bestObjective = objective;
        bestStretch = stretch;
        bestProblem = std::move(stretched);
        bestSolution = std::move(solution);
      }
    } catch (const std::exception& e) {
      LOG(ERROR) << "[LipContactPlanner] cadence stretch failure: " << e.what();
      break;
    }
  }

  if (bestStretch <= 1.0 + kStretchTolerance) return;
  *run.problem = std::move(bestProblem);
  result.solution = std::move(bestSolution);
  result.incumbentObjective = bestObjective;
  // The plan is emitted on the stretched grid; ContactPlan carries its own node duration.
  run.chosenDt = bestStretch * dt;
  if (run.verbose) {
    LOG(INFO) << "[LipContactPlanner] cadence stretched by " << bestStretch << " to dt " << run.chosenDt;
  }
}

}  // namespace ocs2::humanoid
