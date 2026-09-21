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

/** The longest run of equal contact state of `foot` in the assignment, in nodes, for the contact and the swing state. */
std::pair<int, int> longestPhases(const MiqpAssignment& assignment, int numNodes, size_t foot) {
  int longestContact = 0;
  int longestSwing = 0;
  int run = 0;
  std::int8_t runValue = -1;
  for (int node = 0; node < numNodes; ++node) {
    const std::int8_t value = assignment[static_cast<size_t>(ContactLogicState::contactBinaryIndex(node, foot))];
    if (value != runValue) {
      runValue = value;
      run = 0;
    }
    ++run;
    if (runValue > 0) {
      longestContact = std::max(longestContact, run);
    } else if (runValue == 0) {
      longestSwing = std::max(longestSwing, run);
    }
  }
  return {longestContact, longestSwing};
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
                                                scalar_t maxStretch) {
  const GaitLimits& limits = config.shared.gaitLimits;
  const scalar_t dt = config.planner.dt;
  scalar_t stretch = std::max(1.0, maxStretch);
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const std::pair<int, int> phases = longestPhases(assignment, numNodes, foot);
    // A maximum that is already violated at s = 1 cannot be repaired by stretching further, so the bound is only ever
    // applied where the phase still fits: max(1, ...) leaves such a plan alone rather than refusing to stretch at all.
    if (limits.maxSwingDuration > 0.0 && phases.second > 0) {
      stretch = std::min(stretch, std::max(1.0, limits.maxSwingDuration / (dt * static_cast<scalar_t>(phases.second))));
    }
    if (limits.maxContactDuration > 0.0 && phases.first > 0) {
      stretch = std::min(stretch, std::max(1.0, limits.maxContactDuration / (dt * static_cast<scalar_t>(phases.first))));
    }
  }
  return stretch;
}

void CadenceStretchStage::afterSearch(SearchRun& run) const {
  MiqpResult& result = *run.result;
  if (samples_ <= 0 || !result.hasIncumbent) return;

  const scalar_t upperStretch = admissibleStretch(*run.config, result.assignment, run.config->planner.numNodes, maxStretch_);
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
