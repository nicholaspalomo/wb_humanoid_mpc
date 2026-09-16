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

#include <chrono>
#include <functional>
#include <optional>
#include <vector>

#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/MixedIntegerOcpQp.h"
#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningContext.h"
#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningTerm.h"

namespace ocs2::humanoid {

/** Counters of one plan, filled by the branch-and-bound and the search stages. */
struct SearchStatistics {
  int numBranchAndBoundRelaxations = 0;
  int numLocalSearchQps = 0;
  int numHeadingRelinearizations = 0;
  int totalQpIterations = 0;
  scalar_t branchAndBoundTime = 0.0;
  scalar_t localSearchTime = 0.0;
  bool localSearchImproved = false;
};

/** What the stages that run before the branch-and-bound can read and set. */
struct SearchSetup {
  const ContactPlan* previousPlan = nullptr;
  const MiqpAssignment* previousAssignment = nullptr;
  int previousPlanShift = -1;  // -1: no usable previous plan
  int numNodes = 0;
  std::optional<MiqpAssignment> warmStart;  // complete assignment tried first for an incumbent (warm_start_previous_plan)
  MiqpSettings miqpSettings;                // diving is off unless the diving stage turns it on
};

/** What the stages that run after the branch-and-bound work on: the incumbent, the problem, the solver and the counters. */
struct SearchRun {
  using Clock = std::chrono::steady_clock;
  const ContactPlannerInput* input = nullptr;
  const ContactPlanningConfig* config = nullptr;
  const Layout* layout = nullptr;
  const std::vector<MiqpBinaryVariable>* binaries = nullptr;
  const MiqpAssignment* initialAssignment = nullptr;  // fixings that hold for the whole search (the committed schedule)
  const MiqpPropagateFn* propagate = nullptr;
  const MiqpAssignmentCostFn* assignmentCost = nullptr;
  MixedIntegerOcpQp* miqp = nullptr;
  OcpQpProblem* problem = nullptr;  // the problem the incumbent was found on; a stage that re-linearises replaces it
  MiqpResult* result = nullptr;
  SearchStatistics* statistics = nullptr;
  std::function<OcpQpProblem(const HeadingNominal&)> assembleWithNominal;  // re-builds the problem around a nominal
  Clock::time_point start;                                                 // start of the plan, for the time budgets
  bool verbose = false;

  scalar_t elapsedSeconds() const { return std::chrono::duration<scalar_t>(Clock::now() - start).count(); }
};

/**
 * A stage of the search around the branch-and-bound: something that provides an incumbent before it (a warm start, the
 * diving heuristic inside the solver) or refines the incumbent after it (local search, re-linearisation).
 */
class SearchStage : public ContactPlanningTerm {
 public:
  virtual void beforeSearch(SearchSetup& /*setup*/) const {}
  virtual void afterSearch(SearchRun& /*run*/) const {}
};

}  // namespace ocs2::humanoid
