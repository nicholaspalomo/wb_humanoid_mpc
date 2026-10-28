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
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_oc/oc_data/PerformanceIndex.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "ocs2_oc/synchronized_module/ReferenceManagerInterface.h"
#include "ocs2_sqp/SqpSettings.h"

#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureFunction.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactInvariantLayout.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactInvariantProblem.h"
#include "humanoid_multi_contact/planner/contact_invariant/TrajectoryInitializer.h"

namespace ocs2::humanoid::multi_contact {

/** One solve of the continuation: its weights and SQP iterations. Passive
 * data. */
struct ContinuationStage {
  std::string name;
  ContactInvariantStageWeights weights;
  int sqpIterations = 0;
};

/** What one solve found. Passive data. */
struct StageSolution {
  std::string stage;
  // The augmented state and input at the solver's nodes.
  GuessTrajectory trajectory;
  PerformanceIndex performance;
  int iterations = 0;
  double seconds = 0.0;
};

/** What the continuation solves over. Passive data; the pointers are
 * shared with the solver, which holds them by std::shared_ptr. */
struct ContinuationProblem {
  // The physical problem (copied for every solve) and the reference manager
  // its terms read.
  const OptimalControlProblem* absl_nonnull physical;
  std::shared_ptr<ReferenceManagerInterface> physicalReferenceManager;
  const ContactFeatureFunction* absl_nonnull features;
  ContactInvariantLayout layout;
  // The physical state's manifold with the contact variables (nullptr:
  // Euclidean).
  std::shared_ptr<const StateManifold> augmentedManifold;
  ContactInvariantBarriers barriers;
  sqp::Settings settings;
  // [s] The horizon, and the mode of the physical reference manager at its
  // start.
  scalar_t initTime = 0.0;
  scalar_t finalTime = 0.0;
  size_t initMode = 0;
};

/**
 * Solves the contact-invariant problem of `problem` once per stage of
 * `stages` (design section 4.2, "continuation"), each solve on the problem
 * of its weights (augmentProblem()) and warm-started from the last; the
 * first from `guess` (augmented states and inputs, its first state the
 * initial state). Returns every stage's solution.
 *
 * augmentProblem()'s errors, and Internal naming the stage when the SQP
 * fails (its QP has no solution).
 */
absl::StatusOr<std::vector<StageSolution>> solveContinuation(
    const ContinuationProblem& problem,
    const GuessTrajectory& guess,
    const std::vector<ContinuationStage>& stages);

}  // namespace ocs2::humanoid::multi_contact
