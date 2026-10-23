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
#include <optional>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "ocs2_core/constraint/StateInputConstraintCollection.h"
#include "ocs2_oc/oc_data/ProblemMetrics.h"
#include "ocs2_oc/search_strategy/FilterLinesearch.h"
#include "ocs2_sqp/SqpSolverStatus.h"

#include "humanoid_mpc_validation/closed_loop/ClosedLoopMetrics.h"

/*
 * What a solve's SQP step is made of, for the closed-loop metrics (SolveSample::sqpStep) and the equivalence suites'
 * convergence tests: the line search's verdicts and constraint violation from the solver's iteration record
 * (SqpSolver::getIterationRecords()), and the largest residual of the hard swing-foot normal-velocity rows from the
 * values of the solution's nodes (SolverBase::getSolutionMetrics()). The functions read and allocate only what they
 * return; none is meant for the realtime thread.
 */

namespace ocs2::humanoid::validation {

/**
 * Returns the term indices, in `constraints` (an optimal control problem's equalityConstraintPtr), of the hard
 * normal-velocity constraints of `contactNames` (registered under contact_term::kNormalVelocity), in their order; a
 * contact without one is left out, so a problem without the hard rows gives none.
 */
std::vector<size_t> normalVelocityTermIndices(const StateInputConstraintCollection& constraints,
                                              absl::Span<const std::string> contactNames);

/**
 * Returns max |value| of the state-input equality terms `termIndices` over the intermediate nodes of `metrics`; empty
 * when none of them has a value there (a term is inactive at a node where its value is empty). An index past a node's
 * terms is skipped.
 */
std::optional<double> largestEqualityConstraintResidual(const ProblemMetrics& metrics, absl::Span<const size_t> termIndices);

/** Returns the SqpStepKind of the line-search verdict `stepType`. */
SqpStepKind sqpStepKindOf(FilterLinesearch::StepType stepType);

/**
 * Returns the sample of the SQP iteration `iteration`: its step size, the rules that accepted its step and judged its
 * full step, the full step's constraint violation and the constraint violation after the step; and the largest residual
 * of the normal-velocity terms `normalVelocityTerms` (normalVelocityTermIndices()) over `metrics`, the values of the
 * nodes the iteration's step reached.
 */
SqpStepSample sqpStepSampleOf(const sqp::IterationRecord& iteration,
                              const ProblemMetrics& metrics,
                              absl::Span<const size_t> normalVelocityTerms);

}  // namespace ocs2::humanoid::validation
