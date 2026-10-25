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

#include "humanoid_mpc_validation/closed_loop/SqpStepDiagnostics.h"

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/model_data/Metrics.h"

#include "humanoid_common_mpc/common/ContactTermNames.h"

namespace ocs2::humanoid::validation {

std::vector<size_t> normalVelocityTermIndices(
    const StateInputConstraintCollection& constraints,
    absl::Span<const std::string> contactNames) {
  std::vector<size_t> indices;
  for (const std::string& contactName : contactNames) {
    size_t index = 0;
    if (constraints.getTermIndex(
            contact_term::name(contactName, contact_term::kNormalVelocity),
            index))
      indices.push_back(index);
  }
  return indices;
}

std::optional<double> largestEqualityConstraintResidual(
    const ProblemMetrics& metrics, absl::Span<const size_t> termIndices) {
  std::optional<double> largest;
  for (const Metrics& node : metrics.intermediates) {
    for (const size_t term : termIndices) {
      if (term >= node.stateInputEqConstraint.size()) continue;
      const vector_t& value = node.stateInputEqConstraint[term];
      if (value.size() == 0) continue;
      const double residual = value.cwiseAbs().maxCoeff();
      largest = largest.has_value() ? std::max(*largest, residual) : residual;
    }
  }
  return largest;
}

SqpStepKind sqpStepKindOf(FilterLinesearch::StepType stepType) {
  switch (stepType) {
    case FilterLinesearch::StepType::UNKNOWN:
      return SqpStepKind::kUnknown;
    case FilterLinesearch::StepType::CONSTRAINT:
      return SqpStepKind::kConstraint;
    case FilterLinesearch::StepType::DUAL:
      return SqpStepKind::kDual;
    case FilterLinesearch::StepType::COST:
      return SqpStepKind::kCost;
    case FilterLinesearch::StepType::ZERO:
      return SqpStepKind::kZero;
  }
  return SqpStepKind::kUnknown;  // Not a value of the enum.
}

SqpStepSample sqpStepSampleOf(const sqp::IterationRecord& iteration,
                              const ProblemMetrics& metrics,
                              absl::Span<const size_t> normalVelocityTerms) {
  const sqp::StepInfo& step = iteration.stepInfo;
  return SqpStepSample{
      .stepSize = step.stepSize,
      .stepKind = sqpStepKindOf(step.stepType),
      .fullStepKind = sqpStepKindOf(step.fullStepType),
      .fullStepConstraintViolation = step.fullStepConstraintViolation,
      .constraintViolation = step.totalConstraintViolationAfterStep,
      .swingNormalVelocityResidual =
          largestEqualityConstraintResidual(metrics, normalVelocityTerms)};
}

}  // namespace ocs2::humanoid::validation
