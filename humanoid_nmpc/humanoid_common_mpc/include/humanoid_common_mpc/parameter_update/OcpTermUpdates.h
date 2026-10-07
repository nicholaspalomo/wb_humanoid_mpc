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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/misc/Collection.h"
#include "ocs2_core/penalties/augmented/AugmentedPenaltyBase.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"

/**
 * The building blocks every hot-field applier (HotFieldApplier.h) writes the running problem with: finding a term by
 * name, converting a block with its refusal reported by field, and the one place a reload catches an exception. Called
 * on the solver thread between two solves; nothing here is for the realtime thread.
 */
namespace ocs2::humanoid {

/**
 * Runs `call` and returns what it throws as an absl::Status carrying the exception's message: the one place the parameter
 * updater catches. A hot reload reaches code that reports its failures only by throwing - Collection::get, for a term of
 * another type; a penalty's setParameters, for a parameter vector of the wrong size - and nothing a saved file contains
 * may throw out of the solver's pre-solve hook.
 */
absl::Status exceptionsToStatus(absl::FunctionRef<void()> call);

/** Logs that the update of the term `name` failed with `status`, a WARNING naming the term. */
void reportFailedTermUpdate(absl::string_view name, const absl::Status& status);

/**
 * Logs that `what` of the reload of `source` was refused by its conversion, `status`, and that the running values are
 * kept: a WARNING naming the field.
 */
void reportNotApplied(absl::string_view source, absl::string_view what, const absl::Status& status);

/**
 * Applies `update` to the term `name` of `collection`, as a Derived. NotFound when the collection carries no such term,
 * which is the normal case for a term whose cost or constraint the running problem does not list; the error of
 * exceptionsToStatus() when the term is of another type (Collection::get throws std::bad_cast) or the update throws.
 */
template <typename Derived, typename Term>
absl::Status updateTerm(Collection<Term>& collection, const std::string& name, absl::FunctionRef<void(Derived&)> update) {
  size_t index = 0;
  if (!collection.getTermIndex(name, index)) {
    return absl::NotFoundError(absl::StrCat("the problem has no term ", name, "."));
  }
  return exceptionsToStatus([&]() { update(collection.template get<Derived>(name)); });
}

/**
 * updateTerm(), with a failed update logged as a WARNING naming the term (reportFailedTermUpdate()). A term the problem
 * does not carry is not a failure and is not logged: every reload would otherwise log one per foot per worker for each
 * unlisted term.
 */
template <typename Derived, typename Term>
void updateTermIfPresent(Collection<Term>& collection, const std::string& name, absl::FunctionRef<void(Derived&)> update) {
  const absl::Status status = updateTerm<Derived>(collection, name, update);
  if (!status.ok() && !absl::IsNotFound(status)) {
    reportFailedTermUpdate(name, status);
  }
}

/** Whether `collection` carries the term `name`. */
template <typename Term>
bool carriesTerm(const Collection<Term>& collection, const std::string& name) {
  return collection.getTermNameMap().contains(name);
}

/** Hands `parameters` to every penalty of `softConstraint`; PenaltyBaseWrapper delegates them to the penalty it wraps. */
template <typename SoftConstraint>
void setPenaltyParameters(SoftConstraint& softConstraint, const vector_t& parameters) {
  for (std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softConstraint.getPenalty().getPenaltyPtrArray()) {
    penalty->setParameters(parameters);
  }
}

/** The value of `converted`, or nullopt with its refusal reported (reportNotApplied()) as `what` of `source`. */
template <typename T>
std::optional<T> convertedOrReported(absl::StatusOr<T> converted, absl::string_view source, absl::string_view what) {
  if (!converted.ok()) {
    reportNotApplied(source, what, converted.status());
    return std::nullopt;
  }
  return *std::move(converted);
}

/**
 * Sets the weight of the soft term `<foot><termSuffix>` of every foot of `contactNames` in every problem of `problems`
 * (its QuadraticPenalty, wrapped inside a PenaltyBaseWrapper, whose setParameters() delegates to
 * QuadraticPenalty::setParameters()). A term the problem does not list has no penalty to scale, which is not a failure.
 * Precondition: `weight` is finite and positive, as footConstraintFromConfig() holds both soft weights of the foot
 * constraints to at start-up and on a reload alike.
 */
void setSoftTermWeight(std::vector<OptimalControlProblem>& problems,
                       const std::vector<std::string>& contactNames,
                       absl::string_view termSuffix,
                       scalar_t weight);

}  // namespace ocs2::humanoid
