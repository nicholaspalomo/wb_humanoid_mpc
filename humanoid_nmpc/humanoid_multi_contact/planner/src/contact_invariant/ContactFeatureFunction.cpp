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

#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureFunction.h"

#include <exception>
#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureRows.h"

namespace ocs2::humanoid::multi_contact {

ContactFeatureFunction::ContactFeatureFunction(
    std::unique_ptr<CppAdInterface> interface,
    int stateDim,
    int inputDim,
    int numCandidates)
    : interface_(std::move(interface)),
      stateDim_(stateDim),
      inputDim_(inputDim),
      numCandidates_(numCandidates) {}

ContactFeatureFunction::ContactFeatureFunction(
    const ContactFeatureFunction& other)
    : interface_(std::make_unique<CppAdInterface>(*other.interface_)),
      stateDim_(other.stateDim_),
      inputDim_(other.inputDim_),
      numCandidates_(other.numCandidates_) {}

absl::StatusOr<std::unique_ptr<ContactFeatureFunction>>
ContactFeatureFunction::Create(Function function,
                               int stateDim,
                               int inputDim,
                               int numCandidates,
                               const FeatureLibrarySettings& settings) {
  if (stateDim <= 0 || inputDim <= 0 || numCandidates < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("contact features '", settings.name,
                     "': the state and the input must have entries, got ",
                     stateDim, " and ", inputDim));
  }
  const int rows = numFeatureRows(numCandidates);
  CppAdInterface::ad_function_t taped = [function, stateDim, inputDim](
                                            const ad_vector_t& variables,
                                            ad_vector_t& features) {
    features = function(variables.head(stateDim),
                        variables.segment(stateDim, inputDim));
  };
  // CppAD and the code generator throw on a library that cannot be built or
  // loaded; converted at once.
  try {  // NOLINT(exceptions): CppAD's code generation throws.
    std::unique_ptr<CppAdInterface> interface =
        std::make_unique<CppAdInterface>(
            std::move(taped), static_cast<size_t>(stateDim + inputDim),
            settings.name, settings.folder);
    if (settings.recompile) {
      interface->createModels(CppAdInterface::ApproximationOrder::First,
                              settings.verbose);
    } else {
      interface->loadModelsIfAvailable(
          CppAdInterface::ApproximationOrder::First, settings.verbose);
    }
    const vector_t probe =
        interface->getFunctionValue(vector_t::Zero(stateDim + inputDim));
    if (probe.size() != rows) {
      return absl::InvalidArgumentError(absl::StrCat(
          "contact features '", settings.name, "': the function has ",
          probe.size(), " rows, ", numCandidates, " candidates need ", rows));
    }
    return absl::WrapUnique(new ContactFeatureFunction(
        std::move(interface), stateDim, inputDim, numCandidates));
  } catch (const std::exception& error) {  // NOLINT(exceptions): see above.
    return absl::InternalError(
        absl::StrCat("contact features '", settings.name, "': ", error.what()));
  }
}

vector_t ContactFeatureFunction::value(const vector_t& state,
                                       const vector_t& input) const {
  vector_t variables(stateDim_ + inputDim_);
  variables << state, input;
  return interface_->getFunctionValue(variables);
}

VectorFunctionLinearApproximation ContactFeatureFunction::linearApproximation(
    const vector_t& state, const vector_t& input) const {
  vector_t variables(stateDim_ + inputDim_);
  variables << state, input;
  VectorFunctionLinearApproximation result;
  result.f = interface_->getFunctionValue(variables);
  const matrix_t jacobian = interface_->getJacobian(variables);
  result.dfdx = jacobian.leftCols(stateDim_);
  result.dfdu = jacobian.rightCols(inputDim_);
  return result;
}

}  // namespace ocs2::humanoid::multi_contact
