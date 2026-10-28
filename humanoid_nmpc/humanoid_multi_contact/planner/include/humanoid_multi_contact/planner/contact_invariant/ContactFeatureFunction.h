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

#include <functional>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/automatic_differentiation/CppAdInterface.h"
#include "ocs2_core/automatic_differentiation/Types.h"

namespace ocs2::humanoid::multi_contact {

/** Where and how a ContactFeatureFunction's library is built. Passive data. */
struct FeatureLibrarySettings {
  // The library's name; it must carry everything recorded on the tape as a
  // constant (the robot, the candidates, the terrain's hash, the constants).
  std::string name;
  std::string folder;
  bool recompile = false;
  bool verbose = false;
};

/**
 * The contact features of a contact-invariant problem (ContactFeatureRows.h)
 * as a function of the physical state and input, recorded once on a CppAD
 * tape and compiled to a library: its value, and its Jacobian for the
 * Gauss-Newton approximations of the terms that read it.
 *
 * A value type (copies reload the library); not thread-safe, as CppAD's
 * evaluation of one library is not: the solver's workers each hold a copy.
 */
class ContactFeatureFunction {
 public:
  /** The features of a physical state and input: numFeatureRows() values. */
  using Function = std::function<ad_vector_t(const ad_vector_t& state,
                                             const ad_vector_t& input)>;

  /**
   * Tapes `function` of `numCandidates` candidates on a state of `stateDim`
   * and an input of `inputDim` entries, or loads the library `settings` names
   * when it exists. Internal with CppAD's message when the library cannot be
   * built or loaded.
   */
  static absl::StatusOr<std::unique_ptr<ContactFeatureFunction>> Create(
      Function function,
      int stateDim,
      int inputDim,
      int numCandidates,
      const FeatureLibrarySettings& settings);

  ContactFeatureFunction(const ContactFeatureFunction& other);
  ContactFeatureFunction& operator=(const ContactFeatureFunction&) = delete;
  ContactFeatureFunction(ContactFeatureFunction&&) = delete;
  ContactFeatureFunction& operator=(ContactFeatureFunction&&) = delete;
  ~ContactFeatureFunction() = default;

  /** The features at the physical `state` and `input`. */
  vector_t value(const vector_t& state, const vector_t& input) const;
  /** The features and their Jacobians with respect to the physical state and
   * input. */
  VectorFunctionLinearApproximation linearApproximation(
      const vector_t& state, const vector_t& input) const;

  int numCandidates() const { return numCandidates_; }
  int stateDim() const { return stateDim_; }
  int inputDim() const { return inputDim_; }

 private:
  ContactFeatureFunction(std::unique_ptr<CppAdInterface> interface,
                         int stateDim,
                         int inputDim,
                         int numCandidates);

  std::unique_ptr<CppAdInterface> interface_;
  int stateDim_ = 0;
  int inputDim_ = 0;
  int numCandidates_ = 0;
};

}  // namespace ocs2::humanoid::multi_contact
