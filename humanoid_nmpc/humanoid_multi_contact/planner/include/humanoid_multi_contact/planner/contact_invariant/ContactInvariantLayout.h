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

#include <string>
#include <vector>

namespace ocs2::humanoid::multi_contact {

/** One contact candidate of a contact-invariant problem: where its force or
 * wrench sits in the physical input. Passive data. */
struct ActivationCandidate {
  std::string name;
  // The index of the candidate's force in the physical input; a surface
  // candidate's moment follows its force.
  int forceOffset = 0;
  // 3 for a point candidate (a force), 6 for a surface candidate (a wrench).
  int wrenchDim = 0;
};

/**
 * How the state and input of a contact-invariant problem (design section 4.2)
 * extend those of a physical problem: the physical state, then one contact
 * variable c_i per candidate; the physical input, then one rate u_c_i per
 * candidate, with c_dot = u_c. Passive data.
 */
struct ContactInvariantLayout {
  int physicalStateDim = 0;
  int physicalInputDim = 0;
  std::vector<ActivationCandidate> candidates;

  int numCandidates() const { return static_cast<int>(candidates.size()); }
  int stateDim() const { return physicalStateDim + numCandidates(); }
  int inputDim() const { return physicalInputDim + numCandidates(); }
  /** The index of c_i in the augmented state. */
  int activationIndex(int candidate) const {
    return physicalStateDim + candidate;
  }
  /** The index of u_c_i in the augmented input. */
  int activationRateIndex(int candidate) const {
    return physicalInputDim + candidate;
  }
};

}  // namespace ocs2::humanoid::multi_contact
