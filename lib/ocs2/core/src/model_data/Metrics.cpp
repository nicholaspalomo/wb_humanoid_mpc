/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

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

#include "ocs2_core/model_data/Metrics.h"

namespace ocs2 {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void Metrics::swap(Metrics& other) {
  // Cost
  std::swap(cost, other.cost);
  // Dynamics violation
  dynamicsViolation.swap(other.dynamicsViolation);
  // Equality constraints
  stateEqConstraint.swap(other.stateEqConstraint);
  stateInputEqConstraint.swap(other.stateInputEqConstraint);
  // Inequality constraints
  stateIneqConstraint.swap(other.stateIneqConstraint);
  stateInputIneqConstraint.swap(other.stateInputIneqConstraint);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void Metrics::clear() {
  // Cost
  cost = 0.0;
  // Dynamics violation
  dynamicsViolation = vector_t();
  // Equality constraints
  stateEqConstraint.clear();
  stateInputEqConstraint.clear();
  // Inequality constraints
  stateIneqConstraint.clear();
  stateInputIneqConstraint.clear();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector_array_t toConstraintArray(const size_array_t& termsSize, const vector_t& vec) {
  vector_array_t constraintArray;
  constraintArray.reserve(termsSize.size());

  size_t head = 0;
  for (const auto& l : termsSize) {
    constraintArray.emplace_back(vec.segment(head, l));
    head += l;
  }  // end of i loop

  return constraintArray;
}

}  // namespace ocs2
