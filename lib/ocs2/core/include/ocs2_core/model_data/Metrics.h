/******************************************************************************
Copyright (c) 2017, Farbod Farshidian. All rights reserved.

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

#include <algorithm>

#include "ocs2_core/Types.h"

namespace ocs2 {

/**
 * The collection of cost, dynamics violation and constraints in a point of time.
 *     cost : The total cost in a particular time point.
 *     dynamicsViolation : The vector of dynamics violation.
 *     stateEqConstraint : An array of all state equality constraints.
 *     stateInputEqConstraint : An array of all state-input equality constraints.
 *     stateIneqConstraint : An array of all state inequality constraints.
 *     stateInputIneqConstraint : An array of all state-input inequality constraints.
 */
struct Metrics {
  // Cost
  scalar_t cost;

  // Dynamics violation
  vector_t dynamicsViolation;

  // Equality constraints
  vector_array_t stateEqConstraint;
  vector_array_t stateInputEqConstraint;

  // Inequality constraints
  vector_array_t stateIneqConstraint;
  vector_array_t stateInputIneqConstraint;

  /** Exchanges the values of Metrics */
  void swap(Metrics& other);

  /** Clears the value of the Metrics */
  void clear();
};

/** Computes the sum of squared norm of a vector of equality constraints violation. */
inline scalar_t getEqConstraintsSSE(const vector_t& eqConstraint) {
  if (eqConstraint.size() == 0) {
    return 0.0;
  } else {
    return eqConstraint.squaredNorm();
  }
}

/** Computes the sum of squared norm of a vector of inequality constraints violation. */
inline scalar_t getIneqConstraintsSSE(const vector_t& ineqConstraint) {
  if (ineqConstraint.size() == 0) {
    return 0.0;
  } else {
    return ineqConstraint.cwiseMin(0.0).squaredNorm();
  }
}

/** Computes the sum of squared norm of an array of equality constraints violation. */
inline scalar_t getEqConstraintsSSE(const vector_array_t& eqConstraint) {
  scalar_t s = 0.0;
  std::for_each(eqConstraint.begin(), eqConstraint.end(), [&s](const vector_t& v) { s += getEqConstraintsSSE(v); });
  return s;
}

/** Computes the sum of squared norm of an array of inequality constraints violation. */
inline scalar_t getIneqConstraintsSSE(const vector_array_t& ineqConstraint) {
  scalar_t s = 0.0;
  std::for_each(ineqConstraint.begin(), ineqConstraint.end(), [&s](const vector_t& v) { s += getIneqConstraintsSSE(v); });
  return s;
}

/**
 * Deserializes the vector to an array of constraint terms.
 *
 * @param [in] termsSize : An array of constraint terms size. It as the same size as the output array.
 * @param [in] vec : Serialized array of constraint terms of the format :
 *                   (..., constraintArray[i], ...)
 * @return An array of constraint terms.
 */
vector_array_t toConstraintArray(const size_array_t& termsSize, const vector_t& vec);

}  // namespace ocs2
