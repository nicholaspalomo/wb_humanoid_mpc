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

#include <ocs2_core/Types.h>
#include <ocs2_core/manifold/StateManifold.h>

namespace ocs2 {

/**
 * The flat state space R^n as a StateManifold: (+) is addition, (-) subtraction, every derivative map the identity.
 * A solver given nullptr instead keeps its original flat code path; this class exists for composing and for testing the
 * manifold code paths against the flat ones.
 */
class EuclideanStateManifold final : public StateManifold {
 public:
  explicit EuclideanStateManifold(size_t dimension) : dimension_(dimension) {}
  ~EuclideanStateManifold() override = default;

  size_t getAmbientDim() const override { return dimension_; }
  size_t getTangentDim() const override { return dimension_; }

  void retract(const vector_t& x, const vector_t& dx, scalar_t alpha, vector_t& xNew) const override;
  vector_t difference(const vector_t& x0, const vector_t& x1) const override;
  vector_t interpolate(const vector_t& x0, const vector_t& x1, scalar_t alpha) const override;
  void project(vector_t& x) const override;
  scalar_t getMaximumRotationAngle(const vector_t& dx) const override;
  void pullBackStateColumns(const vector_t& x, matrix_t& J) const override;
  void pullBackStateRows(const vector_t& x, matrix_t& M) const override;
  void pullBackGradient(const vector_t& x, vector_t& g) const override;
  void liftStateColumns(const vector_t& x, matrix_t& J) const override;
  void pushForwardDynamics(const vector_t& x, const vector_t& xNext, VectorFunctionLinearApproximation& dynamics) const override;

 private:
  size_t dimension_;
};

}  // namespace ocs2
